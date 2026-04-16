from __future__ import annotations

import inspect
import re
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from miniclaw.observability.safe import safe_finish_span, safe_start_span
from miniclaw.utils.async_bridge import run_sync as _run_provider_sync
from miniclaw.utils.jsonx import safe_loads_with_raw

from miniclaw.memory import build_memory_context
from miniclaw.providers.contracts import ChatMessage, ChatProvider, ChatResponse
from miniclaw.prompting import ContextBuilder
from miniclaw.runtime.state import ActiveCapabilities, RuntimeMessage, RuntimeState, RuntimeUsage
from miniclaw.runtime.tool_loop import (
    apply_tool_calls,
    coerce_active_capabilities,
    format_error,
    parse_tool_arguments,
    resolve_turn_trace as _resolve_turn_trace,
    runtime_message_from_chat,
    trace_messages as _trace_messages,
    trace_tool_calls as _trace_tool_calls,
    trace_tool_names as _trace_tool_names,
)
from miniclaw.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from miniclaw.config.settings import Settings
    from miniclaw.memory.indexer import MemoryIndexer
    from miniclaw.observability.contracts import Tracer

MAX_TOOL_ROUNDS = 16
MAX_CONSECUTIVE_ERRORS = 4

_CLASSIFY_SYSTEM_PROMPT = (
    "你是一个意图分类器。根据用户消息和最近一轮上下文判断意图类型。\n\n"
    "规则：\n"
    "- clarify: 信息不足以执行任何操作（缺少对象、动作不明确，且上下文中也无法推断）\n"
    "- simple: 可以直接执行的请求（查询、闲聊、单步操作、延续上一轮的指令）\n"
    "- planned: 需要多步规划的复杂任务（实现功能、重构、调试）"
)

_CLASSIFY_TOOL = [{
    "type": "function",
    "function": {
        "name": "classify_intent",
        "description": "Classify user intent.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["clarify", "simple", "planned"]},
                "reason": {"type": "string", "description": "Why (for clarify: what info is missing)"},
            },
            "required": ["intent"],
        },
    },
}]

_RULE_PLANNED_MARKERS = ("implement", "build", "refactor", "fix", "plan", "design", "update tests", "migrate", "extend")
_RULE_SIMPLE_MARKERS = ("read ", "open ", "list ", "search ", "find ", "inspect ", "file", "tool",
                        "load skill", "load the skill", "activate skill", "enable skill",
                        "load mcp", "activate mcp", "enable mcp")

from miniclaw.runtime.subagent import ROLE_DEFAULTS

_PLANNER_SYSTEM_PROMPT = (
    "You are a planner. Decide whether the user's request needs to be split into "
    "subagent dispatches, and if so, produce a brief for each subagent.\n\n"
    "Rules:\n"
    "- Each brief has role (researcher/executor/reviewer or a custom role) and task.\n"
    "- A custom role must come with an explicit tools list.\n"
    "- Mark dependencies via depends_on (indices of prior briefs that must finish first).\n"
    "- Empty subagent_briefs is valid: it tells the main agent to execute directly.\n"
    "- Maximum 8 briefs.\n"
)

_PLAN_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit a lightweight plan with optional subagent dispatches.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "subagent_briefs": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "task": {"type": "string"},
                            "expected_output": {"type": "string"},
                            "tools": {"type": "array", "items": {"type": "string"}},
                            "depends_on": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["role", "task"],
                    },
                },
                "executor_notes": {"type": "string"},
            },
            "required": ["summary"],
        },
    },
}]


def _validate_plan(plan: dict[str, Any]) -> None:
    briefs = plan.get("subagent_briefs", []) or []
    if not isinstance(briefs, list):
        raise ValueError("subagent_briefs must be a list")
    if len(briefs) > 8:
        raise ValueError(f"too many briefs: {len(briefs)}")
    for i, brief in enumerate(briefs):
        if not isinstance(brief, dict):
            raise ValueError(f"brief {i} must be an object")
        role = str(brief.get("role", "")).strip()
        task = str(brief.get("task", "")).strip()
        if not role or not task:
            raise ValueError(f"brief {i} missing role or task")
        if role not in ROLE_DEFAULTS:
            tools = brief.get("tools")
            if not isinstance(tools, list) or not tools:
                raise ValueError(f"custom role '{role}' requires explicit tools list")


def _generate_plan(
    provider: ChatProvider,
    state: RuntimeState,
    tool_registry: ToolRegistry | None,
) -> dict[str, Any]:
    user_input = str(state.get("user_input", "")).strip()[:500]
    memory_context = str(state.get("memory_context", "")).strip()[:2000]
    user_message = f"Request: {user_input}"
    if memory_context:
        user_message += f"\n\nMemory context:\n{memory_context}"

    messages = [
        ChatMessage(role="system", content=_PLANNER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_message),
    ]
    response = _run_provider_sync(provider.achat(messages, tools=_PLAN_TOOL))
    if not response.tool_calls:
        raise RuntimeError("planner did not return submit_plan call")
    function = response.tool_calls[0].get("function", {})
    arguments = parse_tool_arguments("submit_plan", function.get("arguments", {}))
    _validate_plan(arguments)
    return arguments


def ingest(state: RuntimeState) -> RuntimeState:
    messages = list(state.get("messages", []))
    user_message: RuntimeMessage = {"role": "user", "content": state["user_input"]}
    user_content_parts = state.get("user_content_parts", [])
    if isinstance(user_content_parts, list) and user_content_parts:
        user_message["content_parts"] = [dict(part) for part in user_content_parts if isinstance(part, dict)]
    messages.append(user_message)
    return {"messages": messages}


def make_classify(
    mini_provider: ChatProvider | None = None,
    main_provider: ChatProvider | None = None,
) -> Callable[[RuntimeState], RuntimeState]:
    def classify(state: RuntimeState) -> RuntimeState:
        user_input = str(state.get("user_input", "")).strip()
        messages = list(state.get("messages", []))
        last_turn = _extract_last_turn_summary(messages)

        provider = mini_provider or main_provider
        if provider is None:
            return _rule_based_classify(user_input)

        result = _call_classify(provider, user_input, last_turn)

        if result is None and main_provider is not None and provider is not main_provider:
            result = _call_classify(main_provider, user_input, last_turn)

        if result is None:
            return _rule_based_classify(user_input)

        return _build_classify_state(result)

    return classify


def _call_classify(provider: ChatProvider, user_input: str, last_turn: str) -> dict[str, str] | None:
    user_msg = f"用户消息: {user_input[:500]}"
    if last_turn:
        user_msg += f"\n\n{last_turn}"
    try:
        response = _run_provider_sync(provider.achat(
            [ChatMessage(role="system", content=_CLASSIFY_SYSTEM_PROMPT),
             ChatMessage(role="user", content=user_msg)],
            tools=_CLASSIFY_TOOL,
        ))
        if not response.tool_calls:
            return None
        raw_call = response.tool_calls[0]
        function = raw_call.get("function", {})
        arguments = parse_tool_arguments("classify_intent", function.get("arguments", {}))
        intent = str(arguments.get("intent", "simple")).strip()
        if intent not in ("clarify", "simple", "planned"):
            intent = "simple"
        return {"intent": intent, "reason": str(arguments.get("reason", ""))}
    except Exception:
        return None


def _extract_last_turn_summary(messages: list) -> str:
    last_user = ""
    last_assistant = ""
    for msg in reversed(messages):
        role = msg.get("role", "") if isinstance(msg, dict) else msg.role
        content = msg.get("content", "") if isinstance(msg, dict) else msg.content
        if role == "assistant" and not last_assistant:
            last_assistant = str(content)[:200]
        elif role == "user" and not last_user:
            last_user = str(content)[:200]
        if last_user and last_assistant:
            break
    if not last_user:
        return ""
    parts = [f"上一轮用户说: {last_user}"]
    if last_assistant:
        parts.append(f"助手回复: {last_assistant}")
    return "\n".join(parts)


def _build_classify_state(result: dict) -> RuntimeState:
    intent = result.get("intent", "simple")
    return {
        "route": intent,
        "needs_clarification": intent == "clarify",
        "clarification_reason": result.get("reason", ""),
        "planner_context": "",
        "plan_summary": "",
        "subagent_briefs": [],
        "executor_notes": "",
    }


def _rule_based_classify(user_input: str) -> RuntimeState:
    normalized = re.sub(r"\s+", " ", user_input.strip().lower())
    if any(marker in normalized for marker in _RULE_PLANNED_MARKERS):
        return _build_classify_state({"intent": "planned"})
    return _build_classify_state({"intent": "simple"})


def make_load_context(
    memory_store: object,
    *,
    retriever: object | None = None,
    indexer: MemoryIndexer | None = None,
    memory_token_budget: int = 2000,
    mini_provider: ChatProvider | None = None,
    main_provider: ChatProvider | None = None,
    settings: object | None = None,
    tracer: object | None = None,
) -> Callable[[RuntimeState], RuntimeState]:
    def load_context(state: RuntimeState) -> RuntimeState:
        # Flush dirty index entries synchronously
        if indexer is not None:
            _run_provider_sync(indexer.flush_dirty())

        runtime_metadata = state.get("runtime_metadata") or {}
        thread_id = _resolve_thread_id(state, runtime_metadata)
        user_input = str(state.get("user_input", "")).strip()
        parent_trace = _resolve_turn_trace(state, "graph.load_context") if tracer is not None else None

        rewrite_result = None
        if (
            settings is not None
            and getattr(settings, "memory_rewrite_enabled", True)
            and user_input
        ):
            from miniclaw.memory.rewrite import RewriteInput, rewrite_query
            provider = _select_rewrite_provider(settings, mini_provider, main_provider)
            if provider is not None:
                recent = _extract_recent_exchanges(
                    state,
                    n=int(getattr(settings, "memory_rewrite_recent_exchanges", 2) or 2),
                )
                rewrite_inputs = RewriteInput(
                    user_input=user_input,
                    recent_exchanges=recent,
                )
                rewrite_span = safe_start_span(
                    tracer, parent_trace, name="memory.rewrite",
                    metadata={"model_tier": str(getattr(settings, "memory_rewrite_model_tier", "auto"))},
                )
                try:
                    rewrite_result = _run_provider_sync(
                        rewrite_query(
                            rewrite_inputs,
                            provider=provider,
                            model=_resolve_rewrite_model(settings),
                            timeout_s=float(getattr(settings, "memory_rewrite_timeout_s", 1.0) or 1.0),
                        )
                    )
                except Exception:
                    rewrite_result = None
                safe_finish_span(
                    tracer, rewrite_span,
                    status="ok" if (rewrite_result and rewrite_result.used_llm) else "fallback",
                    outputs=(
                        {
                            "used_llm": rewrite_result.used_llm,
                            "intent": rewrite_result.intent,
                            "latency_ms": rewrite_result.latency_ms,
                            "failure_reason": rewrite_result.failure_reason,
                            "keywords_count": len(rewrite_result.keywords),
                        }
                        if rewrite_result is not None
                        else {"used_llm": False, "failure_reason": "exception"}
                    ),
                )

        memory_context = ""
        if thread_id:
            retrieve_span = safe_start_span(
                tracer, parent_trace, name="memory.retrieve",
                metadata={
                    "thread_id": thread_id,
                    "intent": rewrite_result.intent if rewrite_result else "ambiguous",
                },
            )
            memory_context = build_memory_context(
                memory_store,
                thread_id,
                retriever=retriever,
                user_input=user_input,
                memory_token_budget=memory_token_budget,
                rewrite=rewrite_result,
            )
            safe_finish_span(
                tracer, retrieve_span,
                status="ok",
                outputs={"memory_context_chars": len(memory_context)},
            )

        planner_context = ""
        if str(state.get("route", "")).strip() == "planned":
            planner_context = _format_planner_context(state)

        return {
            "memory_context": memory_context,
            "planner_context": planner_context,
        }

    return load_context


def _select_rewrite_provider(
    settings: object,
    mini_provider: ChatProvider | None,
    main_provider: ChatProvider | None,
) -> ChatProvider | None:
    tier = str(getattr(settings, "memory_rewrite_model_tier", "auto") or "auto")
    if tier == "mini":
        return mini_provider
    if tier == "main":
        return main_provider
    return mini_provider or main_provider


def _resolve_rewrite_model(settings: object) -> str:
    tier = str(getattr(settings, "memory_rewrite_model_tier", "auto") or "auto")
    if tier == "main":
        return str(getattr(settings, "model", "") or "")
    mini = getattr(settings, "mini_model", None)
    if mini:
        return str(mini)
    return str(getattr(settings, "model", "") or "")


def _extract_recent_exchanges(
    state: RuntimeState,
    *,
    n: int,
) -> tuple[tuple[str, str], ...]:
    """Pull the last n (user, assistant) pairs from state.messages."""
    messages = state.get("messages") or []
    pairs: list[tuple[str, str]] = []
    user_buffer: str | None = None
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "") or ""
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", "") or ""
        if role == "user":
            user_buffer = str(content)
        elif role == "assistant" and user_buffer is not None:
            pairs.append((user_buffer, str(content)))
            user_buffer = None
    return tuple(pairs[-n:])


def make_planner(
    provider: ChatProvider | None = None,
    tool_registry: ToolRegistry | None = None,
) -> Callable[[RuntimeState], RuntimeState]:
    def planner(state: RuntimeState) -> RuntimeState:
        if state.get("route") != "planned":
            return {}
        if provider is None:
            return {"plan_summary": "", "subagent_briefs": [], "executor_notes": ""}
        try:
            plan = _generate_plan(provider, state, tool_registry)
        except ValueError as exc:
            return {"last_error": f"planner validation failed: {exc}"}
        except Exception as exc:
            return {"last_error": f"planner failed: {exc}"}
        return {
            "plan_summary": str(plan.get("summary", "")),
            "subagent_briefs": list(plan.get("subagent_briefs", []) or []),
            "executor_notes": str(plan.get("executor_notes", "")),
        }

    return planner


def make_agent(
    settings: Settings,
    provider: ChatProvider,
    tool_registry: ToolRegistry | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    tracer: "Tracer | None" = None,
) -> Callable[[RuntimeState], RuntimeState]:
    from miniclaw.observability.contracts import NoopTracer, build_run_context
    resolved_tracer = tracer or NoopTracer()

    context_builder = ContextBuilder(
        system_prompt=settings.system_prompt,
        skills_loader=tool_registry.skill_loader if tool_registry is not None else None,
        tool_registry=tool_registry,
        mcp_registry=tool_registry.mcp_registry if tool_registry is not None else None,
        history_char_budget=settings.history_char_budget,
        max_history_messages=settings.max_history_messages,
    )

    max_rounds = settings.max_tool_rounds
    max_errors = settings.max_consecutive_tool_errors
    max_result_chars = settings.max_tool_result_chars

    def _emit(kind: str, **kwargs: Any) -> None:
        if on_event is not None:
            on_event({"kind": kind, **kwargs})

    def agent(state: RuntimeState) -> RuntimeState:
        loop_state = dict(state)
        loop_state["active_capabilities"] = coerce_active_capabilities(state.get("active_capabilities"))
        usage: RuntimeUsage = {}

        # Reuse the turn-level trace context (set by the service layer via
        # runtime_metadata._turn_trace_id / _turn_run_id) so every span in
        # this agent invocation — round spans, provider.achat, tool.<name>,
        # subagent.run — shares the same trace_id as the outer graph nodes.
        # Fall back to a fresh run context only when invoked outside the
        # normal service path (e.g. isolated tests).
        parent_trace = _resolve_turn_trace(loop_state, "graph.agent")

        consecutive_errors = 0
        try:
            for round_idx in range(max_rounds):
                # Build messages/tools BEFORE opening the round span so we can
                # record them as span inputs. Failures here still get traced as
                # an error on a minimal span so there's no silent loss.
                try:
                    messages = context_builder.build_provider_messages(loop_state)
                    visible_tools = _build_provider_tools(tool_registry, loop_state["active_capabilities"])
                except Exception:
                    safe_start_span(resolved_tracer, parent_trace, name=f"agent.tool_loop.round_{round_idx}")
                    # finish immediately — parent chain stays balanced
                    raise

                round_span = safe_start_span(
                    resolved_tracer,
                    parent_trace,
                    name=f"agent.tool_loop.round_{round_idx}",
                    inputs={
                        "messages": _trace_messages(messages),
                        "tools": _trace_tool_names(visible_tools),
                        "model": settings.model,
                    },
                )
                round_status = "ok"
                round_outputs: dict[str, Any] = {}
                try:
                    chat_span = safe_start_span(
                        resolved_tracer,
                        round_span,
                        name="provider.achat",
                        metadata={"provider.model": settings.model},
                        inputs={
                            "messages": _trace_messages(messages),
                            "tool_count": len(visible_tools or []),
                        },
                    )
                    try:
                        response = _run_provider_sync(
                            _invoke_provider(
                                provider,
                                messages,
                                model=settings.model,
                                tools=visible_tools,
                            )
                        )
                    except Exception:
                        safe_finish_span(resolved_tracer, chat_span, status="error")
                        round_status = "error"
                        raise
                    safe_finish_span(
                        resolved_tracer,
                        chat_span,
                        status="ok",
                        metadata={"provider.usage": dict(getattr(response, "usage", {}) or {})},
                        outputs={
                            "content": (response.content or "")[:max_result_chars],
                            "tool_calls": _trace_tool_calls(response.tool_calls),
                            "usage": dict(getattr(response, "usage", {}) or {}),
                        },
                    )
                    round_outputs = {
                        "content": (response.content or "")[:max_result_chars],
                        "tool_calls": _trace_tool_calls(response.tool_calls),
                        "usage": dict(getattr(response, "usage", {}) or {}),
                    }

                    usage = _merge_usage(usage, response)
                    if not response.tool_calls:
                        final = _finish_response(loop_state, response, usage)
                        _emit("chunk", text=str(final.get("response_text", "")))
                        return final  # round_span closed via finally below

                    model_text = str(response.content or "").strip()
                    if model_text:
                        _emit("model_text", text=model_text)

                    # Mint a fleet_id when this round contains spawn_subagent calls.
                    # All spawns in the same round share the same fleet_id.
                    spawn_count = sum(
                        1 for c in response.tool_calls
                        if c.get("function", {}).get("name") == "spawn_subagent"
                    )
                    if spawn_count > 0:
                        fleet_id = f"fleet-{uuid.uuid4().hex[:6]}"
                        rt_meta = dict(loop_state.get("runtime_metadata") or {})
                        rt_meta["current_fleet_id"] = fleet_id
                        rt_meta["_sub_index"] = 0
                        if on_event is not None:
                            rt_meta["_on_event"] = on_event
                        rt_meta["_parent_trace"] = round_span  # round span becomes parent for subagents
                        loop_state = {**loop_state, "runtime_metadata": rt_meta}

                    for raw_call in response.tool_calls:
                        fn = raw_call.get("function", {})
                        tool_name = str(fn.get("name", ""))
                        tool_args = safe_loads_with_raw(fn.get("arguments", {}))
                        _emit("tool_calling", text=tool_name, tool_name=tool_name, arguments=tool_args)

                    loop_state = apply_tool_calls(
                        loop_state,
                        response,
                        tool_registry,
                        max_result_chars=max_result_chars,
                        tracer=resolved_tracer,
                        parent_trace=round_span,
                    )

                    last_tool_messages = [
                        msg for msg in loop_state.get("messages", [])
                        if (msg.get("role") if isinstance(msg, dict) else msg.role) == "tool"
                    ]
                    recent_tool_msgs = last_tool_messages[-len(response.tool_calls):] if response.tool_calls else []
                    for i, raw_call in enumerate(response.tool_calls):
                        fn = raw_call.get("function", {})
                        tool_name = str(fn.get("name", ""))
                        tool_args = safe_loads_with_raw(fn.get("arguments", {}))
                        result_text = ""
                        if i < len(recent_tool_msgs):
                            msg = recent_tool_msgs[i]
                            result_text = str(msg.get("content", "") if isinstance(msg, dict) else msg.content)
                        _emit("tool_done", text=tool_name, tool_name=tool_name, arguments=tool_args, result=result_text[:500])

                    if last_tool_messages and _is_error_content(last_tool_messages[-1]):
                        consecutive_errors += 1
                        if consecutive_errors >= max_errors:
                            round_status = "error"
                            return _error_state(
                                loop_state,
                                f"tool call failed {max_errors} consecutive times",
                                usage,
                            )
                    else:
                        consecutive_errors = 0
                except Exception:
                    round_status = "error"
                    raise
                finally:
                    safe_finish_span(resolved_tracer, round_span, status=round_status, outputs=round_outputs)
        except Exception as exc:
            return _error_state(loop_state, format_error(exc), usage)

        return _error_state(
            loop_state,
            f"tool loop round limit reached after {max_rounds} rounds",
            usage,
        )

    return agent



def complete(state: RuntimeState) -> RuntimeState:
    return {
        "response_text": state.get("response_text", ""),
        "last_error": state.get("last_error", ""),
        "usage": state.get("usage", {}),
        "messages": list(state.get("messages", [])),
    }


def clarify(state: RuntimeState) -> RuntimeState:
    reason = str(state.get("clarification_reason", "")).strip()
    user_input = str(state.get("user_input", "")).strip()
    if reason:
        response = f"我需要更多信息来帮助你：{reason}\n\n你能具体说明一下吗？"
    else:
        response = f"你说的「{user_input}」具体是指什么？能再详细描述一下吗？"
    return {"response_text": response, "last_error": ""}



def error_handler(state: RuntimeState) -> RuntimeState:
    error = str(state.get("last_error", ""))
    if "planner failed" in error:
        response = f"任务规划失败，请重试或简化你的请求。\n\n错误详情：{error}"
    elif "planner validation" in error:
        response = f"规划结果不符合要求，请尝试更具体地描述你的需求。\n\n错误详情：{error}"
    else:
        response = f"处理过程中出现错误：{error}"
    return {"response_text": response, "last_error": error}


def route_on_error(target_if_ok: str) -> Callable[[RuntimeState], str]:
    def _route(state: RuntimeState) -> str:
        if str(state.get("last_error", "") or "").strip():
            return "error_handler"
        return target_if_ok
    return _route


def route_after_classify(state: RuntimeState) -> str:
    if str(state.get("last_error", "") or "").strip():
        return "error_handler"
    if state.get("needs_clarification"):
        return "clarify"
    return "load_context"


def route_after_load_context(state: RuntimeState) -> str:
    if str(state.get("last_error", "") or "").strip():
        return "error_handler"
    if str(state.get("route", "")).strip() == "planned":
        return "planner"
    return "agent"


def route_after_planner(state: RuntimeState) -> str:
    if str(state.get("last_error", "") or "").strip():
        return "error_handler"
    return "agent"


# _run_provider_sync imported from miniclaw.utils.async_bridge


def _invoke_provider(
    provider: ChatProvider,
    messages: list[ChatMessage],
    *,
    model: str | None,
    tools: list[dict[str, Any]],
):
    kwargs: dict[str, Any] = {"model": model}
    if tools and _provider_accepts_tools(provider):
        kwargs["tools"] = tools
    return provider.achat(messages, **kwargs)


def _provider_accepts_tools(provider: ChatProvider) -> bool:
    try:
        signature = inspect.signature(provider.achat)
    except (TypeError, ValueError):
        return False

    if "tools" in signature.parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def _build_provider_tools(
    tool_registry: ToolRegistry | None,
    active_capabilities: ActiveCapabilities,
) -> list[dict[str, Any]]:
    if tool_registry is None:
        return []

    tools: list[dict[str, Any]] = []
    for spec in tool_registry.list_visible_tools(active_capabilities):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": dict(spec.input_schema),
                },
            }
        )
    return tools


def _is_error_content(message: RuntimeMessage | dict[str, Any]) -> bool:
    content = message.get("content", "") if isinstance(message, dict) else message.content
    return str(content or "").startswith("ERROR: ")




def _finish_response(
    state: RuntimeState,
    response: ChatResponse,
    usage: RuntimeUsage,
) -> RuntimeState:
    updated_messages = list(state.get("messages", []))
    updated_messages.append(runtime_message_from_chat(_assistant_response_message(response)))
    return {
        "messages": updated_messages,
        "response_text": response.content,
        "last_error": "",
        "usage": usage,
        "active_capabilities": coerce_active_capabilities(state.get("active_capabilities")),
    }


def _error_state(
    state: RuntimeState,
    message: str,
    usage: RuntimeUsage,
) -> RuntimeState:
    previous_messages = list(state.get("messages", []))
    if previous_messages and previous_messages[-1]["role"] == "user":
        previous_messages = previous_messages[:-1]
    return {
        "messages": previous_messages,
        "response_text": "",
        "last_error": message,
        "usage": usage,
        "active_capabilities": coerce_active_capabilities(state.get("active_capabilities")),
    }


def _assistant_response_message(response: ChatResponse) -> ChatMessage:
    if response.content_parts:
        return ChatMessage(role="assistant", content_parts=[dict(part) for part in response.content_parts])
    return ChatMessage(role="assistant", content=response.content)


def _merge_usage(current: RuntimeUsage, response: ChatResponse) -> RuntimeUsage:
    if response.usage is None:
        return current

    usage = dict(current)
    if response.usage.prompt_tokens is not None:
        usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + response.usage.prompt_tokens
    if response.usage.completion_tokens is not None:
        usage["completion_tokens"] = usage.get("completion_tokens", 0) + response.usage.completion_tokens
    if response.usage.total_tokens is not None:
        usage["total_tokens"] = usage.get("total_tokens", 0) + response.usage.total_tokens
    return usage


def _resolve_thread_id(state: RuntimeState, runtime_metadata: object) -> str:
    if isinstance(runtime_metadata, dict):
        metadata_thread_id = str(runtime_metadata.get("thread_id", "")).strip()
        if metadata_thread_id:
            return metadata_thread_id
    return str(state.get("thread_id", "")).strip()


def _format_planner_context(state: RuntimeState) -> str:
    user_request = str(state.get("user_input", "")).strip()
    return f"Planner context:\nUser request: {user_request}" if user_request else "Planner context:\nUser request: "



