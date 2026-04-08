from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from miniclaw.utils.async_bridge import run_sync as _run_provider_sync

from miniclaw.memory import build_memory_context
from miniclaw.providers.contracts import ChatMessage, ChatProvider, ChatResponse
from miniclaw.prompting import ContextBuilder
from miniclaw.runtime.state import ActiveCapabilities, RuntimeMessage, RuntimeState, RuntimeUsage
from miniclaw.runtime.workers import WorkerManager
from miniclaw.tools.contracts import ToolCall, ToolResult
from miniclaw.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from miniclaw.config.settings import Settings
    from miniclaw.memory.indexer import MemoryIndexer

MAX_TOOL_ROUNDS = 16
MAX_CONSECUTIVE_ERRORS = 4
MAX_TOOL_RESULT_CHARS = 16_000

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

_PLANNER_SYSTEM_PROMPT = (
    "You are a task planner. Break down the user's request into concrete tasks.\n\n"
    "Rules:\n"
    "- Each task has a kind (research/execution/review) and worker_role (researcher/executor/reviewer)\n"
    "- Tasks in the same parallel_group run concurrently; different groups run sequentially\n"
    "- Standard flow: research first, then execution, then review\n"
    "- Simple requests may need only 1-2 tasks; don't over-plan\n"
    "- At least one task must be kind=execution\n"
    "- Max 8 tasks total\n"
    "- executor_notes: brief instructions for the agent that will execute the overall plan\n\n"
    "Available tools the workers can use:\n{tool_names}"
)

_PLAN_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "Submit a structured execution plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One-sentence summary of the plan",
                    },
                    "tasks": {
                        "type": "array",
                        "description": "Ordered list of tasks (max 8)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "kind": {
                                    "type": "string",
                                    "enum": ["research", "execution", "review"],
                                },
                                "worker_role": {
                                    "type": "string",
                                    "enum": ["researcher", "executor", "reviewer"],
                                },
                                "parallel_group": {
                                    "type": "string",
                                    "description": "Tasks in the same group run in parallel",
                                },
                            },
                            "required": ["title", "kind", "worker_role", "parallel_group"],
                        },
                        "minItems": 1,
                        "maxItems": 8,
                    },
                    "executor_notes": {
                        "type": "string",
                        "description": "Instructions for the executor agent",
                    },
                },
                "required": ["summary", "tasks"],
            },
        },
    }
]

_MAX_PLAN_TASKS = 8


def _generate_plan(
    provider: ChatProvider,
    state: RuntimeState,
    tool_registry: ToolRegistry | None,
) -> dict[str, Any]:
    tool_names = _get_planner_tool_names(tool_registry, state)
    system_prompt = _PLANNER_SYSTEM_PROMPT.format(tool_names=", ".join(tool_names))

    user_input = str(state.get("user_input", "")).strip()
    memory_context = str(state.get("memory_context", "")).strip()

    user_message = f"Request: {user_input}"
    if memory_context:
        user_message += f"\n\nContext:\n{memory_context}"

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_message),
    ]

    response = _run_provider_sync(provider.achat(messages, tools=_PLAN_TOOL))

    if not response.tool_calls:
        raise RuntimeError("planner did not return a plan via tool call")

    raw_call = response.tool_calls[0]
    function = raw_call.get("function", {})
    arguments = _parse_tool_arguments("submit_plan", function.get("arguments", {}))
    _validate_plan(arguments)
    return arguments


def _validate_plan(plan: dict[str, Any]) -> None:
    tasks = plan.get("tasks", [])
    if not tasks:
        raise ValueError("plan has no tasks")
    if len(tasks) > _MAX_PLAN_TASKS:
        raise ValueError(f"plan has {len(tasks)} tasks, max is {_MAX_PLAN_TASKS}")
    has_execution = any(t.get("kind") == "execution" for t in tasks)
    if not has_execution:
        raise ValueError("plan must have at least one execution task")


def _build_planned_tasks(raw_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"task-{i + 1}",
            "title": str(t.get("title", "")).strip(),
            "kind": str(t.get("kind", "execution")).strip(),
            "status": "pending",
            "worker_role": str(t.get("worker_role", "executor")).strip(),
            "parallel_group": str(t.get("parallel_group", "execute")).strip(),
        }
        for i, t in enumerate(raw_tasks)
    ]


def _get_planner_tool_names(
    tool_registry: ToolRegistry | None,
    state: RuntimeState,
) -> list[str]:
    if tool_registry is None:
        return []
    active_capabilities = _coerce_active_capabilities(state.get("active_capabilities"))
    return tool_registry.visible_tool_names(active_capabilities)


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
    user_msg = f"用户消息: {user_input}"
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
        arguments = _parse_tool_arguments("classify_intent", function.get("arguments", {}))
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
        "needs_plan": intent == "planned",
        "needs_clarification": intent == "clarify",
        "clarification_reason": result.get("reason", ""),
        "request_kind": "planned_task" if intent == "planned" else "direct_reply",
        "context_profile": "planner" if intent == "planned" else "default",
        "execution_mode": "future_subagent" if intent == "planned" else "direct",
        "planner_context": "",
        "plan_summary": "",
        "plan_steps": [],
        "tasks": [],
        "executor_notes": "",
        "suggested_capabilities": [],
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
) -> Callable[[RuntimeState], RuntimeState]:
    def load_context(state: RuntimeState) -> RuntimeState:
        # Flush dirty index entries synchronously
        if indexer is not None:
            _run_provider_sync(indexer.flush_dirty())

        runtime_metadata = state.get("runtime_metadata") or {}
        thread_id = _resolve_thread_id(state, runtime_metadata)
        user_input = str(state.get("user_input", "")).strip()
        request_kind = str(state.get("request_kind", "direct_reply")).strip() or "direct_reply"

        memory_context = ""
        if thread_id:
            memory_context = build_memory_context(
                memory_store,
                thread_id,
                retriever=retriever,
                user_input=user_input,
                memory_token_budget=memory_token_budget,
            )

        context_profile = "default"
        planner_context = ""
        if request_kind == "planned_task":
            context_profile = "planner"
            planner_context = _format_planner_context(state)
        elif request_kind in {"tool_use", "capability_activation"}:
            context_profile = "tooling"
        elif request_kind == "thread_control":
            context_profile = "control"

        return {
            "memory_context": memory_context,
            "context_profile": context_profile,
            "planner_context": planner_context,
        }

    return load_context


def make_planner(
    provider: ChatProvider | None = None,
    tool_registry: ToolRegistry | None = None,
) -> Callable[[RuntimeState], RuntimeState]:
    def planner(state: RuntimeState) -> RuntimeState:
        execution_mode = str(state.get("execution_mode", "direct")).strip() or "direct"
        if not state.get("needs_plan"):
            return {
                "plan_summary": "",
                "plan_steps": [],
                "tasks": [],
                "executor_notes": "",
                "execution_mode": execution_mode,
            }

        if provider is None:
            return _fallback_hardcoded_plan(state, execution_mode)

        try:
            plan = _generate_plan(provider, state, tool_registry)
        except ValueError as exc:
            return {
                "plan_summary": "",
                "plan_steps": [],
                "tasks": [],
                "executor_notes": "",
                "execution_mode": execution_mode,
                "last_error": f"planner validation failed: {exc}",
            }
        except Exception as exc:
            return {
                "plan_summary": "",
                "plan_steps": [],
                "tasks": [],
                "executor_notes": "",
                "execution_mode": execution_mode,
                "last_error": f"planner failed: {exc}",
            }

        return {
            "plan_summary": str(plan.get("summary", "")),
            "plan_steps": [str(t.get("title", "")) for t in plan.get("tasks", [])],
            "tasks": _build_planned_tasks(plan.get("tasks", [])),
            "executor_notes": str(plan.get("executor_notes", "")),
            "execution_mode": execution_mode,
        }

    return planner


def _fallback_hardcoded_plan(state: RuntimeState, execution_mode: str) -> RuntimeState:
    user_request = str(state.get("user_input", "")).strip()
    plan_steps = [
        "Gather relevant context for the request.",
        "Execute the request with the current single-agent runtime path.",
        "Review the result before completion.",
    ]
    tasks = [
        {
            "id": "task-1",
            "title": "Gather relevant context",
            "kind": "research",
            "status": "pending",
            "worker_role": "researcher",
            "parallel_group": "research",
        },
        {
            "id": "task-2",
            "title": "Execute the request",
            "kind": "execution",
            "status": "pending",
            "worker_role": "executor",
            "parallel_group": "execute",
        },
        {
            "id": "task-3",
            "title": "Review the result",
            "kind": "review",
            "status": "pending",
            "worker_role": "reviewer",
            "parallel_group": "review",
        },
    ]
    summary = "Execute the planned request with the current single-agent runtime."
    if user_request:
        summary = f"Execute the planned request: {user_request}"

    return {
        "plan_summary": summary,
        "plan_steps": plan_steps,
        "tasks": tasks,
        "executor_notes": (
            "Keep execution on the current single-agent path. "
            "Do not spawn subagents yet, even when execution_mode is future_subagent."
        ),
        "execution_mode": execution_mode,
    }


def make_agent(
    settings: Settings,
    provider: ChatProvider,
    tool_registry: ToolRegistry | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[RuntimeState], RuntimeState]:
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
        loop_state["active_capabilities"] = _coerce_active_capabilities(state.get("active_capabilities"))
        usage: RuntimeUsage = {}

        consecutive_errors = 0
        try:
            for _ in range(max_rounds):
                messages = context_builder.build_provider_messages(loop_state)
                visible_tools = _build_provider_tools(tool_registry, loop_state["active_capabilities"])
                response = _run_provider_sync(
                    _invoke_provider(
                        provider,
                        messages,
                        model=settings.model,
                        tools=visible_tools,
                    )
                )
                usage = _merge_usage(usage, response)
                if not response.tool_calls:
                    final = _finish_response(loop_state, response, usage)
                    _emit("chunk", text=str(final.get("response_text", "")))
                    return final

                model_text = str(response.content or "").strip()
                if model_text:
                    _emit("model_text", text=model_text)

                for raw_call in response.tool_calls:
                    fn = raw_call.get("function", {})
                    tool_name = str(fn.get("name", ""))
                    tool_args = fn.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except Exception:
                            tool_args = {"raw": tool_args}
                    _emit("tool_calling", text=tool_name, tool_name=tool_name, arguments=tool_args)

                loop_state = _apply_tool_calls(loop_state, response, tool_registry, max_result_chars=max_result_chars)

                last_tool_messages = [
                    msg for msg in loop_state.get("messages", [])
                    if (msg.get("role") if isinstance(msg, dict) else msg.role) == "tool"
                ]
                recent_tool_msgs = last_tool_messages[-len(response.tool_calls):] if response.tool_calls else []
                for i, raw_call in enumerate(response.tool_calls):
                    fn = raw_call.get("function", {})
                    tool_name = str(fn.get("name", ""))
                    tool_args = fn.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except Exception:
                            tool_args = {"raw": tool_args}
                    result_text = ""
                    if i < len(recent_tool_msgs):
                        msg = recent_tool_msgs[i]
                        result_text = str(msg.get("content", "") if isinstance(msg, dict) else msg.content)
                    _emit("tool_done", text=tool_name, tool_name=tool_name, arguments=tool_args, result=result_text[:500])

                if last_tool_messages and _is_error_content(last_tool_messages[-1]):
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        return _error_state(
                            loop_state,
                            f"tool call failed {max_errors} consecutive times",
                            usage,
                        )
                else:
                    consecutive_errors = 0
        except Exception as exc:
            return _error_state(loop_state, _format_error(exc), usage)

        return _error_state(
            loop_state,
            f"tool loop round limit reached after {max_rounds} rounds",
            usage,
        )

    return agent


def make_executor(
    *,
    settings: Settings,
    provider: ChatProvider,
    tool_registry: ToolRegistry | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[RuntimeState], RuntimeState]:
    agent = make_agent(settings=settings, provider=provider, tool_registry=tool_registry, on_event=on_event)
    worker_manager = WorkerManager(
        provider=provider,
        settings=settings,
        tool_registry=tool_registry,
        tracer=None,
    )

    def executor(state: RuntimeState) -> RuntimeState:
        execution_state = dict(state)
        prepared_tasks = _prepare_executor_tasks(state.get("tasks", []))
        execution_state["tasks"] = prepared_tasks
        orchestration = worker_manager.run(
            tasks=prepared_tasks,
            user_input=str(state.get("user_input", "")).strip(),
        )
        execution_state.update(orchestration)

        result = agent(execution_state)
        completed_tasks = _complete_executor_tasks(prepared_tasks)

        merged_state = dict(state)
        merged_state.update(result)
        merged_state["tasks"] = completed_tasks
        merged_state["worker_runs"] = list(orchestration.get("worker_runs", []))
        merged_state["aggregated_worker_context"] = str(orchestration.get("aggregated_worker_context", ""))
        if merged_state.get("last_error"):
            merged_state["orchestration_status"] = "failed"
        else:
            merged_state["orchestration_status"] = str(orchestration.get("orchestration_status", "complete")) or "complete"
        merged_state["execution_mode"] = str(state.get("execution_mode", "direct")).strip() or "direct"
        return merged_state

    return executor


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


def validate(state: RuntimeState) -> RuntimeState:
    if str(state.get("last_error", "")).strip():
        return state
    tasks = state.get("tasks", [])
    if not tasks:
        return {**state, "last_error": "planner validation failed: no tasks generated"}
    has_execution = any(t.get("kind") == "execution" for t in tasks if isinstance(t, dict))
    if not has_execution:
        return {**state, "last_error": "planner validation failed: no execution task"}
    return state


def error_handler(state: RuntimeState) -> RuntimeState:
    error = str(state.get("last_error", ""))
    if "planner failed" in error:
        response = f"任务规划失败，请重试或简化你的请求。\n\n错误详情：{error}"
    elif "planner validation" in error:
        response = f"规划结果不符合要求，请尝试更具体地描述你的需求。\n\n错误详情：{error}"
    else:
        response = f"处理过程中出现错误：{error}"
    return {"response_text": response, "last_error": error}


def route_after_classify(state: RuntimeState) -> str:
    if state.get("needs_clarification"):
        return "clarify"
    return "load_context"


def route_after_load_context(state: RuntimeState) -> str:
    if state.get("needs_plan"):
        return "planner"
    return "executor"


def route_after_validate(state: RuntimeState) -> str:
    if str(state.get("last_error", "")).strip():
        return "error_handler"
    return "executor"


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


def _apply_tool_calls(
    state: RuntimeState,
    response: ChatResponse,
    tool_registry: ToolRegistry | None,
    *,
    max_result_chars: int = MAX_TOOL_RESULT_CHARS,
) -> RuntimeState:
    updated_messages = list(state.get("messages", []))
    updated_messages.append(_runtime_message_from_chat(_assistant_tool_message(response)))
    active_capabilities = _coerce_active_capabilities(state.get("active_capabilities"))
    runtime_context = dict(state.get("runtime_metadata", {}) or {})
    tool_messages, active_capabilities = _execute_tool_calls(
        response.tool_calls,
        tool_registry,
        active_capabilities,
        max_result_chars=max_result_chars,
        runtime_context=runtime_context,
    )
    updated_messages.extend(tool_messages)
    next_state = dict(state)
    next_state["messages"] = updated_messages
    next_state["active_capabilities"] = active_capabilities
    return next_state


def _assistant_tool_message(response: ChatResponse) -> ChatMessage:
    if response.content_parts:
        return ChatMessage(
            role="assistant",
            content_parts=[dict(part) for part in response.content_parts],
            tool_calls=[dict(call) for call in response.tool_calls],
        )

    return ChatMessage(
        role="assistant",
        content=response.content,
        tool_calls=[dict(call) for call in response.tool_calls],
    )


_ACTIVATION_TOOLS = frozenset({"load_skill_tools", "load_mcp_tools"})


def _execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    tool_registry: ToolRegistry | None,
    active_capabilities: ActiveCapabilities,
    *,
    max_result_chars: int = MAX_TOOL_RESULT_CHARS,
    runtime_context: dict[str, Any] | None = None,
) -> tuple[list[RuntimeMessage], ActiveCapabilities]:
    if tool_registry is None:
        raise RuntimeError("tool registry is not configured")

    resolved_context = runtime_context or {}
    parsed = [
        (_parse_tool_call_id(raw), _inject_context(_parse_tool_call(raw), resolved_context))
        for raw in tool_calls
    ]

    # Activation tools must run first and sequentially (they change capability state).
    activation = [(i, tid, tc) for i, (tid, tc) in enumerate(parsed) if tc.name in _ACTIVATION_TOOLS]
    parallel = [(i, tid, tc) for i, (tid, tc) in enumerate(parsed) if tc.name not in _ACTIVATION_TOOLS]

    updated_capabilities = active_capabilities.model_copy(deep=True)
    ordered_results: dict[int, tuple[RuntimeMessage, Any]] = {}

    # Phase 1: activation tools (sequential).
    for idx, tool_call_id, tool_call in activation:
        result = _exec_one(tool_registry, tool_call, updated_capabilities)
        ordered_results[idx] = (_tool_result_message(tool_call_id, tool_call.name, result, max_chars=max_result_chars), result)
        if not result.is_error:
            updated_capabilities = _apply_activation_result(updated_capabilities, result)

    # Phase 2: remaining tools (parallel when >1).
    if len(parallel) == 1:
        idx, tool_call_id, tool_call = parallel[0]
        result = _exec_one(tool_registry, tool_call, updated_capabilities)
        ordered_results[idx] = (_tool_result_message(tool_call_id, tool_call.name, result, max_chars=max_result_chars), result)
    elif parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = {}
        with ThreadPoolExecutor(max_workers=len(parallel)) as pool:
            for idx, tool_call_id, tool_call in parallel:
                futures[pool.submit(_exec_one, tool_registry, tool_call, updated_capabilities)] = (
                    idx,
                    tool_call_id,
                    tool_call,
                )
            for future in as_completed(futures):
                idx, tool_call_id, tool_call = futures[future]
                result = future.result()
                ordered_results[idx] = (_tool_result_message(tool_call_id, tool_call.name, result, max_chars=max_result_chars), result)

    # Merge results in original call order.
    results: list[RuntimeMessage] = []
    for idx in sorted(ordered_results):
        msg, result = ordered_results[idx]
        results.append(msg)
        if not result.is_error:
            updated_capabilities = _apply_activation_result(updated_capabilities, result)

    return results, updated_capabilities


def _exec_one(
    tool_registry: ToolRegistry,
    tool_call: ToolCall,
    active_capabilities: ActiveCapabilities,
) -> Any:
    from miniclaw.tools.contracts import ToolResult

    try:
        return tool_registry.execute(tool_call, active_capabilities)
    except KeyError as exc:
        return ToolResult(content=_format_error(exc), is_error=True)
    except Exception as exc:
        return ToolResult(content=f"tool execution failed for {tool_call.name}: {_format_error(exc)}", is_error=True)


def _tool_result_message(tool_call_id: str, name: str, result: Any, *, max_chars: int = MAX_TOOL_RESULT_CHARS) -> RuntimeMessage:
    content = result.content
    if result.is_error:
        content = f"ERROR: {content}"
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[truncated: {len(result.content):,} chars, showing first {max_chars:,}]"
    return _runtime_message_from_chat(
        ChatMessage(
            role="tool",
            name=name,
            tool_call_id=tool_call_id,
            content=content,
        )
    )


def _parse_tool_call(raw_call: dict[str, Any]) -> ToolCall:
    function = raw_call.get("function")
    if not isinstance(function, dict):
        raise RuntimeError("tool call is missing function metadata")

    tool_name = str(function.get("name", "")).strip()
    if not tool_name:
        raise RuntimeError("tool call is missing function name")

    arguments = _parse_tool_arguments(tool_name, function.get("arguments", {}))
    return ToolCall(name=tool_name, arguments=arguments)


def _inject_context(call: ToolCall, context: dict[str, Any]) -> ToolCall:
    if not context:
        return call
    return call.model_copy(update={"context": context})


def _parse_tool_call_id(raw_call: dict[str, Any]) -> str:
    tool_call_id = str(raw_call.get("id", "")).strip()
    if not tool_call_id:
        raise RuntimeError("tool call is missing id")
    return tool_call_id


def _parse_tool_arguments(name: str, raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments in (None, ""):
        return {}
    if isinstance(raw_arguments, str):
        try:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(raw_arguments.strip())
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"invalid tool arguments for {name}: {exc}") from exc
    elif isinstance(raw_arguments, dict):
        parsed = raw_arguments
    else:
        raise RuntimeError(f"tool arguments for {name} must be an object")

    if not isinstance(parsed, dict):
        raise RuntimeError(f"tool arguments for {name} must decode to an object")
    return parsed


def _is_error_content(message: RuntimeMessage | dict[str, Any]) -> bool:
    content = message.get("content", "") if isinstance(message, dict) else message.content
    return str(content or "").startswith("ERROR: ")


def _finish_response(
    state: RuntimeState,
    response: ChatResponse,
    usage: RuntimeUsage,
) -> RuntimeState:
    updated_messages = list(state.get("messages", []))
    updated_messages.append(_runtime_message_from_chat(_assistant_response_message(response)))
    return {
        "messages": updated_messages,
        "response_text": response.content,
        "last_error": "",
        "usage": usage,
        "active_capabilities": _coerce_active_capabilities(state.get("active_capabilities")),
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
        "active_capabilities": _coerce_active_capabilities(state.get("active_capabilities")),
    }


def _assistant_response_message(response: ChatResponse) -> ChatMessage:
    if response.content_parts:
        return ChatMessage(role="assistant", content_parts=[dict(part) for part in response.content_parts])
    return ChatMessage(role="assistant", content=response.content)


def _runtime_message_from_chat(message: ChatMessage) -> RuntimeMessage:
    payload: RuntimeMessage = {"role": message.role, "content": message.content or ""}
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [dict(call) for call in message.tool_calls]
    if message.content_parts:
        payload["content_parts"] = [dict(part) for part in message.content_parts]
    return payload


def _coerce_active_capabilities(value: Any) -> ActiveCapabilities:
    if isinstance(value, ActiveCapabilities):
        return value
    if isinstance(value, dict):
        return ActiveCapabilities.model_validate(value)
    return ActiveCapabilities()


def _apply_activation_result(
    active_capabilities: ActiveCapabilities,
    result: ToolResult,
) -> ActiveCapabilities:
    metadata = result.metadata

    updated = active_capabilities.model_copy(deep=True)
    activation_type = str(metadata.get("activation_type", "")).strip()
    if activation_type == "skills":
        updated.skills = _merge_unique(updated.skills, metadata.get("skills"))
    elif activation_type == "mcp":
        server_name = str(metadata.get("server", "")).strip()
        if server_name:
            updated.mcp_servers = _merge_unique(updated.mcp_servers, [server_name])
        updated.mcp_tools = _merge_unique(updated.mcp_tools, metadata.get("tool_names"))
    return updated


def _merge_unique(current: list[str], values: Any) -> list[str]:
    merged = list(current)
    seen = set(merged)
    if not isinstance(values, list):
        return merged
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        merged.append(item)
        seen.add(item)
    return merged


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


def _format_error(exc: Exception) -> str:
    if len(exc.args) == 1 and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)


def _resolve_thread_id(state: RuntimeState, runtime_metadata: object) -> str:
    if isinstance(runtime_metadata, dict):
        metadata_thread_id = str(runtime_metadata.get("thread_id", "")).strip()
        if metadata_thread_id:
            return metadata_thread_id
    return str(state.get("thread_id", "")).strip()


def _format_planner_context(state: RuntimeState) -> str:
    user_request = str(state.get("user_input", "")).strip()
    return f"Planner context:\nUser request: {user_request}" if user_request else "Planner context:\nUser request: "



def _prepare_executor_tasks(tasks: Any) -> list[dict[str, Any]]:
    if not isinstance(tasks, list):
        return []

    prepared: list[dict[str, Any]] = []
    for raw_task in tasks:
        if isinstance(raw_task, dict):
            task = dict(raw_task)
            _normalize_worker_metadata(task)
            prepared.append(task)

    if not prepared:
        return prepared

    for index, task in enumerate(prepared):
        task["status"] = "completed" if index < len(prepared) - 1 else "in_progress"
    return prepared


def _complete_executor_tasks(tasks: Any) -> list[dict[str, Any]]:
    if not isinstance(tasks, list):
        return []

    completed: list[dict[str, Any]] = []
    for raw_task in tasks:
        if not isinstance(raw_task, dict):
            continue
        task = dict(raw_task)
        _normalize_worker_metadata(task)
        if task.get("status") in {"pending", "in_progress"}:
            task["status"] = "completed"
        completed.append(task)
    return completed


def _normalize_worker_metadata(task: dict[str, Any]) -> None:
    worker_role = str(task.get("worker_role", "")).strip()
    if not worker_role:
        worker_role = _default_worker_role(task)
        if worker_role:
            task["worker_role"] = worker_role

    parallel_group = str(task.get("parallel_group", "")).strip()
    if not parallel_group:
        parallel_group = _default_parallel_group(task, worker_role)
        if parallel_group:
            task["parallel_group"] = parallel_group


def _default_worker_role(task: Mapping[str, Any]) -> str:
    kind = str(task.get("kind", "")).strip()
    if kind in {"research", "planning", "preparation"}:
        return "researcher"
    if kind in {"execution", "implementation", "build"}:
        return "executor"
    if kind == "review":
        return "reviewer"
    return ""


def _default_parallel_group(task: Mapping[str, Any], worker_role: str) -> str:
    parallel_group = str(task.get("parallel_group", "")).strip()
    if parallel_group:
        return parallel_group
    if worker_role == "researcher":
        return "research"
    if worker_role == "executor":
        return "execute"
    if worker_role == "reviewer":
        return "review"
    kind = str(task.get("kind", "")).strip()
    return kind
