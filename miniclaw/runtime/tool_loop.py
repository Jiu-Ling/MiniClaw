from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from miniclaw.providers.contracts import ChatMessage, ChatResponse
from miniclaw.runtime.state import ActiveCapabilities, RuntimeMessage, RuntimeState
from miniclaw.tools.contracts import ToolCall, ToolResult
from miniclaw.tools.registry import ToolRegistry
from miniclaw.utils.jsonx import safe_loads_with_raw

if TYPE_CHECKING:
    from miniclaw.observability.contracts import Tracer

from miniclaw.observability.contracts import TraceContext, build_run_context

MAX_TOOL_RESULT_CHARS = 16_000
_ACTIVATION_TOOLS = frozenset({"load_skill_tools", "load_mcp_tools"})

# Cap on messages recorded into a span's input payload. Tracer also truncates
# long strings globally via max_chars; this bound is about row count.
TRACE_MESSAGE_LIMIT = 20


def trace_messages(messages: list[ChatMessage] | None) -> list[dict[str, Any]]:
    """Compact representation of a ChatMessage list for span input payloads."""
    if not messages:
        return []
    tail = messages[-TRACE_MESSAGE_LIMIT:]
    out: list[dict[str, Any]] = []
    for m in tail:
        item: dict[str, Any] = {"role": m.role or ""}
        if isinstance(m.content, str) and m.content:
            item["content"] = m.content
        if m.content_parts:
            item["content_parts_count"] = len(m.content_parts)
        if m.tool_calls:
            item["tool_call_count"] = len(m.tool_calls)
        if m.tool_call_id:
            item["tool_call_id"] = m.tool_call_id
        if m.name:
            item["name"] = m.name
        out.append(item)
    return out


def trace_tool_names(tools: list[dict[str, Any]] | None) -> list[str]:
    """Flatten OpenAI-style tool definitions into a list of names."""
    if not tools:
        return []
    names: list[str] = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


def resolve_turn_trace(state: Any, fallback_name: str) -> TraceContext:
    """Reconstruct a parent TraceContext from ``state["runtime_metadata"]``.

    The service layer writes ``_turn_trace_id`` + ``_turn_run_id`` into
    runtime_metadata at turn start so every span in the turn (graph nodes,
    agent tool-loop rounds, provider.achat, per-tool spans, subagents)
    shares one ``trace_id``. Without those keys we fall back to a fresh
    run context so isolated calls (tests, direct invocations) still work.
    """
    thread_id: str | None = None
    channel: str | None = None
    turn_trace_id: str | None = None
    turn_run_id: str | None = None
    runtime_metadata: Any = None
    if isinstance(state, dict) or (state is not None and hasattr(state, "get")):
        try:
            runtime_metadata = state.get("runtime_metadata")
        except Exception:
            runtime_metadata = None
    if isinstance(runtime_metadata, dict) or (runtime_metadata is not None and hasattr(runtime_metadata, "get")):
        try:
            raw_thread = runtime_metadata.get("thread_id")
            if raw_thread:
                thread_id = str(raw_thread)
        except Exception:
            pass
        try:
            raw_channel = runtime_metadata.get("channel")
            if raw_channel:
                channel = str(raw_channel)
        except Exception:
            pass
        try:
            raw_trace = runtime_metadata.get("_turn_trace_id")
            if raw_trace:
                turn_trace_id = str(raw_trace)
        except Exception:
            pass
        try:
            raw_run = runtime_metadata.get("_turn_run_id")
            if raw_run:
                turn_run_id = str(raw_run)
        except Exception:
            pass

    if turn_trace_id and turn_run_id:
        return TraceContext(
            trace_id=turn_trace_id,
            run_id=turn_run_id,
            thread_id=thread_id,
            channel=channel,
            name="graph.run",
        )
    return build_run_context(name=fallback_name, thread_id=thread_id, channel=channel)


def trace_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Decode tool_calls into a tracer-friendly list with parsed arguments."""
    if not tool_calls:
        return []
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name", "") if isinstance(fn, dict) else ""
        args_raw = fn.get("arguments", "") if isinstance(fn, dict) else ""
        args = safe_loads_with_raw(args_raw)
        out.append({
            "id": tc.get("id") if isinstance(tc, dict) else None,
            "name": str(name),
            "arguments": args,
        })
    return out


def apply_tool_calls(
    state: RuntimeState,
    response: ChatResponse,
    tool_registry: ToolRegistry | None,
    *,
    max_result_chars: int = MAX_TOOL_RESULT_CHARS,
    worker_visible_only: bool = False,
    tracer: "Tracer | None" = None,
    parent_trace: "TraceContext | None" = None,
) -> RuntimeState:
    updated_messages = list(state.get("messages", []))
    updated_messages.append(runtime_message_from_chat(_assistant_tool_message(response)))
    active_capabilities = coerce_active_capabilities(state.get("active_capabilities"))
    runtime_context = dict(state.get("runtime_metadata", {}) or {})
    tool_messages, active_capabilities = execute_tool_calls(
        response.tool_calls,
        tool_registry,
        active_capabilities,
        max_result_chars=max_result_chars,
        runtime_context=runtime_context,
        worker_visible_only=worker_visible_only,
        tracer=tracer,
        parent_trace=parent_trace,
    )
    updated_messages.extend(tool_messages)
    next_state = dict(state)
    next_state["messages"] = updated_messages
    next_state["active_capabilities"] = active_capabilities
    return next_state


def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    tool_registry: ToolRegistry | None,
    active_capabilities: ActiveCapabilities,
    *,
    max_result_chars: int = MAX_TOOL_RESULT_CHARS,
    runtime_context: dict[str, Any] | None = None,
    worker_visible_only: bool = False,
    tracer: "Tracer | None" = None,
    parent_trace: "TraceContext | None" = None,
) -> tuple[list[RuntimeMessage], ActiveCapabilities]:
    if tool_registry is None:
        raise RuntimeError("tool registry is not configured")

    resolved_context = runtime_context or {}
    parsed = [
        (_parse_tool_call_id(raw), _inject_context(_parse_tool_call(raw), resolved_context))
        for raw in tool_calls
    ]

    if worker_visible_only:
        parsed = _filter_worker_visible(parsed, tool_registry)

    # Activation tools must run first and sequentially (they change capability state).
    activation = [(i, tid, tc) for i, (tid, tc) in enumerate(parsed) if tc.name in _ACTIVATION_TOOLS]
    parallel = [(i, tid, tc) for i, (tid, tc) in enumerate(parsed) if tc.name not in _ACTIVATION_TOOLS]

    updated_capabilities = active_capabilities.model_copy(deep=True)
    ordered_results: dict[int, tuple[RuntimeMessage, Any]] = {}

    def _run_one(tc: ToolCall, caps: ActiveCapabilities) -> Any:
        return _exec_one_with_span(
            tool_registry, tc, caps,
            tracer=tracer, parent_trace=parent_trace,
            max_result_chars=max_result_chars,
        )

    # Phase 1: activation tools (sequential).
    for idx, tool_call_id, tool_call in activation:
        result = _run_one(tool_call, updated_capabilities)
        ordered_results[idx] = (_tool_result_message(tool_call_id, tool_call.name, result, max_chars=max_result_chars), result)
        if not result.is_error:
            updated_capabilities = _apply_activation_result(updated_capabilities, result)

    # Phase 2: remaining tools (parallel when >1).
    if len(parallel) == 1:
        idx, tool_call_id, tool_call = parallel[0]
        result = _run_one(tool_call, updated_capabilities)
        ordered_results[idx] = (_tool_result_message(tool_call_id, tool_call.name, result, max_chars=max_result_chars), result)
    elif parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = {}
        with ThreadPoolExecutor(max_workers=len(parallel)) as pool:
            for idx, tool_call_id, tool_call in parallel:
                futures[pool.submit(_run_one, tool_call, updated_capabilities)] = (
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


def _filter_worker_visible(
    parsed: list[tuple[str, ToolCall]],
    tool_registry: ToolRegistry | None,
) -> list[tuple[str, ToolCall]]:
    """Drop tool calls whose ToolSpec.metadata['worker_visible'] is False.

    Pass through unknown tool names (the existing ToolRegistry.execute path
    will return a clean error).
    """
    filtered = []
    for tool_call_id, tool_call in parsed:
        spec = None
        try:
            registered = tool_registry.get(tool_call.name) if tool_registry is not None else None
            spec = registered.spec if registered is not None else None
        except Exception:
            spec = None
        if spec is None or spec.metadata.get("worker_visible", True):
            filtered.append((tool_call_id, tool_call))
        # else: silently drop — subagent must not be allowed to invoke
    return filtered


def _exec_one(
    tool_registry: ToolRegistry,
    tool_call: ToolCall,
    active_capabilities: ActiveCapabilities,
) -> Any:
    try:
        return tool_registry.execute(tool_call, active_capabilities)
    except KeyError as exc:
        return ToolResult(content=format_error(exc), is_error=True)
    except Exception as exc:
        return ToolResult(content=f"tool execution failed for {tool_call.name}: {format_error(exc)}", is_error=True)


def _exec_one_with_span(
    tool_registry: ToolRegistry,
    tool_call: ToolCall,
    active_capabilities: ActiveCapabilities,
    *,
    tracer: "Tracer | None",
    parent_trace: "TraceContext | None",
    max_result_chars: int,
) -> Any:
    """Execute a single tool call, optionally wrapping it in a `tool.<name>` span.

    The span's `inputs` carry the tool arguments; the span's `outputs` carry a
    truncated copy of the result content plus is_error and tool result metadata.
    Errors in tracer calls never propagate — a failing tracer must not break
    tool execution (which in turn would break the agent loop).
    """
    if tracer is None or parent_trace is None:
        return _exec_one(tool_registry, tool_call, active_capabilities)

    span: "TraceContext | None" = None
    try:
        registered = tool_registry.get(tool_call.name)
        metadata: dict[str, str] = {"tool.name": tool_call.name}
        if registered is not None:
            metadata["tool.source"] = registered.spec.source
        span = tracer.start_span(
            parent_trace,
            name=f"tool.{tool_call.name}",
            metadata=metadata,
            inputs={"arguments": _safe_copy_for_trace(tool_call.arguments)},
        )
    except Exception:
        span = None

    result = _exec_one(tool_registry, tool_call, active_capabilities)

    if span is not None:
        content = result.content or ""
        if len(content) > max_result_chars:
            content = content[:max_result_chars] + f"\n[truncated {len(result.content) - max_result_chars} chars]"
        try:
            tracer.finish_span(
                span,
                status="error" if result.is_error else "ok",
                outputs={
                    "content": content,
                    "is_error": result.is_error,
                    "metadata": _safe_copy_for_trace(result.metadata or {}),
                },
            )
        except Exception:
            pass
    return result


def _safe_copy_for_trace(value: Any) -> Any:
    """Shallow copy a value for trace output, filtering out private keys
    (leading underscore) and any non-JSON-serializable things.

    Private keys like `_on_event`, `_parent_trace`, `_sub_index` come from
    runtime_context and don't belong in persisted traces.
    """
    if isinstance(value, dict):
        return {
            str(k): _safe_copy_for_trace(v)
            for k, v in value.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_copy_for_trace(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tool_result_message(tool_call_id: str, name: str, result: Any, *, max_chars: int = MAX_TOOL_RESULT_CHARS) -> RuntimeMessage:
    content = result.content
    if result.is_error:
        content = f"ERROR: {content}"
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[truncated: {len(result.content):,} chars, showing first {max_chars:,}]"
    return runtime_message_from_chat(
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

    arguments = parse_tool_arguments(tool_name, function.get("arguments", {}))
    return ToolCall(name=tool_name, arguments=arguments)


def _parse_tool_call_id(raw_call: dict[str, Any]) -> str:
    tool_call_id = str(raw_call.get("id", "")).strip()
    if not tool_call_id:
        raise RuntimeError("tool call is missing id")
    return tool_call_id


def parse_tool_arguments(name: str, raw_arguments: Any) -> dict[str, Any]:
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


def _inject_context(call: ToolCall, context: dict[str, Any]) -> ToolCall:
    if not context:
        return call
    return call.model_copy(update={"context": context})


def runtime_message_from_chat(message: ChatMessage) -> RuntimeMessage:
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


def coerce_active_capabilities(value: Any) -> ActiveCapabilities:
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


def format_error(exc: Exception) -> str:
    if len(exc.args) == 1 and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)
