from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from miniclaw.runtime.tool_loop import (
    trace_messages,
    trace_tool_calls,
    trace_tool_names,
)

if TYPE_CHECKING:
    from miniclaw.config.settings import Settings
    from miniclaw.observability.contracts import TraceContext, Tracer
    from miniclaw.providers.contracts import ChatProvider
    from miniclaw.tools.registry import ToolRegistry


def _safe_tracer_call(fn: Callable[..., Any], *args: Any, fallback: Any = None, **kwargs: Any) -> Any:
    """Invoke a tracer method, swallowing exceptions so a broken tracer never
    propagates into the subagent loop. Returns `fallback` on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return fallback


@dataclass(frozen=True)
class SubagentBrief:
    role: str
    task: str
    expected_output: str = ""
    tools: tuple[str, ...] | None = None
    context: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubagentResult:
    sub_id: str
    fleet_id: str
    role: str
    status: Literal["completed", "failed"]
    result: str = ""
    error: str = ""
    summary: str = ""
    rounds_used: int = 0
    usage: dict[str, int] = field(default_factory=dict)


ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "researcher": {
        "prompt": (
            "You are a researcher subagent. Your job is to investigate the assigned task "
            "and produce concise findings. You have read-only tools. Do not modify state. "
            "When you have enough information, respond with a final answer (no tool calls)."
        ),
        "tools": (
            "read_file", "web_search", "memory_search",
            "list_skills", "load_skill_tools",
        ),
    },
    "executor": {
        "prompt": (
            "You are an executor subagent. Your job is to carry out the assigned task and "
            "report concrete results. Use write_file to create or modify files in your "
            "user sandbox (path is relative to the sandbox root). Use the read-only shell "
            "to inspect files. When the task is complete, respond with a final summary "
            "(no tool calls)."
        ),
        "tools": (
            "read_file", "write_file", "shell", "memory_search",
            "load_skill_tools", "load_mcp_tools",
        ),
    },
    "reviewer": {
        "prompt": (
            "You are a reviewer subagent. Your job is to verify the work described in the "
            "task brief, identify gaps, and suggest follow-up checks. You have read-only "
            "tools. When done, respond with a final review (no tool calls)."
        ),
        "tools": (
            "read_file", "shell", "memory_search",
        ),
    },
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GENERIC_PROMPT = (
    "You are a subagent. Carry out the assigned task and respond with a final answer "
    "when complete. Use the available tools as needed."
)

WORKER_BLACKLIST = frozenset({"spawn_subagent", "spawn_worker"})


# ---------------------------------------------------------------------------
# Core async function
# ---------------------------------------------------------------------------

async def run_subagent(
    *,
    brief: SubagentBrief,
    fleet_id: str,
    sub_id: str,
    provider: "ChatProvider",
    settings: "Settings",
    tool_registry: "ToolRegistry",
    runtime_metadata: Mapping[str, Any],
    tracer: "Tracer",
    parent_trace: "TraceContext",
    on_event: Callable[[Mapping[str, Any]], None] | None = None,
) -> SubagentResult:
    """Run a single subagent turn loop and return a SubagentResult.

    Builds clean-room messages (system + user), then iterates a tool loop
    using execute_tool_calls(..., worker_visible_only=True) until the provider
    returns a response with no tool_calls or limits are reached.
    """
    from miniclaw.providers.contracts import ChatMessage
    from miniclaw.runtime.state import ActiveCapabilities
    from miniclaw.runtime.tool_loop import execute_tool_calls

    span = tracer.start_span(
        parent_trace,
        name="subagent.run",
        metadata={
            "subagent.fleet_id": fleet_id,
            "subagent.sub_id": sub_id,
            "subagent.role": brief.role,
            "subagent.task_summary": brief.task[:120],
            "subagent.depth": 1,
            "subagent.tools_count": len(_resolve_tools(brief)),
        },
    )

    _safe_emit(on_event, {
        "kind": "subagent_dispatched",
        "fleet_id": fleet_id,
        "sub_id": sub_id,
        "role": brief.role,
        "task_summary": brief.task[:120],
    })
    _safe_emit(on_event, {"kind": "subagent_started", "sub_id": sub_id})

    tools_schema = _build_tool_schemas(brief, tool_registry)
    messages: list[ChatMessage] = _build_messages(brief, runtime_metadata, fleet_id, sub_id)

    last_response_text = ""
    rounds_used = 0
    consecutive_errors = 0
    final_usage: dict[str, Any] = {}
    max_result_chars = int(getattr(settings, "subagent_max_tool_result_chars", 2_000) or 2_000)

    try:
        for round_idx in range(settings.max_tool_rounds):
            rounds_used = round_idx + 1
            round_span = _safe_tracer_call(
                tracer.start_span,
                span,
                name=f"subagent.tool_loop.round_{round_idx}",
                metadata={"subagent.sub_id": sub_id},
                inputs={
                    "messages": trace_messages(messages),
                    "tools": trace_tool_names(tools_schema),
                    "model": settings.model,
                },
                fallback=span,
            )
            chat_span = _safe_tracer_call(
                tracer.start_span,
                round_span,
                name="provider.achat",
                metadata={"subagent.sub_id": sub_id, "provider.model": settings.model},
                inputs={"messages": trace_messages(messages), "tool_count": len(tools_schema or [])},
                fallback=round_span,
            )
            try:
                response = await provider.achat(
                    messages,
                    model=settings.model,
                    tools=tools_schema or None,
                )
            except Exception as exc:
                _safe_tracer_call(
                    tracer.finish_span, chat_span,
                    status="error", metadata={"error": str(exc)}, fallback=None,
                )
                _safe_tracer_call(
                    tracer.finish_span, round_span,
                    status="error", fallback=None,
                )
                raise

            usage_raw = getattr(response, "usage", None)
            if usage_raw is not None:
                if hasattr(usage_raw, "model_dump"):
                    final_usage = {k: v for k, v in usage_raw.model_dump().items() if v is not None}
                elif isinstance(usage_raw, dict):
                    final_usage = dict(usage_raw)

            last_response_text = str(response.content or "")
            round_outputs = {
                "content": last_response_text[:max_result_chars],
                "tool_calls": trace_tool_calls(response.tool_calls),
                "usage": dict(final_usage),
            }
            _safe_tracer_call(
                tracer.finish_span, chat_span,
                status="ok",
                metadata={"provider.usage": final_usage},
                outputs=round_outputs,
                fallback=None,
            )

            if not response.tool_calls:
                _safe_tracer_call(
                    tracer.finish_span, round_span,
                    status="ok", outputs=round_outputs, fallback=None,
                )
                _emit_completed(on_event, sub_id, "completed", last_response_text, rounds_used)
                _safe_tracer_call(
                    tracer.finish_span, span,
                    status="ok", metadata={"rounds_used": rounds_used}, fallback=None,
                )
                return SubagentResult(
                    sub_id=sub_id,
                    fleet_id=fleet_id,
                    role=brief.role,
                    status="completed",
                    result=last_response_text,
                    summary=_summarize(last_response_text),
                    rounds_used=rounds_used,
                    usage=final_usage,
                )

            # Append assistant message with tool calls.
            messages.append(_assistant_with_tool_calls(response))

            # Execute tool calls (worker_visible_only=True filters blacklisted tools).
            # Pass tracer so each tool.<name> span hangs under the round span.
            tool_msgs, _ = execute_tool_calls(
                response.tool_calls,
                tool_registry,
                ActiveCapabilities(),
                max_result_chars=max_result_chars,
                runtime_context=dict(runtime_metadata),
                worker_visible_only=True,
                tracer=tracer,
                parent_trace=round_span,
            )

            # Convert RuntimeMessage dicts back to ChatMessage for the next round.
            for msg_dict in tool_msgs:
                messages.append(ChatMessage(
                    role="tool",
                    name=msg_dict.get("name"),
                    tool_call_id=msg_dict.get("tool_call_id", ""),
                    content=msg_dict.get("content", ""),
                ))

            # Truncate any oversized assistant messages to prevent context
            # explosion from large model completions (e.g., executor generating
            # 10k+ chars of markdown that gets re-sent every round).
            _MAX_ASSISTANT_CHARS = 4000
            for idx_msg in range(len(messages)):
                msg = messages[idx_msg]
                if getattr(msg, "role", "") == "assistant":
                    content = getattr(msg, "content", "") or ""
                    if len(content) > _MAX_ASSISTANT_CHARS:
                        messages[idx_msg] = ChatMessage(
                            role="assistant",
                            content=content[:_MAX_ASSISTANT_CHARS] + "\n...[response truncated]",
                            tool_calls=getattr(msg, "tool_calls", None),
                        )

            _safe_tracer_call(
                tracer.finish_span, round_span,
                status="ok",
                metadata={"tool_calls": len(response.tool_calls)},
                outputs=round_outputs,
                fallback=None,
            )

            # Track consecutive tool errors.
            if tool_msgs and _last_is_error(tool_msgs):
                consecutive_errors += 1
                if consecutive_errors >= settings.max_consecutive_tool_errors:
                    tracer.finish_span(span, status="error", metadata={"error": "consecutive errors"})
                    _emit_completed(on_event, sub_id, "failed",
                                    f"tool errors x{consecutive_errors}", rounds_used)
                    return SubagentResult(
                        sub_id=sub_id,
                        fleet_id=fleet_id,
                        role=brief.role,
                        status="failed",
                        error="consecutive tool errors",
                        summary=_summarize(last_response_text),
                        rounds_used=rounds_used,
                        usage=final_usage,
                    )
            else:
                consecutive_errors = 0

    except Exception as exc:
        tracer.finish_span(span, status="error", metadata={"error": str(exc)})
        _emit_completed(on_event, sub_id, "failed", str(exc), rounds_used)
        return SubagentResult(
            sub_id=sub_id,
            fleet_id=fleet_id,
            role=brief.role,
            status="failed",
            error=str(exc),
            summary=_summarize(last_response_text),
            rounds_used=rounds_used,
            usage=final_usage,
        )

    # Round limit exhausted.
    tracer.finish_span(span, status="error", metadata={"error": "round limit"})
    _emit_completed(on_event, sub_id, "failed", "round limit reached", rounds_used)
    return SubagentResult(
        sub_id=sub_id,
        fleet_id=fleet_id,
        role=brief.role,
        status="failed",
        error=f"round limit ({settings.max_tool_rounds}) reached",
        summary=_summarize(last_response_text),
        rounds_used=rounds_used,
        usage=final_usage,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_emit(
    on_event: Callable[[Mapping[str, Any]], None] | None,
    payload: Mapping[str, Any],
) -> None:
    """Emit an event to the channel callback, swallowing any exceptions.

    Channel callback failures must not crash the subagent or leave trace
    spans open — so we intentionally silence all Exception subclasses here.
    """
    if on_event is None:
        return
    try:
        on_event(payload)
    except Exception:
        pass  # channel callback failures must not crash the subagent


def _emit_completed(
    on_event: Callable[[Mapping[str, Any]], None] | None,
    sub_id: str,
    status: str,
    summary: str,
    rounds: int,
) -> None:
    _safe_emit(on_event, {
        "kind": "subagent_completed",
        "sub_id": sub_id,
        "status": status,
        "result_summary": summary[:500] if status == "completed" else "",
        "error": summary[:500] if status == "failed" else "",
        "rounds_used": rounds,
    })


def _summarize(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= 500:
        return text
    return text[:500] + "..."


def _resolve_tools(brief: SubagentBrief) -> tuple[str, ...]:
    """Return the effective tool list for a brief, excluding WORKER_BLACKLIST."""
    if brief.tools is not None:
        return tuple(t for t in brief.tools if t not in WORKER_BLACKLIST)
    defaults = ROLE_DEFAULTS.get(brief.role)
    if defaults is None:
        return ()
    return tuple(t for t in defaults["tools"] if t not in WORKER_BLACKLIST)


def _build_tool_schemas(brief: SubagentBrief, registry: "ToolRegistry") -> list[dict[str, Any]]:
    """Build the OpenAI-style tool schemas for the provider call."""
    names = _resolve_tools(brief)
    schemas: list[dict[str, Any]] = []
    for name in names:
        spec = None
        try:
            tool = registry.get(name)
            spec = tool.spec if tool is not None else None
        except Exception:
            spec = None
        if spec is None:
            continue
        # Defense-in-depth: skip tools marked worker_visible=False even if resolved.
        if not spec.metadata.get("worker_visible", True):
            continue
        schemas.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema or {"type": "object", "properties": {}},
            },
        })
    return schemas


def _build_messages(
    brief: SubagentBrief,
    runtime_metadata: Mapping[str, Any],
    fleet_id: str,
    sub_id: str,
) -> "list[Any]":
    """Build the initial [system, user] ChatMessage list for the subagent.

    The system message is split into two content_parts:
    - Static part (cache_control: ephemeral): role_prompt + stable runtime fields
      (thread_id, channel). These are identical for all spawns of the same
      (role, thread_id, channel) combination, enabling provider-side cache hits.
    - Dynamic part (no cache_control): fleet_id and sub_id, which change per spawn.
    """
    from miniclaw.providers.contracts import ChatMessage

    role_defaults = ROLE_DEFAULTS.get(brief.role)
    role_prompt = role_defaults["prompt"] if role_defaults else _GENERIC_PROMPT

    # Static segment: stable across all spawns of (role, thread_id, channel)
    static_lines = [
        role_prompt,
        "",
        "## Runtime",
        f"thread_id: {runtime_metadata.get('thread_id', '')}",
        f"channel: {runtime_metadata.get('channel', '')}",
        "parent_role: main_agent",
    ]
    static_text = "\n".join(static_lines)

    # Dynamic segment: per-spawn (changes every dispatch)
    dynamic_lines = [
        f"fleet_id: {fleet_id}",
        f"sub_id: {sub_id}",
    ]
    dynamic_text = "\n".join(dynamic_lines)

    user_lines = [f"Task: {brief.task}"]
    if brief.expected_output:
        user_lines.append(f"Expected output: {brief.expected_output}")
    if brief.context:
        user_lines.append("Context:\n" + "\n---\n".join(brief.context))
    user_text = "\n\n".join(user_lines)

    return [
        ChatMessage(
            role="system",
            content_parts=[
                {"type": "text", "text": static_text, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": dynamic_text},
            ],
        ),
        ChatMessage(role="user", content=user_text),
    ]


def _assistant_with_tool_calls(response: Any) -> "Any":
    """Convert a ChatResponse with tool_calls into an assistant ChatMessage."""
    from miniclaw.providers.contracts import ChatMessage

    return ChatMessage(
        role="assistant",
        content=response.content or "",
        tool_calls=[dict(call) for call in response.tool_calls],
    )


def _last_is_error(tool_msgs: list[dict[str, Any]]) -> bool:
    """Return True if the last tool message content starts with 'ERROR:'."""
    if not tool_msgs:
        return False
    last = tool_msgs[-1]
    content = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
    return isinstance(content, str) and content.startswith("ERROR:")
