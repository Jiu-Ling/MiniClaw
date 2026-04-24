"""Emit a `prompt.cache.usage` trace event from a ChatResponse's usage field.

Centralized so both main agent, subagent, and any future call site can
report cache usage consistently. No-op when tracer is None, span is None,
or the response has no usage. Never raises.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from miniclaw.observability.safe import safe_record_event

if TYPE_CHECKING:
    from miniclaw.observability.contracts import TraceContext, Tracer
    from miniclaw.providers.contracts import ChatResponse


def emit_cache_usage(
    tracer: "Tracer | None",
    span: "TraceContext | None",
    response: "ChatResponse",
) -> None:
    """Record a `prompt.cache.usage` event under `span` with usage counts.

    Payload keys:
      prompt_tokens, completion_tokens, total_tokens,
      cached_tokens, cache_creation_tokens, cache_hit_rate

    cache_hit_rate = cached_tokens / prompt_tokens (0.0-1.0). Omitted when
    prompt_tokens is None or zero.
    """
    if tracer is None or span is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    prompt_tokens: int = getattr(usage, "prompt_tokens", None) or 0
    cached_tokens: int = getattr(usage, "cached_tokens", None) or 0
    cache_creation_tokens: int = getattr(usage, "cache_creation_tokens", None) or 0
    completion_tokens: int = getattr(usage, "completion_tokens", None) or 0
    total_tokens: int = getattr(usage, "total_tokens", None) or 0

    payload: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
    }
    if prompt_tokens > 0:
        payload["cache_hit_rate"] = round(cached_tokens / prompt_tokens, 4)

    safe_record_event(tracer, span, name="prompt.cache.usage", payload=payload)
