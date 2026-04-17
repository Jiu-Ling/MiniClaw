"""LLM-driven query rewrite for memory retrieval.

Uses provider.achat_structured() for type-safe Pydantic responses via
provider-native function calling. Falls back gracefully on any failure.
Never raises, never blocks more than timeout_s seconds.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from miniclaw.providers.contracts import ChatProvider

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Intent = Literal["recall_prior", "new_topic", "direct_task", "ambiguous"]

__all__ = ["RewriteInput", "RewriteResult", "RewriteResponseSchema", "rewrite_query"]


class RewriteResponseSchema(BaseModel):
    """Pydantic schema for structured LLM output."""

    rewritten_query: str = Field(description="Self-contained retrieval query, resolves pronouns, <100 chars")
    keywords: list[str] = Field(default_factory=list, description="3-6 nouns/identifiers for keyword search")
    intent: str = Field(default="ambiguous", description="recall_prior | new_topic | direct_task | ambiguous")


@dataclass(frozen=True)
class RewriteInput:
    user_input: str
    recent_exchanges: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RewriteResult:
    rewritten_query: str
    keywords: tuple[str, ...]
    intent: Intent
    used_llm: bool
    latency_ms: int = 0
    raw_response: str = ""
    failure_reason: str = ""


async def rewrite_query(
    inputs: RewriteInput,
    *,
    provider: ChatProvider | None = None,
    model: str = "",
    timeout_s: float = 1.0,
) -> RewriteResult:
    """Rewrite user_input into a retrieval query with intent. Never raises.

    provider: a ChatProvider with achat_structured support.
    """
    if provider is None:
        return _fallback(inputs, reason="no_provider")
    if not inputs.user_input.strip():
        return _fallback(inputs, reason="empty_input")

    messages = _build_messages(inputs)
    start = time.monotonic()
    try:
        # Use raw achat + extract_json_object for rewrite — it's a 3-field JSON,
        # the lightweight path is faster and more compatible than achat_structured
        # (which injects schema descriptions and uses json_mode/function_calling,
        # adding latency that easily exceeds the 1s timeout on DashScope).
        from miniclaw.utils.jsonx import extract_json_object
        raw_response = await asyncio.wait_for(
            provider.achat(messages, model=model or None, tools=None),
            timeout=timeout_s,
        )
        raw_text = str(raw_response.content or "")
        parsed = extract_json_object(raw_text, default={})
        response = RewriteResponseSchema(
            rewritten_query=str(parsed.get("rewritten_query", "")),
            keywords=parsed.get("keywords", []) if isinstance(parsed.get("keywords"), list) else [],
            intent=str(parsed.get("intent", "ambiguous")),
        )
    except asyncio.TimeoutError:
        return _fallback(inputs, reason="timeout", latency_ms=_ms(start))
    except Exception as exc:
        logger.warning("rewrite provider error: %s", exc)
        return _fallback(inputs, reason=f"provider_error: {exc}", latency_ms=_ms(start))

    latency_ms = _ms(start)
    rewritten = (response.rewritten_query or "").strip()
    if not rewritten:
        return _fallback(inputs, reason="missing_rewritten_query", latency_ms=latency_ms)

    keywords = tuple(k.strip() for k in response.keywords if k.strip())[:8]
    intent_raw = (response.intent or "ambiguous").strip()
    intent: Intent = (
        intent_raw
        if intent_raw in ("recall_prior", "new_topic", "direct_task", "ambiguous")
        else "ambiguous"
    )

    return RewriteResult(
        rewritten_query=rewritten[:200],
        keywords=keywords,
        intent=intent,
        used_llm=True,
        latency_ms=latency_ms,
        raw_response=str(response)[:500],
    )


def _fallback(
    inputs: RewriteInput,
    *,
    reason: str,
    latency_ms: int = 0,
) -> RewriteResult:
    return RewriteResult(
        rewritten_query=inputs.user_input[:200],
        keywords=(),
        intent="ambiguous",
        used_llm=False,
        latency_ms=latency_ms,
        failure_reason=reason,
    )


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


_SYSTEM = (
    "You are a memory retrieval assistant. "
    "Classify the user's intent and rewrite their message into a self-contained retrieval query. "
    "Respond in JSON format."
)

_USER_TEMPLATE = """\
## Recent conversation (last {n} exchanges, may be empty)
{recent_exchanges}

## User's latest message
{user_input}

Determine intent (recall_prior / new_topic / direct_task / ambiguous), \
rewrite the query resolving pronouns, and extract 3-6 search keywords."""


def _build_messages(inputs: RewriteInput) -> list[dict[str, str]]:
    exchanges_text = _format_exchanges(inputs.recent_exchanges)
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                n=len(inputs.recent_exchanges),
                recent_exchanges=exchanges_text or "(no prior exchanges)",
                user_input=inputs.user_input.strip()[:500],
            ),
        },
    ]


def _format_exchanges(exchanges: tuple[tuple[str, str], ...]) -> str:
    if not exchanges:
        return ""
    lines = []
    for user_text, assistant_text in exchanges:
        lines.append(f"User: {user_text.strip()[:200]}")
        if assistant_text.strip():
            lines.append(f"Assistant: {assistant_text.strip()[:200]}")
    return "\n".join(lines)
