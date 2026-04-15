"""LLM-driven query rewrite for memory retrieval.

Called from load_context before hitting the retriever. On any failure
(no provider / timeout / bad JSON / missing required field) returns a
RewriteResult flagged with used_llm=False so the caller can degrade to
raw user input. Never raises, never blocks more than timeout_s seconds.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal

from miniclaw.providers.contracts import ChatMessage, ChatProvider
from miniclaw.utils.jsonx import extract_json_object

logger = logging.getLogger(__name__)

Intent = Literal["recall_prior", "new_topic", "direct_task", "ambiguous"]

__all__ = ["RewriteInput", "RewriteResult", "rewrite_query"]


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
    provider: ChatProvider | None,
    model: str,
    timeout_s: float = 1.0,
) -> RewriteResult:
    """Rewrite user_input into a retrieval query with intent. Never raises."""
    if provider is None:
        return _fallback(inputs, reason="no_provider")
    if not inputs.user_input.strip():
        return _fallback(inputs, reason="empty_input")

    prompt = _build_prompt(inputs)
    start = time.monotonic()
    try:
        response = await asyncio.wait_for(
            provider.achat(prompt, model=model, tools=None),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        return _fallback(inputs, reason="timeout", latency_ms=_ms(start))
    except Exception as exc:
        logger.warning("rewrite provider error: %s", exc)
        return _fallback(inputs, reason=f"provider_error: {exc}", latency_ms=_ms(start))

    latency_ms = _ms(start)
    raw = str(getattr(response, "content", "") or "")
    parsed = extract_json_object(raw, default={})

    rewritten = str(parsed.get("rewritten_query", "")).strip()
    if not rewritten:
        return _fallback(inputs, reason="missing_rewritten_query", latency_ms=latency_ms, raw=raw)

    keywords_raw = parsed.get("keywords", [])
    if not isinstance(keywords_raw, list):
        keywords_raw = []
    keywords = tuple(str(k).strip() for k in keywords_raw if str(k).strip())[:8]

    intent_raw = str(parsed.get("intent", "ambiguous")).strip()
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
        raw_response=raw[:500],
    )


def _fallback(
    inputs: RewriteInput,
    *,
    reason: str,
    latency_ms: int = 0,
    raw: str = "",
) -> RewriteResult:
    return RewriteResult(
        rewritten_query=inputs.user_input[:200],
        keywords=(),
        intent="ambiguous",
        used_llm=False,
        latency_ms=latency_ms,
        raw_response=raw,
        failure_reason=reason,
    )


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


_PROMPT_TEMPLATE = """You are a memory retrieval assistant. Given the user's latest message and a \
tiny slice of recent conversation, produce a retrieval query and classify the intent. \
Output ONLY a single JSON object, no prose, no code fences.

## Recent conversation (last {n} exchanges, may be empty)
{recent_exchanges}

## User's latest message
{user_input}

## Task
1. intent ∈ {{recall_prior, new_topic, direct_task, ambiguous}}
2. rewritten_query: self-contained, resolves pronouns, <100 chars
3. keywords: 3-6 nouns/identifiers, no stopwords, no pronouns, mix zh/en

## Output (strict JSON, no other text)
{{"rewritten_query": "string", "keywords": ["kw"], "intent": "new_topic"}}
"""


def _build_prompt(inputs: RewriteInput) -> list[ChatMessage]:
    exchanges_text = _format_exchanges(inputs.recent_exchanges)
    user_content = _PROMPT_TEMPLATE.format(
        n=len(inputs.recent_exchanges),
        recent_exchanges=exchanges_text or "(no prior exchanges)",
        user_input=inputs.user_input.strip()[:500],
    )
    return [
        ChatMessage(role="system", content="You output strict JSON. No prose. No code fences."),
        ChatMessage(role="user", content=user_content),
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
