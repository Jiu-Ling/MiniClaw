"""Compression-time literal extraction.

Provides synchronous regex-based extraction of paths, URLs, and identifiers
from compressed conversation messages. Used by `_trim_history` to populate
a `## Pinned References` block in the summary message — preserving verbatim
references that would otherwise be lost to truncation.

Phase 5 adds `CompressionExtraction` + `llm_extract_facts` for LLM-driven
structured extraction of narratives and durable facts from compressed messages.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from miniclaw.memory.files import FactCandidate
from miniclaw.providers.contracts import ChatMessage

if TYPE_CHECKING:
    from miniclaw.providers.contracts import ChatProvider

__all__ = [
    "PinnedReferences",
    "extract_pinned_references",
    "CompressionExtraction",
    "llm_extract_facts",
]

_MAX_PER_CATEGORY = 20

# File extension list shared by path regexes.
_EXTS = r"(?:py|md|json|yaml|yml|toml|js|ts|tsx|jsx|html|css|sh|sql|txt|cfg|ini|env)"

_PATH_RE = re.compile(
    r"(?<![='\"\w])"            # not preceded by = ' " or word char
    r"(?:"
    # Absolute POSIX paths with known extension, optional :lineno
    r"(?:/[\w.\-/]+\." + _EXTS + r"(?::\d+)?)"
    r"|"
    # Relative paths starting with ./ or ../
    r"(?:\.{1,2}/[\w.\-/]+\." + _EXTS + r"(?::\d+)?)"
    r"|"
    # Windows-style paths (keep for cross-platform support)
    r"(?:[A-Za-z]:\\[\w.\\\-]+\." + _EXTS + r"(?::\d+)?)"
    r")"
)

_URL_RE = re.compile(r"https?://[^\s<>\)\]\"']+[^\s<>\)\]\".,;'!?]")

# Matches: lowercase_func() or Module.method (no trailing parens required for dotted)
_FUNC_CALL_RE = re.compile(
    r"(?<![.\w])"
    r"("
    r"[a-z_][a-z0-9_]*\(\)"           # snake_case_func()
    r"|[A-Z][A-Za-z0-9]+\.[a-z_][a-z0-9_]+"  # ClassName.method_name
    r")"
)

# ALL_CAPS_WITH_UNDERSCORE constants (at least one underscore, all caps)
_CONST_RE = re.compile(r"(?<!\w)([A-Z][A-Z0-9]*_[A-Z0-9_]+)(?!\w)")

# Fenced code blocks — captures the body only
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class PinnedReferences:
    """Verbatim references extracted from compressed messages."""

    paths: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    code_blocks: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.paths or self.urls or self.identifiers or self.code_blocks)

    def render_markdown(self) -> str:
        """Render as a `## Pinned References` markdown section. Empty string if no refs."""
        if self.is_empty():
            return ""
        lines: list[str] = ["## Pinned References (verbatim)"]
        for items in (self.paths, self.urls, self.identifiers):
            for item in items:
                lines.append(f"- {item}")
        if self.code_blocks:
            lines.append("")
            lines.append("Code excerpts:")
            for block in self.code_blocks:
                snippet = block.strip()
                if len(snippet) > 500:
                    snippet = snippet[:500] + "\n...[truncated]"
                lines.append(f"```\n{snippet}\n```")
        return "\n".join(lines)


def _iter_message_text(messages: Iterable[ChatMessage]) -> Iterable[str]:
    for msg in messages:
        if msg.content:
            yield msg.content
        for part in msg.content_parts or []:
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, str):
                yield text


def _dedup_capped(items: Iterable[str], cap: int = _MAX_PER_CATEGORY) -> list[str]:
    """Return deduplicated list preserving insertion order, capped at `cap` entries."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= cap:
            break
    return result


def extract_pinned_references(messages: Iterable[ChatMessage]) -> PinnedReferences:
    """Scan messages for verbatim references worth preserving in compression summary.

    Pure function: regex-based, no I/O, no LLM. Designed for sync use inside
    `_trim_history`. Caps each category at 20 entries (5 for code blocks) to
    keep summaries bounded.
    """
    raw_paths: list[str] = []
    raw_urls: list[str] = []
    raw_idents: list[str] = []
    raw_blocks: list[str] = []

    for text in _iter_message_text(messages):
        raw_paths.extend(_PATH_RE.findall(text))
        raw_urls.extend(_URL_RE.findall(text))
        raw_idents.extend(_FUNC_CALL_RE.findall(text))
        raw_idents.extend(_CONST_RE.findall(text))
        raw_blocks.extend(_CODE_BLOCK_RE.findall(text))

    return PinnedReferences(
        paths=_dedup_capped(raw_paths),
        urls=_dedup_capped(raw_urls),
        identifiers=_dedup_capped(raw_idents),
        code_blocks=_dedup_capped(raw_blocks, cap=5),
    )


# ---------------------------------------------------------------------------
# Phase 5: LLM-driven structured extraction
# ---------------------------------------------------------------------------

class CompressionExtraction(BaseModel):
    """LLM-extracted structured summary of a compressed message batch.

    `narrative` — 3-5 sentence overview written to daily MD.
    `facts_to_remember` — decision/preference facts for MEMORY.md.
    `discovered_facts` — tool-result-derived facts (api endpoints, paths, etc).
    Both lists pass through `add_facts_batch` with dedup.
    """

    narrative: str = Field(default="", description="3-5 sentence narrative summary")
    facts_to_remember: list[FactCandidate] = Field(default_factory=list)
    decisions_made: list[str] = Field(default_factory=list)
    open_threads: list[str] = Field(default_factory=list)
    discovered_facts: list[FactCandidate] = Field(default_factory=list)


_LLM_EXTRACT_SYSTEM = """\
You are extracting durable knowledge from compressed conversation turns.

Rules:
- `narrative`: 3-5 sentences summarizing what happened. Past tense.
- `facts_to_remember`: explicit user preferences, project decisions, naming conventions.
  Use tier="critical" ONLY for stable user preferences with `reason`. Default tier="normal".
- `discovered_facts`: facts the agent learned from tool results (API endpoints, paths,
  port numbers, version strings). source="compression".
- Skip transient state (in-progress task names, "currently doing X").
- Each fact <= 200 chars, narrative <= 500 chars.

Output strict JSON matching the schema. No prose outside JSON."""


def _format_messages_for_extraction(
    messages: list[ChatMessage],
    pinned: PinnedReferences,
) -> list[ChatMessage]:
    """Build [system, user] messages for the extraction LLM call."""
    system = ChatMessage(role="system", content=_LLM_EXTRACT_SYSTEM)
    body_parts: list[str] = []
    if not pinned.is_empty():
        body_parts.append("## Pinned References (already preserved verbatim)")
        for p in pinned.paths:
            body_parts.append(f"- path: {p}")
        for u in pinned.urls:
            body_parts.append(f"- url: {u}")
        for ident in pinned.identifiers:
            body_parts.append(f"- ident: {ident}")
        body_parts.append("")
    body_parts.append("## Compressed messages")
    for m in messages:
        text = m.content or "".join(
            p.get("text", "") for p in (m.content_parts or []) if isinstance(p, dict)
        )
        body_parts.append(f"[{m.role}] {text[:1000]}")
    user = ChatMessage(role="user", content="\n".join(body_parts))
    return [system, user]


async def llm_extract_facts(
    *,
    provider: "ChatProvider",
    model: str,
    compressed_messages: list[ChatMessage],
    pinned_references: PinnedReferences,
    timeout_s: float = 15.0,
) -> CompressionExtraction:
    """Run an LLM extraction over compressed messages. Raises on provider error.

    Caller (compression_promote) wraps this in try/except — failures degrade
    to "no LLM extraction this round; pinned refs already in summary".
    """
    achat_structured = getattr(provider, "achat_structured", None)
    if achat_structured is None:
        raise RuntimeError("provider does not support achat_structured")
    messages = _format_messages_for_extraction(compressed_messages, pinned_references)
    return await asyncio.wait_for(
        achat_structured(messages, schema=CompressionExtraction, model=model),
        timeout=timeout_s,
    )
