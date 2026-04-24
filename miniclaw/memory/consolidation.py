"""LLM-driven memory consolidation. Raises on failure; caller routes to fallback."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from miniclaw.memory.files import MemoryFileStore

if TYPE_CHECKING:
    from miniclaw.memory.indexer import MemoryIndexer
    from miniclaw.providers.contracts import ChatProvider

logger = logging.getLogger(__name__)
__all__ = [
    "ConsolidationResponseSchema",
    "ConsolidationResult",
    "llm_consolidate",
    "regex_consolidate_fallback",
]


# --- Pydantic schema for structured output --------------------------------

class _FactEntry(BaseModel):
    fact: str = ""
    tier: str = Field(default="normal", description="critical | normal")
    reason: str = Field(default="", description="Why this tier")
    action: str = Field(default="add", description="add | update | skip_duplicate")
    updates_fact_id: int | None = None


class ConsolidationResponseSchema(BaseModel):
    """Pydantic schema for provider.achat_structured(). Exported for callers."""

    thread_narrative: str = Field(description="3-5 sentence summary of this segment")
    long_term_facts: list[_FactEntry] = Field(default_factory=list)
    trace: dict = Field(default_factory=lambda: {"confidence": 0.0})


# --- Result dataclass -----------------------------------------------------

@dataclass(frozen=True)
class ConsolidationResult:
    thread_narrative: str
    facts_added: int = 0
    facts_updated: int = 0
    facts_skipped: int = 0
    facts_critical: int = 0
    confidence: float = 0.0
    fallback_triggered: bool = False


# --- Core function --------------------------------------------------------

async def llm_consolidate(
    *,
    thread_id: str,
    provider: ChatProvider,
    model: str,
    memory_path: Path,
    digests: list[str],
    recent_exchanges: list[tuple[str, str]],
    daily_dir: Path | None = None,
    indexer: MemoryIndexer | None = None,
    critical_max: int = 12,
) -> ConsolidationResult:
    """Run LLM consolidation. RAISES on any failure."""
    store = MemoryFileStore(memory_path)
    document = store.read()
    crit = list(document.critical_preferences)
    norm = list(document.long_term_facts)
    messages = _build_messages(crit, norm, digests, recent_exchanges)

    # Prefer achat_structured (function-calling based); fall back to raw achat + parse
    achat_structured = getattr(provider, "achat_structured", None)
    if achat_structured is not None:
        parsed = await asyncio.wait_for(
            achat_structured(messages, schema=ConsolidationResponseSchema, model=model),
            timeout=30.0,
        )
    else:
        from miniclaw.utils.jsonx import extract_json_object
        resp = await asyncio.wait_for(
            provider.achat(messages, model=model, tools=None),
            timeout=30.0,
        )
        raw = str(resp.content or "")
        raw_parsed = extract_json_object(raw, default=None)
        if not raw_parsed or "thread_narrative" not in raw_parsed:
            raise ValueError(f"invalid consolidation JSON: {raw[:200]}")
        parsed = ConsolidationResponseSchema(**raw_parsed)

    narrative = (parsed.thread_narrative or "").strip()
    if not narrative:
        raise ValueError("empty thread_narrative")

    confidence = 0.0
    if isinstance(parsed.trace, dict):
        confidence = float(parsed.trace.get("confidence", 0.0))

    new_crit, new_norm = list(crit), list(norm)
    added = updated = skipped = n_crit = 0

    for entry in parsed.long_term_facts:
        fact = entry.fact.strip()
        if not fact:
            continue
        action = entry.action.strip()
        tier = entry.tier.strip()
        reason = entry.reason.strip()
        if action == "skip_duplicate":
            skipped += 1
            continue
        if tier == "critical" and not reason:
            tier = "normal"
        if action == "update":
            fid = entry.updates_fact_id
            if isinstance(fid, int) and 0 <= fid < len(new_crit) + len(new_norm):
                if fid < len(new_crit):
                    new_crit[fid] = fact
                else:
                    new_norm[fid - len(new_crit)] = fact
                updated += 1
            else:
                skipped += 1
            continue
        if tier == "critical":
            new_crit.append(fact)
            n_crit += 1
            while len(new_crit) > critical_max:
                new_crit.pop(0)
        else:
            new_norm.append(fact)
        added += 1

    store.update(
        critical_preferences=new_crit,
        long_term_facts=new_norm,
        recent_work={},
    )
    if daily_dir is not None:
        # Indexer is consolidation's concern, kept out of MemoryFileStore.
        journal_store = MemoryFileStore(memory_path, daily_dir=daily_dir)
        written = journal_store.append_to_daily_journal(
            thread_id=thread_id,
            narrative=narrative,
            source="consolidation",
        )
        if indexer is not None:
            try:
                indexer.mark_dirty(written.name)
            except Exception:
                pass
    return ConsolidationResult(
        thread_narrative=narrative, facts_added=added, facts_updated=updated,
        facts_skipped=skipped, facts_critical=n_crit, confidence=confidence,
    )


def regex_consolidate_fallback(
    *,
    thread_id: str,
    memory_path: Path,
    digests: list[str],
    daily_dir: Path | None = None,
) -> None:
    """Fallback when LLM consolidation fails. Writes a simple narrative built
    from the most recent digests to the daily MD. Does NOT touch MEMORY.md
    (no fact extraction in this branch).
    """
    if not digests:
        return
    if daily_dir is None:
        return
    narrative = "\n".join(f"- {d}" for d in digests[-5:])
    store = MemoryFileStore(memory_path, daily_dir=daily_dir)
    store.append_to_daily_journal(
        thread_id=thread_id,
        narrative=narrative,
        source="regex_fallback",
    )


# --- Helpers --------------------------------------------------------------

_SYSTEM = "Output strict JSON only. No prose. No code fences."

_USER_TEMPLATE = (
    "## Existing facts\n{facts}\n\n## Exchanges\n{exchanges}\n\n## Digests\n{digests}\n\n"
    "Classification: CRITICAL only if stable user preference, forgetting repeats mistake, "
    "applies broadly, stated by user. Never critical: task state, implementation details."
)


def _build_messages(
    crit: list[str], norm: list[str], digests: list[str], recent: list[tuple[str, str]],
) -> list[dict[str, str]]:
    facts = "".join(f"[id:{i}] [CRITICAL] {f}\n" for i, f in enumerate(crit))
    facts += "".join(f"[id:{len(crit)+i}] {f}\n" for i, f in enumerate(norm))
    exchanges = "".join(f"User: {u[:200]}\nAssistant: {a[:200]}\n" for u, a in recent)
    content = _USER_TEMPLATE.format(
        facts=facts or "(none)",
        exchanges=exchanges or "(none)",
        digests="\n".join(f"- {d}" for d in digests) or "(none)",
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": content},
    ]
