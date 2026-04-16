"""LLM-driven memory consolidation. Raises on failure; caller routes to fallback."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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
    crit, norm, recent = _read_memory(memory_path)
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

    new_recent = {**recent, f"thread:{thread_id}": [narrative]}
    _write_memory(memory_path, new_crit, new_norm, new_recent)
    if daily_dir is not None:
        _write_daily(daily_dir, thread_id, narrative, indexer)
    return ConsolidationResult(
        thread_narrative=narrative, facts_added=added, facts_updated=updated,
        facts_skipped=skipped, facts_critical=n_crit, confidence=confidence,
    )


def regex_consolidate_fallback(*, thread_id: str, memory_path: Path, digests: list[str]) -> None:
    """Minimal fallback: append digests to recent_work. No fact extraction."""
    if not digests:
        return
    store = MemoryFileStore(memory_path)
    doc = store.read()
    key = f"thread:{thread_id}"
    entries = list(doc.recent_work.get(key, [])) + digests[-3:]
    doc.recent_work[key] = entries[-3:]
    store.update(long_term_facts=doc.long_term_facts, recent_work=doc.recent_work)


# --- Helpers --------------------------------------------------------------

def _read_memory(path: Path) -> tuple[list[str], list[str], dict[str, list[str]]]:
    if not path.is_file():
        return [], [], {}
    crit: list[str] = []
    norm: list[str] = []
    recent: dict[str, list[str]] = {}
    section = thread_key = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s == "## Critical Preferences":
            section = "c"
        elif s == "## Long-term Facts":
            section = "n"
        elif s == "## Recent Work":
            section = "r"
        elif s.startswith("## "):
            section = None
        elif section == "r" and s.startswith("### "):
            thread_key = s.removeprefix("### ").strip()
            recent.setdefault(thread_key, [])
        elif s.startswith("- ") and s != "-":
            fact = s.removeprefix("- ").strip()
            if fact.startswith("[id:"):
                fact = fact.split("] ", 1)[-1]
            if section == "c":
                crit.append(fact)
            elif section == "n":
                norm.append(fact)
            elif section == "r" and thread_key:
                recent[thread_key].append(fact)
    return crit, norm, recent


def _write_memory(
    path: Path, crit: list[str], norm: list[str], recent: dict[str, list[str]]
) -> None:
    lines = ["# Memory", "", "## Critical Preferences"]
    lines += [f"- [id:{i}] {f}" for i, f in enumerate(crit)] or ["-"]
    lines += ["", "## Long-term Facts"]
    lines += [f"- [id:{len(crit)+i}] {f}" for i, f in enumerate(norm)] or ["-"]
    lines += ["", "## Recent Work"]
    if recent:
        for key in sorted(recent):
            lines += ["", f"### {key}"]
            lines += ([f"- {e}" for e in recent[key]] or ["-"])
    else:
        lines.append("-")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_daily(daily_dir: Path, thread_id: str, narrative: str, indexer: MemoryIndexer | None) -> None:
    daily_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = daily_dir / f"{today}.md"
    if not path.exists():
        path.write_text(f"# {today}\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n### thread:{thread_id}\n{narrative}\n")
    if indexer is not None:
        try:
            indexer.mark_dirty(f"{today}.md")
        except Exception:
            pass


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
