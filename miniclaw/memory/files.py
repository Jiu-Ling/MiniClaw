from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class MemoryDocument:
    critical_preferences: list[str] = field(default_factory=list)
    long_term_facts: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FactCandidate:
    """A candidate fact considered for write into MEMORY.md.

    `tier=critical` requires non-empty `reason`. Caller responsibility.
    `source` is free-form; common values: "agent", "compression", "user".
    """

    fact: str
    tier: Literal["critical", "normal"] = "normal"
    source: str = "agent"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AddFactsResult:
    added: int
    skipped_duplicates: int
    evicted_critical: int
    final_critical_count: int
    final_long_term_count: int


def _normalize_for_dedup(value: str) -> str:
    """Normalize a fact for exact-string dedup (case-fold + whitespace collapse)."""
    return " ".join(value.casefold().split())


class MemoryFileStore:
    def __init__(
        self,
        path: Path,
        *,
        daily_dir: Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.daily_dir = Path(daily_dir) if daily_dir is not None else self.path.parent / "daily"
        self._lock = threading.RLock()

    def read(self) -> MemoryDocument:
        if not self.path.is_file():
            return MemoryDocument()

        text = self.path.read_text(encoding="utf-8")
        if not text.strip():
            return MemoryDocument()
        return self._parse(text)

    def update(
        self,
        *,
        critical_preferences: list[str] | None = None,
        long_term_facts: list[str],
    ) -> None:
        with self._lock:
            document = MemoryDocument(
                critical_preferences=self._normalize_lines(critical_preferences or []),
                long_term_facts=self._normalize_lines(long_term_facts),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self._render(document), encoding="utf-8")

    def add_fact(
        self,
        fact: str,
        *,
        tier: Literal["critical", "normal"] = "normal",
        source: str = "agent",
        reason: str = "",
        dedup: bool = True,
    ) -> bool:
        """Idempotent single-fact write.

        Returns True if the fact was added, False if skipped (empty or duplicate).
        Dedup checks across BOTH critical_preferences and long_term_facts using
        normalized exact-string match (`_normalize_for_dedup`).

        `source` and `reason` are recorded in trace events but not persisted to MEMORY.md.
        Critical-tier capping is enforced by `add_facts_batch`; this single-fact
        API does not auto-evict — callers wanting capping should batch.
        """
        fact = fact.strip()
        if not fact:
            return False

        with self._lock:
            doc = self.read()
            if dedup:
                normalized = _normalize_for_dedup(fact)
                existing = (
                    _normalize_for_dedup(f)
                    for f in (*doc.critical_preferences, *doc.long_term_facts)
                )
                if any(e == normalized for e in existing):
                    return False

            new_critical = list(doc.critical_preferences)
            new_long_term = list(doc.long_term_facts)
            if tier == "critical":
                new_critical.append(fact)
            else:
                new_long_term.append(fact)

            self.update(
                critical_preferences=new_critical,
                long_term_facts=new_long_term,
            )
            return True

    def add_facts_batch(
        self,
        candidates: list[FactCandidate],
        *,
        dedup_against_existing: bool = True,
        critical_max: int | None = None,
    ) -> AddFactsResult:
        """Bulk-write facts in a single read-modify-write cycle.

        Dedups WITHIN the batch and (if `dedup_against_existing`) against
        existing MEMORY.md contents. Empty facts count as skipped_duplicates.

        If `critical_max` is set and added critical facts push the count past
        it, the oldest critical facts are evicted FIFO until the cap is met.
        Eviction only runs when at least one critical fact was added.
        """
        with self._lock:
            doc = self.read()
            new_critical = list(doc.critical_preferences)
            new_long_term = list(doc.long_term_facts)

            seen: set[str] = set()
            if dedup_against_existing:
                seen = {
                    _normalize_for_dedup(f)
                    for f in (*new_critical, *new_long_term)
                }

            added = 0
            skipped = 0
            critical_added = 0
            for candidate in candidates:
                fact = candidate.fact.strip()
                if not fact:
                    skipped += 1
                    continue
                normalized = _normalize_for_dedup(fact)
                if normalized in seen:
                    skipped += 1
                    continue
                seen.add(normalized)
                if candidate.tier == "critical":
                    new_critical.append(fact)
                    critical_added += 1
                else:
                    new_long_term.append(fact)
                added += 1

            evicted = 0
            if critical_max is not None and critical_added > 0:
                while len(new_critical) > critical_max:
                    new_critical.pop(0)
                    evicted += 1

            if added > 0 or evicted > 0:
                self.update(
                    critical_preferences=new_critical,
                    long_term_facts=new_long_term,
                )

            return AddFactsResult(
                added=added,
                skipped_duplicates=skipped,
                evicted_critical=evicted,
                final_critical_count=len(new_critical),
                final_long_term_count=len(new_long_term),
            )

    def append_to_daily_journal(
        self,
        thread_id: str,
        narrative: str,
        *,
        source: str = "consolidation",
    ) -> Path:
        """Append a per-thread narrative to today's daily MD.

        Daily MDs are the canonical persistent storage for conversation
        narratives. The daily directory is indexed by FTS5+vec via
        MemoryIndexer; entries are retrievable through memory_search.

        Returns the file path written.
        """
        thread_id = thread_id.strip()
        narrative = narrative.strip()
        if not thread_id or not narrative:
            raise ValueError("thread_id and narrative are required")

        with self._lock:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = self.daily_dir / f"{today}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                if f.tell() == 0:
                    f.write(f"# {today}\n")
                f.write(
                    f"\n### thread:{thread_id}\n<!-- source: {source} -->\n{narrative}\n"
                )
            return path

    def _parse(self, text: str) -> MemoryDocument:
        critical_preferences: list[str] = []
        long_term_facts: list[str] = []
        in_critical = False
        in_long_term = False
        in_skipped_section = False  # ## Recent Work or any other unknown section

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line == "## Critical Preferences":
                in_critical = True
                in_long_term = False
                in_skipped_section = False
                continue
            if line == "## Long-term Facts":
                in_critical = False
                in_long_term = True
                in_skipped_section = False
                continue
            if line.startswith("## "):
                # Any other section (including legacy "## Recent Work") is silently skipped.
                in_critical = False
                in_long_term = False
                in_skipped_section = True
                continue
            if line == "# Memory":
                continue
            if line == "-":
                continue
            if line.startswith("- ") and not in_skipped_section:
                fact = line.removeprefix("- ").strip()
                if fact.startswith("[id:") and "] " in fact:
                    fact = fact.split("] ", 1)[1]
                if in_critical:
                    critical_preferences.append(fact)
                elif in_long_term:
                    long_term_facts.append(fact)

        return MemoryDocument(
            critical_preferences=critical_preferences,
            long_term_facts=long_term_facts,
        )

    def _render(self, document: MemoryDocument) -> str:
        lines = ["# Memory", "", "## Critical Preferences"]
        if document.critical_preferences:
            lines.extend(f"- {fact}" for fact in document.critical_preferences)
        else:
            lines.append("-")

        lines.extend(["", "## Long-term Facts"])
        if document.long_term_facts:
            lines.extend(f"- {fact}" for fact in document.long_term_facts)
        else:
            lines.append("-")

        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _normalize_lines(lines: list[str]) -> list[str]:
        return [line.strip() for line in lines if line.strip()]
