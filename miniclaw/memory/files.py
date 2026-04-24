from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def compress_memory_entry(text: str, max_chars: int = 200) -> str:
    """Compress a memory entry by stripping tables, code blocks, and truncating.

    Goal: a single weather forecast table (20+ lines) becomes "weather: 24C cloudy".
    """
    if not text:
        return text
    # Strip markdown tables (lines starting/containing |...|)
    lines = text.splitlines()
    lines = [ln for ln in lines if not re.match(r"^\s*\|", ln)]
    # Strip code blocks
    in_code = False
    cleaned: list[str] = []
    for ln in lines:
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            cleaned.append(ln)
    # Collapse multiple blank lines
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


@dataclass(slots=True)
class MemoryDocument:
    critical_preferences: list[str] = field(default_factory=list)
    long_term_facts: list[str] = field(default_factory=list)
    recent_work: dict[str, list[str]] = field(default_factory=dict)


class MemoryFileStore:
    def __init__(
        self,
        path: Path,
        *,
        recent_work_limit: int = 3,
        daily_dir: Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.recent_work_limit = recent_work_limit
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
        recent_work: dict[str, list[str]],
    ) -> None:
        with self._lock:
            document = MemoryDocument(
                critical_preferences=self._normalize_lines(critical_preferences or []),
                long_term_facts=self._normalize_lines(long_term_facts),
                recent_work={
                    thread_id.strip(): self._normalize_lines(entries)[-self.recent_work_limit :]
                    for thread_id, entries in recent_work.items()
                    if thread_id.strip()
                },
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self._render(document), encoding="utf-8")

    def append_recent_work(self, thread_key: str, entry: str) -> None:
        thread_key = thread_key.strip()
        entry = compress_memory_entry(entry.strip())
        if not thread_key or not entry:
            return

        with self._lock:
            document = self.read()
            entries = list(document.recent_work.get(thread_key, []))
            entries.append(entry)
            document.recent_work[thread_key] = entries[-self.recent_work_limit :]
            self.update(
                critical_preferences=document.critical_preferences,
                long_term_facts=document.long_term_facts,
                recent_work=document.recent_work,
            )

    def merge_long_term_facts(self, facts: list[str]) -> None:
        with self._lock:
            document = self.read()
            merged = list(
                dict.fromkeys(
                    [
                        *document.long_term_facts,
                        *self._normalize_lines(facts),
                    ]
                )
            )
            self.update(
                critical_preferences=document.critical_preferences,
                long_term_facts=merged,
                recent_work=document.recent_work,
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
        narratives (recent_work in MEMORY.md is removed in Task 9). The
        daily directory is indexed by FTS5+vec via MemoryIndexer; entries
        are retrievable through memory_search.

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
            if not path.exists():
                path.write_text(f"# {today}\n", encoding="utf-8")
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    f"\n### thread:{thread_id}\n<!-- source: {source} -->\n{narrative}\n"
                )
            return path

    def _parse(self, text: str) -> MemoryDocument:
        critical_preferences: list[str] = []
        long_term_facts: list[str] = []
        recent_work: dict[str, list[str]] = {}
        current_thread: str | None = None
        in_critical = False
        in_long_term = False
        in_recent = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line == "## Critical Preferences":
                in_critical = True
                in_long_term = False
                in_recent = False
                current_thread = None
                continue
            if line == "## Long-term Facts":
                in_critical = False
                in_long_term = True
                in_recent = False
                current_thread = None
                continue
            if line == "## Recent Work":
                in_critical = False
                in_long_term = False
                in_recent = True
                current_thread = None
                continue
            if line.startswith("### "):
                current_thread = line.removeprefix("### ").strip()
                if current_thread:
                    recent_work.setdefault(current_thread, [])
                continue
            if line == "# Memory":
                continue
            if line == "-":
                continue
            if line.startswith("- "):
                fact = line.removeprefix("- ").strip()
                # Strip [id:N] prefix written by consolidation
                if fact.startswith("[id:") and "] " in fact:
                    fact = fact.split("] ", 1)[1]
                if in_critical:
                    critical_preferences.append(fact)
                elif in_long_term:
                    long_term_facts.append(fact)
                elif in_recent and current_thread:
                    recent_work[current_thread].append(fact)

        return MemoryDocument(
            critical_preferences=critical_preferences,
            long_term_facts=long_term_facts,
            recent_work=recent_work,
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

        lines.extend(["", "## Recent Work"])
        if document.recent_work:
            for thread_id in sorted(document.recent_work):
                lines.extend(["", f"### {thread_id}"])
                entries = document.recent_work[thread_id]
                if entries:
                    lines.extend(f"- {entry}" for entry in entries)
                else:
                    lines.append("-")
        else:
            lines.append("-")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _normalize_lines(lines: list[str]) -> list[str]:
        return [line.strip() for line in lines if line.strip()]
