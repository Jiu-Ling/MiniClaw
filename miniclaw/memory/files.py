from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class MemoryDocument:
    critical_preferences: list[str] = field(default_factory=list)
    long_term_facts: list[str] = field(default_factory=list)


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
