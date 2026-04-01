from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Chunk:
    content: str
    chunk_index: int
    kind: str
    metadata: dict[str, Any]


_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
_THREAD_PREFIX = "thread:"


def chunk_daily_file(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    date = _extract_date(path, text)
    sections = _split_sections(text)
    chunks: list[Chunk] = []
    for index, (heading, body) in enumerate(sections):
        thread_id = _extract_thread_id(heading)
        content = f"### {heading}\n{body}".strip()
        chunks.append(
            Chunk(
                content=content,
                chunk_index=index,
                kind="daily_summary",
                metadata={"date": date, "thread_id": thread_id},
            )
        )
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on `### thread:` headings only.

    Other `### ` headings inside summary content are treated as body text,
    not as new sections.
    """
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("### thread:"):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line.removeprefix("### ").strip()
            current_lines = []
        elif current_heading is not None:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections


def _extract_date(path: Path, text: str) -> str:
    match = _DATE_PATTERN.search(path.stem)
    if match:
        return match.group(0)
    for line in text.splitlines()[:3]:
        match = _DATE_PATTERN.search(line)
        if match:
            return match.group(0)
    return ""


def _extract_thread_id(heading: str) -> str | None:
    if heading.startswith(_THREAD_PREFIX):
        return heading.removeprefix(_THREAD_PREFIX).strip() or None
    return None
