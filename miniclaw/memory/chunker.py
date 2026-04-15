"""Parent-child chunking for memory files.

Parents preserve semantic boundaries (one per `### thread:xxx` section per
daily file, hard-capped at 2000 chars; longer sections split on paragraph
boundaries). Children are ~250-char overlap-windowed slices of parents, used
as the retrieval unit for FTS + vec search. The retriever clusters matched
children by parent_id to reconstruct context.

MEMORY.md long-term facts are chunked as atomic children with no parent;
each `- fact` bullet is one self-contained retrieval unit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

_PARENT_MAX_CHARS = 2000
_CHILD_TARGET_CHARS = 250
_CHILD_MAX_CHARS = 400
_CHILD_OVERLAP_CHARS = 50

_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
_THREAD_PREFIX = "thread:"

_CHILD_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_CHILD_TARGET_CHARS,
    chunk_overlap=_CHILD_OVERLAP_CHARS,
    length_function=len,
    separators=[
        "\n\n",
        "\n",
        "。", "！", "？",
        ". ", "! ", "? ",
        "，", ", ",
        " ",
        "",
    ],
    keep_separator=True,
)


@dataclass(frozen=True)
class ParentChunk:
    """A semantic narrative unit stored in memory_parents.

    id: stable UUID assigned at chunk time; used as FK target for children
        and as the lookup key in assemble_adaptive.
    source_file: basename of the source file (e.g., "2026-04-15.md").
    parent_index: ordinal within source_file, for deterministic ordering.
    heading: the `### thread:xxx` heading or a synthetic label.
    content: full narrative text of this parent section.
    kind: "daily_summary" for daily-md content.
    metadata: {"date": "YYYY-MM-DD", "thread_id": str | None}.
    """

    id: str
    source_file: str
    parent_index: int
    heading: str
    content: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChildChunk:
    """A fine-grained retrieval unit stored in memory_chunks.

    content: the child's text (overlap with neighbors by ~50 chars).
    chunk_index: ordinal within the parent (starting at 0).
    kind: "daily_summary" or "long_term_fact".
    parent_id: FK to memory_parents.id, or None for atomic facts.
    metadata: {"date": ..., "thread_id": ...} inherited from parent.
    """

    content: str
    chunk_index: int
    kind: str
    parent_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_daily_file(path: Path) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Parse a daily md file into parents and children.

    Splits on `### thread:xxx` headings (one parent per heading). Sections
    exceeding 2000 chars are further split on paragraph boundaries into
    multiple parents. Each parent is then divided into overlap-windowed
    children via RecursiveCharacterTextSplitter.
    """
    text = path.read_text(encoding="utf-8")
    date = _extract_date(path, text)
    source_file = path.name

    sections = _split_thread_sections(text)
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    parent_index = 0

    for heading, body in sections:
        thread_id = _extract_thread_id(heading)
        metadata = {"date": date, "thread_id": thread_id}

        parent_texts = _split_parent_text(body, max_chars=_PARENT_MAX_CHARS)
        for parent_text in parent_texts:
            parent = ParentChunk(
                id=str(uuid4()),
                source_file=source_file,
                parent_index=parent_index,
                heading=heading,
                content=parent_text,
                kind="daily_summary",
                metadata=dict(metadata),
            )
            parents.append(parent)
            parent_index += 1

            child_texts = _CHILD_SPLITTER.split_text(parent_text)
            for idx, child_text in enumerate(child_texts):
                if not child_text.strip():
                    continue
                children.append(
                    ChildChunk(
                        content=child_text,
                        chunk_index=idx,
                        kind="daily_summary",
                        parent_id=parent.id,
                        metadata=dict(metadata),
                    )
                )

    return parents, children


def chunk_memory_file(path: Path) -> list[ChildChunk]:
    """Parse MEMORY.md into atomic long-term-fact children.

    Each bullet under `## Long-term Facts` or `## Critical Preferences`
    becomes one ChildChunk with parent_id=None. Tier is recorded in metadata.
    """
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    children: list[ChildChunk] = []
    current_section: str | None = None
    idx = 0
    for raw in text.splitlines():
        line = raw.strip()
        if line == "## Critical Preferences":
            current_section = "critical"
            continue
        if line == "## Long-term Facts":
            current_section = "long_term"
            continue
        if line.startswith("## "):
            current_section = None
            continue
        if current_section is None:
            continue
        if not line.startswith("- "):
            continue
        fact_text = line.removeprefix("- ").strip()
        if not fact_text or fact_text == "-":
            continue
        children.append(
            ChildChunk(
                content=fact_text,
                chunk_index=idx,
                kind="long_term_fact",
                parent_id=None,
                metadata={"tier": current_section},
            )
        )
        idx += 1
    return children


# --- helpers ---------------------------------------------------------------

def _split_thread_sections(text: str) -> list[tuple[str, str]]:
    """Split on `### thread:` headings only. Other headings are body text."""
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("### thread:"):
            if current_heading is not None:
                sections.append(
                    (current_heading, "\n".join(current_lines).strip())
                )
            current_heading = line.removeprefix("### ").strip()
            current_lines = []
        elif current_heading is not None:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections


def _split_parent_text(body: str, *, max_chars: int) -> list[str]:
    """Split body into parents of at most max_chars, preferring paragraph gaps."""
    body = body.strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [body]

    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    parents: list[str] = []
    buf = ""
    for para in paragraphs:
        if not buf:
            buf = para
            continue
        candidate = f"{buf}\n\n{para}"
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            parents.append(buf)
            buf = para

    if buf:
        parents.append(buf)

    # Handle pathological cases where a single paragraph exceeds max_chars:
    # split it hard by character count.
    expanded: list[str] = []
    for p in parents:
        if len(p) <= max_chars:
            expanded.append(p)
            continue
        start = 0
        while start < len(p):
            end = min(start + max_chars, len(p))
            expanded.append(p[start:end])
            start = end
    return expanded


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
