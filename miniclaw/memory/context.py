from __future__ import annotations

import json
from typing import Any

from miniclaw.memory.files import MemoryFileStore
from miniclaw.persistence.memory_store import MemoryItem, MemoryStore
from miniclaw.utils.async_bridge import run_sync as _run_sync

MEMORY_SECTION_TITLE = "Memory"

# Rough estimate: 1 token ~ 4 chars for mixed Chinese/English
_CHARS_PER_TOKEN = 4


def build_memory_context(
    store: MemoryStore,
    thread_id: str,
    *,
    memory_file: MemoryFileStore | None = None,
    retriever: object | None = None,
    user_input: str = "",
    memory_token_budget: int = 2000,
    limit: int = 5,
) -> str:
    resolved_memory_file = memory_file or getattr(store, "memory_file_store", None)
    sections: list[str] = []

    # 1. Long-term Facts — always injected
    if resolved_memory_file is not None:
        document = resolved_memory_file.read()
        if document.long_term_facts:
            sections.append(
                "\n".join(
                    ["## Long-term Facts", *[f"- {fact}" for fact in document.long_term_facts]]
                )
            )

    # 2. Related Context — on-demand retrieval
    if retriever is not None and user_input:
        search = getattr(retriever, "search", None)
        if callable(search):
            chunks = _run_sync(search(user_input, top_k=20))
            if chunks:
                char_budget = memory_token_budget * _CHARS_PER_TOKEN
                retrieved_lines: list[str] = []
                used_chars = 0
                for chunk in chunks:
                    chunk_chars = len(chunk.content)
                    if used_chars + chunk_chars > char_budget:
                        break
                    retrieved_lines.append(f"[{chunk.created_at[:10]}] {chunk.content}")
                    used_chars += chunk_chars
                if retrieved_lines:
                    sections.append(
                        "\n".join(["## Related Context", *retrieved_lines])
                    )

    # 3. Fallback: if no retriever and no memory_file, use old behavior
    if not sections:
        items = store.list_recent(thread_id, limit=limit)
        if items:
            lines = ["Relevant memory:"]
            for item in reversed(items):
                lines.append(_format_item(item))
            return "\n".join(lines)

    # 4. Backwards-compatible: if memory_file provided but no retriever,
    #    also emit Recent Work and Thread Context (legacy behavior)
    if resolved_memory_file is not None and retriever is None:
        document = resolved_memory_file.read()
        thread_key = f"thread:{thread_id}"
        work_items = document.recent_work.get(thread_key, [])
        if work_items:
            sections.append(
                "\n".join(["## Recent Work", f"### {thread_key}", *[f"- {item}" for item in work_items]])
            )

        items = store.list_recent(thread_id, limit=limit)
        if items:
            sections.append("\n".join(["## Thread Context", *[_format_item(item) for item in reversed(items)]]))

    return "\n\n".join(sections)


def render_memory_section(memory_context: str) -> str:
    memory_context = memory_context.strip()
    if not memory_context:
        return ""
    return f"## {MEMORY_SECTION_TITLE}\n{memory_context}"


def _format_item(item: MemoryItem) -> str:
    line = f"- [{item.kind}] {item.content}"
    if item.metadata:
        line += f" {json.dumps(_normalize_metadata(item.metadata), ensure_ascii=False, sort_keys=True)}"
    return line


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in metadata.items()}



# _run_sync imported from miniclaw.utils.async_bridge
