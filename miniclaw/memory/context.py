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

    char_budget = memory_token_budget * _CHARS_PER_TOKEN
    facts_budget = char_budget // 2  # Facts get at most half the budget

    # 1. Long-term Facts — budget-capped (was "always injected" with no limit)
    if resolved_memory_file is not None:
        document = resolved_memory_file.read()
        if document.long_term_facts:
            facts_text = "\n".join(
                ["## Long-term Facts", *[f"- {fact}" for fact in document.long_term_facts]]
            )
            if len(facts_text) > facts_budget:
                facts_text = facts_text[:facts_budget] + "\n...[facts truncated]"
            sections.append(facts_text)

    # 2. Related Context — adaptive parent-child assembly
    if retriever is not None and user_input:
        search = getattr(retriever, "search", None)
        if callable(search):
            chunks = _run_sync(search(user_input, top_k=5))
            if chunks:
                from miniclaw.memory.retriever import assemble_adaptive
                parent_loader = getattr(retriever, "load_parent", None)
                neighbor_loader = getattr(retriever, "load_neighbors", None)
                if callable(parent_loader) and callable(neighbor_loader):
                    assembled = assemble_adaptive(
                        chunks,
                        budget_chars=char_budget,
                        parent_loader=parent_loader,
                        neighbor_loader=lambda ch: neighbor_loader(ch, radius=1),
                    )
                else:
                    assembled = [c.content for c in chunks]
                if assembled:
                    sections.append(
                        "\n".join(["## Related Context", *assembled])
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


_MAX_MEMORY_SECTION_CHARS = 4000


def render_memory_section(memory_context: str) -> str:
    memory_context = memory_context.strip()
    if not memory_context:
        return ""
    if len(memory_context) > _MAX_MEMORY_SECTION_CHARS:
        memory_context = memory_context[:_MAX_MEMORY_SECTION_CHARS] + "\n...[memory truncated]"
    return f"## {MEMORY_SECTION_TITLE}\n{memory_context}"


def _format_item(item: MemoryItem) -> str:
    line = f"- [{item.kind}] {item.content}"
    if item.metadata:
        line += f" {json.dumps(_normalize_metadata(item.metadata), ensure_ascii=False, sort_keys=True)}"
    return line


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in metadata.items()}



# _run_sync imported from miniclaw.utils.async_bridge
