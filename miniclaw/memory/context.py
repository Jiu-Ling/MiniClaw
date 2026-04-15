from __future__ import annotations

from typing import TYPE_CHECKING

from miniclaw.memory.files import MemoryFileStore
from miniclaw.persistence.memory_store import MemoryStore
from miniclaw.utils.async_bridge import run_sync as _run_sync

if TYPE_CHECKING:
    from miniclaw.memory.rewrite import RewriteResult

MEMORY_SECTION_TITLE = "Memory"

# Rough estimate: 1 token ~ 4 chars for mixed Chinese/English
_CHARS_PER_TOKEN = 4

_MAX_MEMORY_SECTION_CHARS = 4000


def build_memory_context(
    store: MemoryStore,
    thread_id: str,
    *,
    memory_file: MemoryFileStore | None = None,
    retriever: object | None = None,
    user_input: str = "",
    memory_token_budget: int = 2000,
    rewrite: "RewriteResult | None" = None,
) -> str:
    resolved_memory_file = memory_file or getattr(store, "memory_file_store", None)
    sections: list[str] = []

    char_budget = memory_token_budget * _CHARS_PER_TOKEN
    facts_budget = char_budget // 2

    # 1. Long-term Facts — budget-capped, always resident
    if resolved_memory_file is not None:
        document = resolved_memory_file.read()
        if document.long_term_facts:
            facts_text = "\n".join(
                ["## Long-term Facts", *[f"- {fact}" for fact in document.long_term_facts]]
            )
            if len(facts_text) > facts_budget:
                facts_text = facts_text[:facts_budget] + "\n...[facts truncated]"
            sections.append(facts_text)

    # 2. Related Context — rewrite-driven retrieval
    if retriever is not None and user_input:
        intent = rewrite.intent if rewrite is not None else "ambiguous"
        if intent != "new_topic":
            query = rewrite.rewritten_query if rewrite is not None else user_input
            keywords = rewrite.keywords if rewrite is not None else ()
            top_k = 3 if intent == "direct_task" else 5
            search = getattr(retriever, "search", None)
            if callable(search):
                chunks = _run_sync(search(query, top_k=top_k, keywords=keywords))
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
                        sections.append("\n".join(["## Related Context", *assembled]))

    return "\n\n".join(sections)


def render_memory_section(memory_context: str) -> str:
    memory_context = memory_context.strip()
    if not memory_context:
        return ""
    if len(memory_context) > _MAX_MEMORY_SECTION_CHARS:
        memory_context = memory_context[:_MAX_MEMORY_SECTION_CHARS] + "\n...[memory truncated]"
    return f"## {MEMORY_SECTION_TITLE}\n{memory_context}"
