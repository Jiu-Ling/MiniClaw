from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from miniclaw.memory.files import MemoryFileStore
from miniclaw.persistence.memory_store import MemoryStore
from miniclaw.utils.async_bridge import run_sync as _run_sync

if TYPE_CHECKING:
    from miniclaw.memory.retriever import HybridRetriever
    from miniclaw.memory.rewrite import RewriteResult

# Rough estimate: 1 token ~ 4 chars for mixed Chinese/English
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """Structured memory context separating cache-eligible from volatile parts."""

    critical_preferences: list[str] = field(default_factory=list)
    long_term_facts: list[str] = field(default_factory=list)
    related_context: str = ""

    def is_empty(self) -> bool:
        return (
            not self.critical_preferences
            and not self.long_term_facts
            and not self.related_context.strip()
        )


def build_memory_context(
    store: MemoryStore,
    thread_id: str,
    *,
    memory_file: MemoryFileStore | None = None,
    retriever: HybridRetriever | None = None,
    user_input: str = "",
    memory_token_budget: int = 2000,
    rewrite: "RewriteResult | None" = None,
) -> MemoryContext:
    resolved_memory_file = memory_file or getattr(store, "memory_file_store", None)
    char_budget = memory_token_budget * _CHARS_PER_TOKEN

    critical: list[str] = []
    long_term: list[str] = []
    if resolved_memory_file is not None:
        document = resolved_memory_file.read()
        critical = list(document.critical_preferences)
        long_term = list(document.long_term_facts)

    related = ""
    if retriever is not None and user_input:
        intent = rewrite.intent if rewrite is not None else "ambiguous"
        if intent != "new_topic":
            query = rewrite.rewritten_query if rewrite is not None else user_input
            keywords = rewrite.keywords if rewrite is not None else ()
            top_k = 3 if intent == "direct_task" else 5
            chunks = _run_sync(retriever.search(query, top_k=top_k, keywords=keywords))
            if chunks:
                from miniclaw.memory.retriever import assemble_adaptive
                if hasattr(retriever, "load_parent") and hasattr(retriever, "load_neighbors"):
                    assembled = assemble_adaptive(
                        chunks,
                        budget_chars=char_budget,
                        parent_loader=retriever.load_parent,
                        neighbor_loader=lambda ch: retriever.load_neighbors(ch, radius=1),
                    )
                else:
                    assembled = [c.content for c in chunks]
                if assembled:
                    related = "\n".join(assembled)

    return MemoryContext(
        critical_preferences=critical,
        long_term_facts=long_term,
        related_context=related,
    )
