"""Orchestrate background compression promotion: extract → write facts → write daily MD.

Called from `_trim_history`'s `on_compression` callback (Phase 5 Task 3).
Composes `llm_extract_facts` + `MemoryFileStore.add_facts_batch` +
`append_to_daily_journal` into a single BackgroundScheduler job. Failures
are swallowed (logged) so a flaky LLM never crashes the foreground.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from miniclaw.memory.extract import (
    PinnedReferences,
    llm_extract_facts,
)
from miniclaw.memory.files import MemoryFileStore
from miniclaw.providers.contracts import ChatMessage
from miniclaw.runtime.background import BackgroundJob, BackgroundScheduler

if TYPE_CHECKING:
    from miniclaw.memory.indexer import MemoryIndexer
    from miniclaw.observability.contracts import TraceContext
    from miniclaw.providers.contracts import ChatProvider

logger = logging.getLogger(__name__)

__all__ = ["CompressionEvent", "schedule_compression_promotion"]


@dataclass(frozen=True, slots=True)
class CompressionEvent:
    """Snapshot of a compression event passed to the on_compression callback."""

    dropped_messages: list[ChatMessage]
    pinned_references: PinnedReferences
    thread_id: str
    parent_trace: "TraceContext | None"


def schedule_compression_promotion(
    *,
    scheduler: BackgroundScheduler,
    provider: "ChatProvider",
    model: str,
    memory_store: MemoryFileStore,
    indexer: "MemoryIndexer | None",
    daily_dir: Path,
    event: CompressionEvent,
    timeout_s: float = 15.0,
    critical_max: int | None = None,
) -> None:
    """Submit a fire-and-forget background job that LLM-extracts and persists facts.

    The job:
      1. calls llm_extract_facts(provider, dropped_messages, pinned)
      2. add_facts_batch(facts_to_remember + discovered_facts)
      3. append_to_daily_journal(thread_id, narrative, source="compression")
      4. indexer.mark_dirty(daily_md_path) if indexer provided

    Failures are logged but never re-raised. Pinned references are already in
    the summary message (Phase 4 sync path), so a failed extraction only loses
    the LLM-derived narrative + semantic facts, not the literal references.
    """

    def _do_promotion() -> None:
        try:
            extraction = asyncio.run(
                llm_extract_facts(
                    provider=provider,
                    model=model,
                    compressed_messages=list(event.dropped_messages),
                    pinned_references=event.pinned_references,
                    timeout_s=timeout_s,
                )
            )
        except Exception as exc:
            logger.warning(
                "compression promotion: llm_extract_facts failed thread=%s err=%s",
                event.thread_id,
                exc,
            )
            return

        candidates = list(extraction.facts_to_remember) + list(extraction.discovered_facts)
        if candidates:
            try:
                memory_store.add_facts_batch(
                    candidates=candidates,
                    dedup_against_existing=True,
                    critical_max=critical_max,
                )
            except Exception as exc:
                logger.warning(
                    "compression promotion: add_facts_batch failed thread=%s err=%s",
                    event.thread_id,
                    exc,
                )

        narrative = (extraction.narrative or "").strip()
        if narrative:
            try:
                written = memory_store.append_to_daily_journal(
                    thread_id=event.thread_id,
                    narrative=narrative,
                    source="compression",
                )
                if indexer is not None:
                    try:
                        indexer.mark_dirty(str(written))
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning(
                    "compression promotion: journal write failed thread=%s err=%s",
                    event.thread_id,
                    exc,
                )

    job = BackgroundJob(
        fn=_do_promotion,
        kind="memory.compression_promote",
        metadata={"thread_id": event.thread_id},
        parent_trace=event.parent_trace,
    )
    scheduler.submit(job)
