from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import sqlite_vec

from miniclaw.memory.embedding import OllamaEmbedder
from miniclaw.utils.jsonx import safe_loads_dict

_RRF_K = 60
_RECALL_LIMIT = 50


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    content: str
    score: float
    source_file: str
    kind: str
    created_at: str
    metadata: dict[str, Any]
    parent_id: str | None = None


class HybridRetriever:
    def __init__(self, db_path: Path, embedder: OllamaEmbedder) -> None:
        self.db_path = db_path
        self.embedder = embedder

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        thread_id: str | None = None,
        thread_scope: Literal["current", "global", "current_then_global"] = "current_then_global",
        keywords: tuple[str, ...] = (),
        date_range: tuple[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        query_vec = await self.embedder.embed_one(query)

        fts_ranked = self._fts_search(query, keywords)
        vec_ranked = self._vec_search(query_vec)

        all_chunk_ids = {cid for cid, _ in fts_ranked} | {cid for cid, _ in vec_ranked}
        if not all_chunk_ids:
            return []

        chunk_meta = self._load_chunk_metadata(all_chunk_ids)

        # Apply date_range filter
        if date_range is not None:
            start, end = date_range
            chunk_meta = {
                cid: meta for cid, meta in chunk_meta.items()
                if start <= _extract_date(meta) <= end
            }

        def _apply_thread_filter(meta_dict: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
            """Filter by current thread, but always keep long_term_fact chunks."""
            return {
                cid: meta for cid, meta in meta_dict.items()
                if meta.get("thread_id") == thread_id or meta.get("kind") == "long_term_fact"
            }

        if thread_scope == "global" or thread_id is None:
            filtered_meta = chunk_meta
        elif thread_scope == "current":
            filtered_meta = _apply_thread_filter(chunk_meta)
        else:  # current_then_global
            filtered_meta = _apply_thread_filter(chunk_meta)

        results = self._score_and_build(filtered_meta, fts_ranked, vec_ranked, top_k=top_k)

        # current_then_global fallback: if fewer than top_k, widen to global and merge
        if thread_scope == "current_then_global" and thread_id is not None and len(results) < top_k:
            seen_ids = {r.chunk_id for r in results}
            global_results = self._score_and_build(chunk_meta, fts_ranked, vec_ranked, top_k=top_k)
            for r in global_results:
                if r.chunk_id not in seen_ids:
                    results.append(r)
                    seen_ids.add(r.chunk_id)
                if len(results) >= top_k:
                    break

        return results[:top_k]

    def _score_and_build(
        self,
        chunk_meta: dict[str, dict[str, Any]],
        fts_ranked: list[tuple[str, int]],
        vec_ranked: list[tuple[str, int]],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        allowed_ids = set(chunk_meta.keys())
        fts_filtered = [(cid, rank) for cid, rank in fts_ranked if cid in allowed_ids]
        vec_filtered = [(cid, rank) for cid, rank in vec_ranked if cid in allowed_ids]

        scores = _rrf_fuse(fts_filtered, vec_filtered)

        # Filter out low-relevance noise before selecting top-k
        _MIN_RRF_SCORE = 0.01
        scores = {cid: s for cid, s in scores.items() if s >= _MIN_RRF_SCORE}

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

        results: list[RetrievedChunk] = []
        for chunk_id in sorted_ids:
            meta = chunk_meta[chunk_id]
            parsed_metadata = safe_loads_dict(meta.get("metadata_json", "{}"))
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=meta["content"],
                    score=scores[chunk_id],
                    source_file=meta["source_file"],
                    kind=meta["kind"],
                    created_at=meta["created_at"],
                    metadata=parsed_metadata,
                    parent_id=meta.get("parent_id"),
                )
            )
        return results

    def _fts_search(
        self,
        query: str,
        keywords: tuple[str, ...] = (),
    ) -> list[tuple[str, int]]:
        tokens = list(query.split())
        for kw in keywords:
            if kw and kw not in tokens:
                tokens.append(kw)
        if not tokens:
            return []
        escaped = [t.replace('"', '""') for t in tokens]
        fts_query = " OR ".join(f'"{t}"' for t in escaped)
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT chunk_id, rank FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fts_query, _RECALL_LIMIT),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(row[0], idx) for idx, row in enumerate(rows)]

    def _vec_search(self, query_vec: list[float]) -> list[tuple[str, int]]:
        vec_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)
        try:
            with self._vec_conn() as conn:
                rows = conn.execute(
                    "SELECT chunk_id, distance FROM memory_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                    (vec_bytes, _RECALL_LIMIT),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(row[0], idx) for idx, row in enumerate(rows)]

    def _load_chunk_metadata(self, chunk_ids: set[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT id, content, source_file, thread_id, kind, created_at, metadata_json, parent_id
                FROM memory_chunks
                WHERE id IN ({placeholders})
                """,
                list(chunk_ids),
            ).fetchall()
        return {
            row[0]: {
                "content": row[1],
                "source_file": row[2],
                "thread_id": row[3],
                "kind": row[4],
                "created_at": row[5],
                "metadata_json": row[6],
                "parent_id": row[7],
            }
            for row in rows
        }

    def _vec_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def load_parent(self, parent_id: str) -> str | None:
        """Fetch the full content of a parent chunk by id. None if missing or table absent."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT content FROM memory_parents WHERE id = ?",
                    (parent_id,),
                ).fetchone()
        except sqlite3.OperationalError:
            return None
        return row[0] if row else None

    def load_neighbors(
        self,
        chunk: RetrievedChunk,
        *,
        radius: int = 1,
    ) -> list[str]:
        """Return chunk.content plus +/- radius neighbor siblings within the same parent."""
        if not chunk.parent_id:
            return [chunk.content]
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT content, chunk_index FROM memory_chunks
                WHERE parent_id = ?
                ORDER BY chunk_index
                """,
                (chunk.parent_id,),
            ).fetchall()
        if not rows:
            return [chunk.content]
        contents = [r[0] for r in rows]
        try:
            target_idx = next(i for i, r in enumerate(rows) if r[0] == chunk.content)
        except StopIteration:
            return [chunk.content]
        lo = max(0, target_idx - radius)
        hi = min(len(rows), target_idx + radius + 1)
        return contents[lo:hi]


def _rrf_fuse(
    fts_ranked: list[tuple[str, int]],
    vec_ranked: list[tuple[str, int]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for chunk_id, rank in fts_ranked:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
    for chunk_id, rank in vec_ranked:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
    return scores


def _extract_date(meta: dict[str, Any]) -> str:
    parsed = safe_loads_dict(meta.get("metadata_json", "{}"))
    return str(parsed.get("date", ""))


def assemble_adaptive(
    matched: list[RetrievedChunk],
    *,
    budget_chars: int,
    parent_loader: Callable[[str], str | None],
    neighbor_loader: Callable[[RetrievedChunk], list[str]],
) -> list[str]:
    """Cluster matched children by parent; return prompt-ready strings.

    Strategy:
      - Group children by parent_id (orphans = long_term_fact with parent_id None)
      - Rank parents by (hit_count DESC, total_score DESC)
      - Parent with >=2 hits -> return full parent content (or joined hits as fallback)
      - Parent with 1 hit -> return neighbor window (child + neighbors)
      - Fit remaining budget with orphan facts ranked by score
      - Stop when budget_chars would be exceeded
    """
    parent_groups: dict[str, list[RetrievedChunk]] = {}
    orphans: list[RetrievedChunk] = []
    for ch in matched:
        if ch.parent_id:
            parent_groups.setdefault(ch.parent_id, []).append(ch)
        else:
            orphans.append(ch)

    ranked = sorted(
        parent_groups.items(),
        key=lambda kv: (len(kv[1]), sum(c.score for c in kv[1])),
        reverse=True,
    )

    out: list[str] = []
    used = 0

    for parent_id, hits in ranked:
        if len(hits) >= 2:
            text = parent_loader(parent_id)
            if text is None:
                text = "\n".join(h.content for h in hits)
        else:
            neighbors = neighbor_loader(hits[0])
            text = "\n".join(neighbors)
        if used + len(text) > budget_chars:
            break
        out.append(text)
        used += len(text)

    for ch in sorted(orphans, key=lambda c: c.score, reverse=True):
        if used + len(ch.content) > budget_chars:
            break
        out.append(ch.content)
        used += len(ch.content)

    return out
