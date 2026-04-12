from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlite_vec

from miniclaw.memory.embedding import OllamaEmbedder

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
        date_range: tuple[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        query_vec = await self.embedder.embed_one(query)

        fts_ranked = self._fts_search(query)
        vec_ranked = self._vec_search(query_vec)

        all_chunk_ids = {cid for cid, _ in fts_ranked} | {cid for cid, _ in vec_ranked}
        if not all_chunk_ids:
            return []

        chunk_meta = self._load_chunk_metadata(all_chunk_ids)

        # Apply metadata filters
        if thread_id is not None:
            chunk_meta = {
                cid: meta for cid, meta in chunk_meta.items()
                if meta.get("thread_id") == thread_id
            }
        if date_range is not None:
            start, end = date_range
            chunk_meta = {
                cid: meta for cid, meta in chunk_meta.items()
                if start <= _extract_date(meta) <= end
            }

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
            parsed_metadata = json.loads(meta.get("metadata_json", "{}"))
            if not isinstance(parsed_metadata, dict):
                parsed_metadata = {}
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=meta["content"],
                    score=scores[chunk_id],
                    source_file=meta["source_file"],
                    kind=meta["kind"],
                    created_at=meta["created_at"],
                    metadata=parsed_metadata,
                )
            )
        return results

    def _fts_search(self, query: str) -> list[tuple[str, int]]:
        fts_query = _prepare_fts_query(query)
        if not fts_query:
            return []
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
                SELECT id, content, source_file, thread_id, kind, created_at, metadata_json
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
            }
            for row in rows
        }

    def _vec_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn


def _prepare_fts_query(query: str) -> str:
    tokens = query.split()
    if not tokens:
        return ""
    escaped = [token.replace('"', '""') for token in tokens]
    return " OR ".join(f'"{t}"' for t in escaped)


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
    metadata_json = meta.get("metadata_json", "{}")
    try:
        parsed = json.loads(metadata_json)
        return parsed.get("date", "")
    except (json.JSONDecodeError, TypeError):
        return ""
