from __future__ import annotations

import json
import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sqlite_vec

from miniclaw.memory.chunker import Chunk, chunk_daily_file
from miniclaw.memory.embedding import OllamaEmbedder


class MemoryIndexer:
    def __init__(
        self,
        db_path: Path,
        embedder: OllamaEmbedder,
        memory_dir: Path,
    ) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.memory_dir = memory_dir

    def mark_dirty(self, file_path: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory_dirty_files (file_path, dirty_at) VALUES (?, ?)",
                (file_path, now),
            )

    async def flush_dirty(self) -> None:
        # Auto-detect unindexed files (exist on disk but not in memory_chunks)
        self._mark_unindexed_files()

        with sqlite3.connect(self.db_path) as conn:
            dirty_rows = conn.execute("SELECT file_path FROM memory_dirty_files").fetchall()

        for (file_path,) in dirty_rows:
            await self._process_file(file_path)

    def _mark_unindexed_files(self) -> None:
        """Mark any daily md files that exist on disk but have no chunks indexed."""
        if not self.memory_dir.is_dir():
            return
        with sqlite3.connect(self.db_path) as conn:
            indexed_files = {
                row[0] for row in conn.execute(
                    "SELECT DISTINCT source_file FROM memory_chunks"
                ).fetchall()
            }
            already_dirty = {
                row[0] for row in conn.execute(
                    "SELECT file_path FROM memory_dirty_files"
                ).fetchall()
            }
        now = datetime.now(timezone.utc).isoformat()
        for md_file in self.memory_dir.glob("*.md"):
            file_name = md_file.name
            if file_name not in indexed_files and file_name not in already_dirty:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO memory_dirty_files (file_path, dirty_at) VALUES (?, ?)",
                        (file_name, now),
                    )

    async def rebuild_all(self) -> None:
        with self._vec_conn() as conn:
            conn.execute("DELETE FROM memory_chunks")
            conn.execute("DELETE FROM memory_fts")
            conn.execute("DELETE FROM memory_vec")
            conn.execute("DELETE FROM memory_dirty_files")

        for md_file in sorted(self.memory_dir.glob("*.md")):
            await self._process_file(md_file.name)

    async def _process_file(self, file_path: str) -> None:
        full_path = self.memory_dir / file_path
        if not full_path.is_file():
            self._clear_dirty(file_path)
            return

        new_chunks = chunk_daily_file(full_path)
        old_chunks = self._load_existing_chunks(file_path)
        old_by_index = {c["chunk_index"]: c for c in old_chunks}
        new_by_index = {c.chunk_index: c for c in new_chunks}

        to_delete = [old_by_index[i] for i in old_by_index if i not in new_by_index]
        to_add = [new_by_index[i] for i in new_by_index if i not in old_by_index]
        to_update = [
            (old_by_index[i], new_by_index[i])
            for i in new_by_index
            if i in old_by_index and old_by_index[i]["content"] != new_by_index[i].content
        ]

        # Embed all new/changed content in one batch
        texts_to_embed = [c.content for c in to_add] + [new.content for _, new in to_update]
        embeddings: list[list[float]] = []
        if texts_to_embed:
            embeddings = await self.embedder.embed(texts_to_embed)

        now = datetime.now(timezone.utc).isoformat()
        embed_idx = 0

        with self._vec_conn() as conn:
            self._ensure_vec_table(conn)

            for old in to_delete:
                self._delete_chunk(conn, old["id"])

            for chunk in to_add:
                chunk_id = str(uuid.uuid4())
                self._insert_chunk(
                    conn,
                    chunk_id=chunk_id,
                    source_file=file_path,
                    chunk=chunk,
                    embedding=embeddings[embed_idx],
                    model_name=self.embedder.model,
                    now=now,
                )
                embed_idx += 1

            for old, new in to_update:
                self._update_chunk(
                    conn,
                    chunk_id=old["id"],
                    chunk=new,
                    embedding=embeddings[embed_idx],
                    model_name=self.embedder.model,
                    now=now,
                )
                embed_idx += 1

        self._clear_dirty(file_path)

    def _ensure_vec_table(self, conn: sqlite3.Connection) -> None:
        """Recreate memory_vec with the embedder's actual dimension if it differs."""
        dims = self.embedder.dims
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_vec'"
        ).fetchall()
        # vec0 virtual tables may not appear in sqlite_master the same way;
        # use the shadow tables to detect existing dimension.
        existing_dims: int | None = None
        try:
            row = conn.execute("SELECT embedding FROM memory_vec LIMIT 1").fetchone()
            if row is not None:
                existing_dims = len(struct.unpack(f"{len(row[0]) // 4}f", row[0]))
        except sqlite3.OperationalError:
            existing_dims = None

        # If we have data and dimensions match, nothing to do
        if existing_dims is not None and existing_dims == dims:
            return

        # If table has no data or dimension mismatch, recreate
        count = conn.execute("SELECT count(*) FROM memory_vec").fetchone()[0]
        if count == 0:
            # Recreate with correct dims
            conn.execute("DROP TABLE IF EXISTS memory_vec")
            conn.execute(
                f"CREATE VIRTUAL TABLE memory_vec USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[{dims}])"
            )

    def _load_existing_chunks(self, source_file: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, chunk_index, content FROM memory_chunks WHERE source_file = ?",
                (source_file,),
            ).fetchall()
        return [{"id": row[0], "chunk_index": row[1], "content": row[2]} for row in rows]

    def _insert_chunk(
        self,
        conn: sqlite3.Connection,
        *,
        chunk_id: str,
        source_file: str,
        chunk: Chunk,
        embedding: list[float],
        model_name: str,
        now: str,
    ) -> None:
        thread_id = chunk.metadata.get("thread_id")
        metadata_json = json.dumps(chunk.metadata, ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO memory_chunks
                (id, source_file, thread_id, chunk_index, content, kind, created_at, updated_at, model_name, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, source_file, thread_id, chunk.chunk_index, chunk.content, chunk.kind, now, now, model_name, metadata_json),
        )
        conn.execute(
            "INSERT INTO memory_fts (chunk_id, content) VALUES (?, ?)",
            (chunk_id, chunk.content),
        )
        conn.execute(
            "INSERT INTO memory_vec (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, _serialize_vec(embedding)),
        )

    def _update_chunk(
        self,
        conn: sqlite3.Connection,
        *,
        chunk_id: str,
        chunk: Chunk,
        embedding: list[float],
        model_name: str,
        now: str,
    ) -> None:
        metadata_json = json.dumps(chunk.metadata, ensure_ascii=False)
        thread_id = chunk.metadata.get("thread_id")
        conn.execute(
            """
            UPDATE memory_chunks
            SET content = ?, kind = ?, thread_id = ?, updated_at = ?, model_name = ?, metadata_json = ?
            WHERE id = ?
            """,
            (chunk.content, chunk.kind, thread_id, now, model_name, metadata_json, chunk_id),
        )
        conn.execute("DELETE FROM memory_fts WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            "INSERT INTO memory_fts (chunk_id, content) VALUES (?, ?)",
            (chunk_id, chunk.content),
        )
        conn.execute(
            "UPDATE memory_vec SET embedding = ? WHERE chunk_id = ?",
            (_serialize_vec(embedding), chunk_id),
        )

    def _delete_chunk(self, conn: sqlite3.Connection, chunk_id: str) -> None:
        conn.execute("DELETE FROM memory_chunks WHERE id = ?", (chunk_id,))
        conn.execute("DELETE FROM memory_fts WHERE chunk_id = ?", (chunk_id,))
        conn.execute("DELETE FROM memory_vec WHERE chunk_id = ?", (chunk_id,))

    def _clear_dirty(self, file_path: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memory_dirty_files WHERE file_path = ?", (file_path,))

    def _vec_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn


def _serialize_vec(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)
