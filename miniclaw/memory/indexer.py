from __future__ import annotations

import json
import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import sqlite_vec

import re as _re

from miniclaw.memory.chunker import (
    ChildChunk,
    ParentChunk,
    chunk_daily_file,
    chunk_memory_file,
)
from miniclaw.memory.embedding import OllamaEmbedder

_MAX_CHUNK_INDEX_CHARS = 500


def _clean_for_indexing(content: str) -> str:
    """Strip noise (tables, code blocks) and truncate before embedding."""
    # Strip markdown tables
    lines = content.splitlines()
    lines = [ln for ln in lines if not _re.match(r"^\s*\|", ln)]
    # Strip code blocks
    in_code = False
    cleaned: list[str] = []
    for ln in lines:
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            cleaned.append(ln)
    text = "\n".join(cleaned)
    text = _re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > _MAX_CHUNK_INDEX_CHARS:
        text = text[:_MAX_CHUNK_INDEX_CHARS]
    return text


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

    def _ensure_schema(self) -> None:
        """Create memory_parents and memory_chunks.parent_id if missing.

        Idempotent and safe to call every flush. Does NOT migrate old data;
        pre-existing chunks with parent_id=NULL remain queryable but have
        no parent reference.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_parents (
                    id            TEXT PRIMARY KEY,
                    source_file   TEXT NOT NULL,
                    parent_index  INTEGER NOT NULL,
                    heading       TEXT,
                    content       TEXT NOT NULL,
                    kind          TEXT NOT NULL,
                    created_at    TEXT NOT NULL,
                    metadata_json TEXT,
                    UNIQUE(source_file, parent_index)
                )
                """
            )
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info('memory_chunks')")
            }
            if "parent_id" not in cols:
                conn.execute("ALTER TABLE memory_chunks ADD COLUMN parent_id TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_parent ON memory_chunks(parent_id)"
            )

    def mark_dirty(self, file_path: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory_dirty_files (file_path, dirty_at) VALUES (?, ?)",
                (file_path, now),
            )

    async def flush_dirty(self) -> None:
        self._ensure_schema()
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
        self._ensure_schema()
        with self._vec_conn() as conn:
            conn.execute("DELETE FROM memory_chunks")
            conn.execute("DELETE FROM memory_fts")
            conn.execute("DELETE FROM memory_vec")
            conn.execute("DELETE FROM memory_dirty_files")

        # Also clear parents on full rebuild
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memory_parents")

        for md_file in sorted(self.memory_dir.glob("*.md")):
            await self._process_file(md_file.name)

    async def _process_file(self, file_path: str) -> None:
        self._ensure_schema()
        full_path = self.memory_dir / file_path
        if not full_path.is_file():
            self._clear_dirty(file_path)
            return

        # MEMORY.md is parsed differently: atomic facts, no parents
        if full_path.name == "MEMORY.md":
            new_children = chunk_memory_file(full_path)
            await self._write_atomic_children(file_path, new_children)
            self._clear_dirty(file_path)
            return

        new_parents, new_children = chunk_daily_file(full_path)

        now = datetime.now(timezone.utc).isoformat()

        # --- parents ---
        old_parents = self._load_existing_parents(file_path)
        old_parent_by_idx = {p["parent_index"]: p for p in old_parents}
        new_parent_by_idx = {p.parent_index: p for p in new_parents}

        to_delete_parents = [
            old_parent_by_idx[i]
            for i in old_parent_by_idx
            if i not in new_parent_by_idx
        ]
        to_add_parents = [
            new_parent_by_idx[i]
            for i in new_parent_by_idx
            if i not in old_parent_by_idx
        ]
        to_update_parents = [
            (old_parent_by_idx[i], new_parent_by_idx[i])
            for i in new_parent_by_idx
            if i in old_parent_by_idx
            and old_parent_by_idx[i]["content"] != new_parent_by_idx[i].content
        ]

        # Reuse existing parent IDs where possible (for FK stability)
        # Map new parents' generated IDs to existing IDs when unchanged or updated.
        parent_id_remap: dict[str, str] = {}
        for i, new_p in new_parent_by_idx.items():
            if i in old_parent_by_idx:
                parent_id_remap[new_p.id] = old_parent_by_idx[i]["id"]

        # --- children ---
        # Clean content before embedding (same as before)
        cleaned_children = [
            ChildChunk(
                content=_clean_for_indexing(c.content),
                chunk_index=c.chunk_index,
                kind=c.kind,
                parent_id=parent_id_remap.get(c.parent_id, c.parent_id) if c.parent_id else None,
                metadata=c.metadata,
            )
            for c in new_children
        ]
        # Drop children whose cleaned content is empty
        cleaned_children = [c for c in cleaned_children if c.content.strip()]

        old_children = self._load_existing_chunks(file_path)
        # Key children by (parent_id, chunk_index) for daily_summary, or (None, idx)
        old_child_by_key = {
            (c.get("parent_id"), c["chunk_index"]): c for c in old_children
        }
        new_child_by_key = {
            (c.parent_id, c.chunk_index): c for c in cleaned_children
        }

        to_delete_children = [
            old_child_by_key[k]
            for k in old_child_by_key
            if k not in new_child_by_key
        ]
        to_add_children = [
            new_child_by_key[k]
            for k in new_child_by_key
            if k not in old_child_by_key
        ]
        to_update_children = [
            (old_child_by_key[k], new_child_by_key[k])
            for k in new_child_by_key
            if k in old_child_by_key
            and old_child_by_key[k]["content"] != new_child_by_key[k].content
        ]

        # Embed new/changed children in one batch
        texts_to_embed = [c.content for c in to_add_children] + [
            new.content for _, new in to_update_children
        ]
        embeddings: list[list[float]] = []
        if texts_to_embed:
            embeddings = await self.embedder.embed(texts_to_embed)

        with self._vec_conn() as conn:
            self._ensure_vec_table(conn)

            # Parents: delete then insert; updates are delete+insert for simplicity
            for old_p in to_delete_parents:
                self._delete_parent(conn, old_p["id"])
            for new_p in to_add_parents:
                self._insert_parent(conn, parent=new_p, source_file=file_path, now=now)
            for old_p, new_p in to_update_parents:
                # Keep the old id, update content/metadata in place
                self._update_parent(conn, parent_id=old_p["id"], new_p=new_p, now=now)

            # Children: old flow, but now carries parent_id
            embed_idx = 0
            for old_c in to_delete_children:
                self._delete_chunk(conn, old_c["id"])
            for new_c in to_add_children:
                chunk_id = str(uuid4())
                self._insert_chunk(
                    conn,
                    chunk_id=chunk_id,
                    source_file=file_path,
                    child=new_c,
                    embedding=embeddings[embed_idx],
                    model_name=self.embedder.model,
                    now=now,
                )
                embed_idx += 1
            for old_c, new_c in to_update_children:
                self._update_chunk(
                    conn,
                    chunk_id=old_c["id"],
                    child=new_c,
                    embedding=embeddings[embed_idx],
                    model_name=self.embedder.model,
                    now=now,
                )
                embed_idx += 1

        self._clear_dirty(file_path)

    async def _write_atomic_children(
        self, file_path: str, new_children: list[ChildChunk]
    ) -> None:
        """Rewrite the atomic-fact chunks for MEMORY.md.

        No parents to manage; every child has parent_id=None. Drop old
        MEMORY.md chunks and insert the new set in one pass.
        """
        now = datetime.now(timezone.utc).isoformat()

        cleaned = [
            ChildChunk(
                content=_clean_for_indexing(c.content),
                chunk_index=c.chunk_index,
                kind=c.kind,
                parent_id=None,
                metadata=c.metadata,
            )
            for c in new_children
            if c.content.strip()
        ]

        texts = [c.content for c in cleaned]
        embeddings: list[list[float]] = []
        if texts:
            embeddings = await self.embedder.embed(texts)

        with self._vec_conn() as conn:
            self._ensure_vec_table(conn)
            old_children = self._load_existing_chunks(file_path)
            for old_c in old_children:
                self._delete_chunk(conn, old_c["id"])
            for idx, new_c in enumerate(cleaned):
                chunk_id = str(uuid4())
                self._insert_chunk(
                    conn,
                    chunk_id=chunk_id,
                    source_file=file_path,
                    child=new_c,
                    embedding=embeddings[idx],
                    model_name=self.embedder.model,
                    now=now,
                )

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

    def _load_existing_parents(self, source_file: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, parent_index, content FROM memory_parents WHERE source_file = ?",
                (source_file,),
            ).fetchall()
        return [
            {"id": row[0], "parent_index": row[1], "content": row[2]}
            for row in rows
        ]

    def _load_existing_chunks(self, source_file: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, parent_id, chunk_index, content FROM memory_chunks WHERE source_file = ?",
                (source_file,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "parent_id": row[1],
                "chunk_index": row[2],
                "content": row[3],
            }
            for row in rows
        ]

    def _insert_parent(
        self,
        conn: sqlite3.Connection,
        *,
        parent: ParentChunk,
        source_file: str,
        now: str,
    ) -> None:
        metadata_json = json.dumps(parent.metadata, ensure_ascii=False)
        conn.execute(
            """
            INSERT OR REPLACE INTO memory_parents
                (id, source_file, parent_index, heading, content, kind, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parent.id,
                source_file,
                parent.parent_index,
                parent.heading,
                parent.content,
                parent.kind,
                now,
                metadata_json,
            ),
        )

    def _update_parent(
        self,
        conn: sqlite3.Connection,
        *,
        parent_id: str,
        new_p: ParentChunk,
        now: str,
    ) -> None:
        metadata_json = json.dumps(new_p.metadata, ensure_ascii=False)
        conn.execute(
            """
            UPDATE memory_parents
            SET heading = ?, content = ?, kind = ?, metadata_json = ?
            WHERE id = ?
            """,
            (new_p.heading, new_p.content, new_p.kind, metadata_json, parent_id),
        )

    def _delete_parent(self, conn: sqlite3.Connection, parent_id: str) -> None:
        conn.execute("DELETE FROM memory_parents WHERE id = ?", (parent_id,))

    def _insert_chunk(
        self,
        conn: sqlite3.Connection,
        *,
        chunk_id: str,
        source_file: str,
        child: ChildChunk,
        embedding: list[float],
        model_name: str,
        now: str,
    ) -> None:
        thread_id = child.metadata.get("thread_id")
        metadata_json = json.dumps(child.metadata, ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO memory_chunks
                (id, source_file, thread_id, chunk_index, content, kind, created_at, updated_at, model_name, metadata_json, parent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                source_file,
                thread_id,
                child.chunk_index,
                child.content,
                child.kind,
                now,
                now,
                model_name,
                metadata_json,
                child.parent_id,
            ),
        )
        conn.execute(
            "INSERT INTO memory_fts (chunk_id, content) VALUES (?, ?)",
            (chunk_id, child.content),
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
        child: ChildChunk,
        embedding: list[float],
        model_name: str,
        now: str,
    ) -> None:
        metadata_json = json.dumps(child.metadata, ensure_ascii=False)
        thread_id = child.metadata.get("thread_id")
        conn.execute(
            """
            UPDATE memory_chunks
            SET content = ?, kind = ?, thread_id = ?, updated_at = ?,
                model_name = ?, metadata_json = ?, parent_id = ?
            WHERE id = ?
            """,
            (
                child.content,
                child.kind,
                thread_id,
                now,
                model_name,
                metadata_json,
                child.parent_id,
                chunk_id,
            ),
        )
        conn.execute("DELETE FROM memory_fts WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            "INSERT INTO memory_fts (chunk_id, content) VALUES (?, ?)",
            (chunk_id, child.content),
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
