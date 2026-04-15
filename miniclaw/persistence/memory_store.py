from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import sqlite_vec

from miniclaw.utils.jsonx import safe_loads_dict


@dataclass(slots=True)
class MemoryItem:
    thread_id: str
    content: str
    kind: str
    metadata: dict[str, Any]
    created_at: str


@runtime_checkable
class MemoryStore(Protocol):
    def initialize(self) -> None: ...

    def append_fact(
        self,
        thread_id: str,
        content: str,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def list_recent(self, thread_id: str, limit: int) -> list[MemoryItem]: ...

    def list_recent_by_kind(self, thread_id: str, kind: str, limit: int) -> list[MemoryItem]: ...

    def search(self, thread_id: str, query: str, limit: int) -> list[MemoryItem]: ...

    def prune(self, thread_id: str) -> None: ...


class SQLiteMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_thread_id_id ON memory_items(thread_id, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_thread_id_content ON memory_items(thread_id, content)"
            )

            # -- Hybrid retrieval tables --
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_chunks (
                    id            TEXT PRIMARY KEY,
                    source_file   TEXT NOT NULL,
                    thread_id     TEXT,
                    chunk_index   INTEGER NOT NULL,
                    content       TEXT NOT NULL,
                    kind          TEXT NOT NULL,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL,
                    model_name    TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON memory_chunks(source_file)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_thread ON memory_chunks(thread_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_created ON memory_chunks(created_at)")

            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    chunk_id,
                    content,
                    tokenize='unicode61'
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_dirty_files (
                    file_path   TEXT PRIMARY KEY,
                    dirty_at    TEXT NOT NULL
                )
                """
            )

            # Load sqlite-vec extension and create vec0 table
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE memory_vec USING vec0(
                        chunk_id TEXT PRIMARY KEY,
                        embedding float[1024]
                    )
                    """
                )
            except sqlite3.OperationalError:
                # Table already exists — vec0 may not support IF NOT EXISTS
                pass

    def append_fact(
        self,
        thread_id: str,
        content: str,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO memory_items (thread_id, content, kind, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (thread_id, content, kind, json.dumps(metadata or {}, ensure_ascii=False)),
            )

    def list_recent(self, thread_id: str, limit: int) -> list[MemoryItem]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT thread_id, content, kind, metadata_json, created_at
                FROM memory_items
                WHERE thread_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def list_recent_by_kind(self, thread_id: str, kind: str, limit: int) -> list[MemoryItem]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT thread_id, content, kind, metadata_json, created_at
                FROM memory_items
                WHERE thread_id = ? AND kind = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (thread_id, kind, limit),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def search(self, thread_id: str, query: str, limit: int) -> list[MemoryItem]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT thread_id, content, kind, metadata_json, created_at
                FROM memory_items
                WHERE thread_id = ? AND content LIKE ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (thread_id, f"%{query}%", limit),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def prune(self, thread_id: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM memory_items WHERE thread_id = ?", (thread_id,))

    @staticmethod
    def _row_to_item(row: tuple[Any, ...]) -> MemoryItem:
        metadata = safe_loads_dict(row[3]) if row[3] else {}
        return MemoryItem(
            thread_id=str(row[0]),
            content=str(row[1]),
            kind=str(row[2]),
            metadata=metadata,
            created_at=str(row[4]),
        )
