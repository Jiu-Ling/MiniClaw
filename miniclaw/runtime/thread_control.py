from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ThreadControlState:
    thread_id: str
    stopped: bool = False


class SQLiteThreadControlStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._initialize()

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_controls (
                    thread_id TEXT PRIMARY KEY,
                    stopped INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get(self, thread_id: str) -> ThreadControlState:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT stopped FROM thread_controls WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if row is None:
            return ThreadControlState(thread_id=thread_id)
        return ThreadControlState(thread_id=thread_id, stopped=bool(row[0]))

    def set_stopped(self, thread_id: str, stopped: bool) -> ThreadControlState:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO thread_controls (thread_id, stopped, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id) DO UPDATE SET
                    stopped = excluded.stopped,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, int(stopped)),
            )
        return ThreadControlState(thread_id=thread_id, stopped=stopped)

    def clear(self, thread_id: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM thread_controls WHERE thread_id = ?", (thread_id,))
