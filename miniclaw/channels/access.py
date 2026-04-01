from __future__ import annotations

import random
import sqlite3
import string
from datetime import datetime
from pathlib import Path
from typing import Any

_CODE_CHARS = string.ascii_uppercase + string.digits
_CODE_LENGTH = 6
_CODE_EXPIRY_SECONDS = 600


class ChannelAccessStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_access (
                    channel_id    TEXT PRIMARY KEY,
                    channel_type  TEXT NOT NULL,
                    status        TEXT NOT NULL,
                    pair_code     TEXT,
                    authorized_at TEXT,
                    created_at    TEXT NOT NULL
                )
                """
            )

    def is_authorized(self, channel_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT status FROM channel_access WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        return row is not None and row[0] == "authorized"

    def create_or_get_pending(self, channel_id: str, channel_type: str) -> str:
        now = datetime.now().astimezone()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT pair_code, created_at, status FROM channel_access WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
            if row is not None:
                if row[2] == "authorized":
                    return ""
                code, created_at_str = row[0], row[1]
                if code and not self._is_expired(created_at_str, now):
                    return code
            code = self._generate_code()
            conn.execute(
                """
                INSERT OR REPLACE INTO channel_access
                    (channel_id, channel_type, status, pair_code, authorized_at, created_at)
                VALUES (?, ?, 'pending', ?, NULL, ?)
                """,
                (channel_id, channel_type, code, now.isoformat()),
            )
        return code

    def authorize(self, code: str) -> str | None:
        code = code.strip().upper()
        now = datetime.now().astimezone()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT channel_id, created_at FROM channel_access WHERE pair_code = ? AND status = 'pending'",
                (code,),
            ).fetchone()
            if row is None:
                return None
            channel_id, created_at_str = row
            if self._is_expired(created_at_str, now):
                return None
            conn.execute(
                "UPDATE channel_access SET status = 'authorized', pair_code = NULL, authorized_at = ? WHERE channel_id = ?",
                (now.isoformat(), channel_id),
            )
        return channel_id

    def revoke(self, channel_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM channel_access WHERE channel_id = ?", (channel_id,))

    def list_pending(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT channel_id, channel_type, pair_code, created_at, status FROM channel_access WHERE status = 'pending'"
            ).fetchall()
        return [
            {"channel_id": r[0], "channel_type": r[1], "pair_code": r[2], "created_at": r[3], "status": r[4]}
            for r in rows
        ]

    @staticmethod
    def _generate_code() -> str:
        return "".join(random.choices(_CODE_CHARS, k=_CODE_LENGTH))

    @staticmethod
    def _is_expired(created_at_str: str, now: datetime) -> bool:
        try:
            created = datetime.fromisoformat(created_at_str)
            return (now - created).total_seconds() > _CODE_EXPIRY_SECONDS
        except (ValueError, TypeError):
            return True
