import sqlite3
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from miniclaw.runtime.state import ActiveCapabilities

_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules={("miniclaw.runtime.state", "ActiveCapabilities")},
)


class AsyncSQLiteCheckpointer(BaseCheckpointSaver[str], AbstractAsyncContextManager["AsyncSQLiteCheckpointer"]):
    """Minimal SQLite-backed LangGraph checkpointer for the runtime MVP."""

    def __init__(self, path: Path) -> None:
        super().__init__(serde=_SERDE)
        self.path = Path(path)
        self._lock = Lock()
        self._initialize()

    async def __aenter__(self) -> "AsyncSQLiteCheckpointer":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool | None:
        return None

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    checkpoint_blob BLOB NOT NULL,
                    metadata_type TEXT NOT NULL,
                    metadata_blob BLOB NOT NULL,
                    parent_checkpoint_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    value_blob BLOB NOT NULL,
                    task_path TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, check_same_thread=False)

    @staticmethod
    def _thread_key(config: RunnableConfig) -> tuple[str, str]:
        configurable = config.get("configurable", {})
        return str(configurable["thread_id"]), str(configurable.get("checkpoint_ns", ""))

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id, checkpoint_ns = self._thread_key(config)
        checkpoint_id = get_checkpoint_id(config)
        query = """
            SELECT checkpoint_id, checkpoint_type, checkpoint_blob, metadata_type, metadata_blob, parent_checkpoint_id
            FROM langgraph_checkpoints
            WHERE thread_id = ? AND checkpoint_ns = ?
        """
        params: list[Any] = [thread_id, checkpoint_ns]
        if checkpoint_id is not None:
            query += " AND checkpoint_id = ?"
            params.append(checkpoint_id)
        query += " ORDER BY checkpoint_id DESC LIMIT 1"

        with self._lock, self._connect() as conn:
            row = conn.execute(query, params).fetchone()
            if row is None:
                return None

            resolved_checkpoint_id = str(row[0])
            writes = conn.execute(
                """
                SELECT task_id, channel, value_type, value_blob
                FROM langgraph_writes
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                ORDER BY idx ASC
                """,
                (thread_id, checkpoint_ns, resolved_checkpoint_id),
            ).fetchall()

        checkpoint = self.serde.loads_typed((str(row[1]), row[2]))
        metadata = self.serde.loads_typed((str(row[3]), row[4]))
        parent_checkpoint_id = row[5]

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": resolved_checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": str(parent_checkpoint_id),
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=[
                (str(task_id), str(channel), self.serde.loads_typed((str(value_type), value_blob)))
                for task_id, channel, value_type, value_blob in writes
            ],
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        del filter  # Not needed for the runtime MVP.
        thread_id = None if config is None else str(config.get("configurable", {}).get("thread_id"))
        checkpoint_ns = None if config is None else str(config.get("configurable", {}).get("checkpoint_ns", ""))
        before_id = None if before is None else get_checkpoint_id(before)
        query = """
            SELECT thread_id, checkpoint_ns, checkpoint_id
            FROM langgraph_checkpoints
        """
        clauses: list[str] = []
        params: list[Any] = []
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if checkpoint_ns is not None:
            clauses.append("checkpoint_ns = ?")
            params.append(checkpoint_ns)
        if before_id is not None:
            clauses.append("checkpoint_id < ?")
            params.append(before_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY checkpoint_id DESC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        for row_thread_id, row_checkpoint_ns, row_checkpoint_id in rows:
            item = self.get_tuple(
                {
                    "configurable": {
                        "thread_id": str(row_thread_id),
                        "checkpoint_ns": str(row_checkpoint_ns),
                        "checkpoint_id": str(row_checkpoint_id),
                    }
                }
            )
            if item is not None:
                yield item

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        del new_versions  # This saver persists full checkpoints.
        thread_id, checkpoint_ns = self._thread_key(config)
        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(checkpoint)
        metadata_type, metadata_blob = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        parent_checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO langgraph_checkpoints (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    checkpoint_type,
                    checkpoint_blob,
                    metadata_type,
                    metadata_blob,
                    parent_checkpoint_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_type,
                    checkpoint_blob,
                    metadata_type,
                    metadata_blob,
                    parent_checkpoint_id,
                ),
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, checkpoint_ns = self._thread_key(config)
        checkpoint_id = str(config.get("configurable", {}).get("checkpoint_id", ""))
        if not checkpoint_id:
            return

        rows = []
        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            value_type, value_blob = self.serde.dumps_typed(value)
            rows.append(
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    write_idx,
                    channel,
                    value_type,
                    value_blob,
                    task_path,
                )
            )

        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO langgraph_writes (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    idx,
                    channel,
                    value_type,
                    value_blob,
                    task_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def delete_thread(self, thread_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM langgraph_checkpoints WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM langgraph_writes WHERE thread_id = ?", (thread_id,))

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)
