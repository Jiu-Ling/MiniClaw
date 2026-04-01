from __future__ import annotations

from pathlib import Path

from miniclaw.persistence.memory_store import MemoryStore, SQLiteMemoryStore


def build_memory_store(sqlite_path: Path, backend: str = "sqlite") -> MemoryStore:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "sqlite":
        store = SQLiteMemoryStore(sqlite_path)
        store.initialize()
        return store
    if normalized_backend == "postgres":
        raise NotImplementedError("Postgres memory store is not implemented yet.")
    raise ValueError(f"Unsupported memory backend: {backend}")
