"""Async-to-sync bridge utility.

Used by modules that need to call async code from synchronous graph nodes,
tool executors, or other sync contexts.
"""
from __future__ import annotations

import asyncio
from queue import Queue
from threading import Thread


def run_sync(awaitable):
    """Run an awaitable synchronously.

    If no event loop is running, uses asyncio.run() directly.
    Otherwise, spawns a daemon thread with its own event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    queue: Queue[tuple[bool, object]] = Queue(maxsize=1)

    def _runner() -> None:
        try:
            queue.put((True, asyncio.run(awaitable)))
        except Exception as exc:
            queue.put((False, exc))

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    success, payload = queue.get()
    thread.join()
    if success:
        return payload
    raise payload  # type: ignore[misc]
