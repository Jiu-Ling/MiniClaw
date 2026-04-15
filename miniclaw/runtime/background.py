"""Single-queue, single-worker background scheduler for fire-and-forget jobs.

Primary consumer is memory consolidation (Phase 4). Scheduler is deliberately
constrained to one worker to serialize writes to MEMORY.md without ad-hoc
locking. Do NOT use for periodic async services (heartbeat, cron) or for
short-lived parallel fan-out (tool_loop).
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from miniclaw.observability.contracts import TraceContext

logger = logging.getLogger(__name__)

__all__ = [
    "BackgroundJob",
    "BackgroundScheduler",
    "InlineBackgroundScheduler",
]


@dataclass(frozen=True)
class BackgroundJob:
    """A unit of work submitted to the background scheduler.

    fn:           callable; must be idempotent-safe and self-contained.
    kind:         logical category, used for trace span naming.
    job_id:       stable id for trace correlation; auto-assigned if empty.
    metadata:     opaque dict surfaced as trace span metadata.
    on_failure:   callback invoked with the exception if fn raises.
    parent_trace: trace context captured at submit time; if set, the job's
                  execution span hangs under it. None -> no span emitted.
    """

    fn: Callable[[], Any]
    kind: str
    job_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    on_failure: Callable[[Exception], None] | None = None
    parent_trace: TraceContext | None = None


class BackgroundScheduler:
    """Single daemon worker consuming BackgroundJobs from a bounded queue.

    Lifetime: start() -> submit() ... -> stop(). Both lifecycle methods
    are idempotent. Exactly one worker thread; jobs run in FIFO order.
    """

    def __init__(
        self,
        *,
        name: str = "miniclaw-bg",
        max_queue: int = 32,
        tracer: Any | None = None,
    ) -> None:
        self.name = name
        self.max_queue = max_queue
        self.tracer = tracer
        self._queue: queue.Queue[BackgroundJob | None] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._run, name=self.name, daemon=True
        )
        self._thread.start()
        self._started = True

    def stop(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        if not self._started:
            return
        self._stopping.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if wait and self._thread is not None:
            self._thread.join(timeout=timeout)
        self._started = False
        self._thread = None

    def is_running(self) -> bool:
        return self._started

    def submit(self, job: BackgroundJob) -> bool:
        if not self._started:
            if self._stopping.is_set():
                return False
            raise RuntimeError("BackgroundScheduler.submit called before start() (not started)")
        if self._stopping.is_set():
            return False
        job_to_submit = job if job.job_id else _with_id(job)
        try:
            self._queue.put_nowait(job_to_submit)
            return True
        except queue.Full:
            logger.warning(
                "background queue full; dropping job kind=%s id=%s",
                job.kind, job_to_submit.job_id,
            )
            return False

    def queue_size(self) -> int:
        return self._queue.qsize()

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if job is None:
                return
            self._execute(job)

    def _execute(self, job: BackgroundJob) -> None:
        span = None
        if self.tracer is not None and job.parent_trace is not None:
            try:
                span = self.tracer.start_span(
                    job.parent_trace,
                    name=f"bg.{job.kind}",
                    metadata={
                        "job_id": job.job_id,
                        "queue_size_at_execute": self._queue.qsize(),
                        **job.metadata,
                    },
                )
            except Exception:
                span = None
        try:
            job.fn()
        except Exception as exc:
            logger.exception(
                "background job failed kind=%s id=%s", job.kind, job.job_id
            )
            if span is not None and self.tracer is not None:
                try:
                    self.tracer.finish_span(
                        span, status="error", metadata={"error": str(exc)}
                    )
                except Exception:
                    pass
            if job.on_failure is not None:
                try:
                    job.on_failure(exc)
                except Exception:
                    logger.exception(
                        "on_failure handler itself failed for kind=%s", job.kind
                    )
            return
        if span is not None and self.tracer is not None:
            try:
                self.tracer.finish_span(span, status="ok")
            except Exception:
                pass


def _with_id(job: BackgroundJob) -> BackgroundJob:
    return BackgroundJob(
        fn=job.fn,
        kind=job.kind,
        job_id=f"{job.kind}-{uuid4().hex[:8]}",
        metadata=dict(job.metadata),
        on_failure=job.on_failure,
        parent_trace=job.parent_trace,
    )


class InlineBackgroundScheduler:
    """Test fake: executes jobs synchronously on submit.

    Used in unit tests to avoid real threading. API-compatible with
    BackgroundScheduler for drop-in substitution via DI.
    """

    def __init__(self) -> None:
        self.submitted: list[BackgroundJob] = []

    def start(self) -> None:
        pass

    def stop(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        pass

    def is_running(self) -> bool:
        return True

    def submit(self, job: BackgroundJob) -> bool:
        self.submitted.append(job)
        try:
            job.fn()
        except Exception as exc:
            if job.on_failure is not None:
                try:
                    job.on_failure(exc)
                except Exception:
                    logger.exception("InlineBackgroundScheduler on_failure raised")
        return True

    def queue_size(self) -> int:
        return 0
