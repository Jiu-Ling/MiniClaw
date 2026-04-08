from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

TraceKind = Literal["run_start", "run_finish", "span_start", "span_finish", "event"]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_trace_id() -> str:
    return f"trace_{uuid4().hex}"


def new_run_id() -> str:
    return f"run_{uuid4().hex}"


def new_span_id() -> str:
    return f"span_{uuid4().hex}"


@dataclass(slots=True)
class TraceContext:
    trace_id: str
    run_id: str
    span_id: str | None = None
    parent_span_id: str | None = None
    thread_id: str | None = None
    channel: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceRecord:
    kind: TraceKind
    timestamp: str = field(default_factory=_utc_timestamp)
    trace_id: str = ""
    run_id: str = ""
    span_id: str | None = None
    parent_span_id: str | None = None
    thread_id: str | None = None
    channel: str | None = None
    name: str = ""
    status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "kind": self.kind,
            "metadata": self.metadata,
            "name": self.name,
            "output": self.output,
            "payload": self.payload,
            "parent_span_id": self.parent_span_id,
            "run_id": self.run_id,
            "span_id": self.span_id,
            "status": self.status,
            "timestamp": self.timestamp,
            "thread_id": self.thread_id,
            "trace_id": self.trace_id,
        }


@runtime_checkable
class Tracer(Protocol):
    def start_run(
        self,
        *,
        name: str,
        thread_id: str | None = None,
        channel: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        context: TraceContext | None = None,
    ) -> TraceContext: ...

    def finish_run(
        self,
        context: TraceContext,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def start_span(
        self,
        parent: TraceContext,
        *,
        name: str,
        metadata: Mapping[str, Any] | None = None,
        context: TraceContext | None = None,
        inputs: Mapping[str, Any] | None = None,
        run_type: str | None = None,
    ) -> TraceContext: ...

    def finish_span(
        self,
        context: TraceContext,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
    ) -> None: ...

    def record_event(
        self,
        context: TraceContext,
        *,
        name: str,
        payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        status: str | None = None,
    ) -> None: ...


class NoopTracer:
    def start_run(
        self,
        *,
        name: str,
        thread_id: str | None = None,
        channel: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        context: TraceContext | None = None,
    ) -> TraceContext:
        return context or TraceContext(
            trace_id=new_trace_id(),
            run_id=new_run_id(),
            thread_id=thread_id,
            channel=channel,
            name=name,
            metadata=dict(metadata or {}),
        )

    def finish_run(
        self,
        context: TraceContext,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        return None

    def start_span(
        self,
        parent: TraceContext,
        *,
        name: str,
        metadata: Mapping[str, Any] | None = None,
        context: TraceContext | None = None,
        inputs: Mapping[str, Any] | None = None,
        run_type: str | None = None,
    ) -> TraceContext:
        return context or TraceContext(
            trace_id=parent.trace_id,
            run_id=parent.run_id,
            span_id=new_span_id(),
            parent_span_id=parent.span_id,
            thread_id=parent.thread_id,
            channel=parent.channel,
            name=name,
            metadata=dict(metadata or {}),
        )

    def finish_span(
        self,
        context: TraceContext,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
    ) -> None:
        return None

    def record_event(
        self,
        context: TraceContext,
        *,
        name: str,
        payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        status: str | None = None,
    ) -> None:
        return None


def build_run_context(
    *,
    name: str,
    thread_id: str | None = None,
    channel: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TraceContext:
    return TraceContext(
        trace_id=new_trace_id(),
        run_id=new_run_id(),
        thread_id=thread_id,
        channel=channel,
        name=name,
        metadata=dict(metadata or {}),
    )


def build_span_context(
    parent: TraceContext,
    *,
    name: str,
    metadata: Mapping[str, Any] | None = None,
) -> TraceContext:
    return TraceContext(
        trace_id=parent.trace_id,
        run_id=parent.run_id,
        span_id=new_span_id(),
        parent_span_id=parent.span_id,
        thread_id=parent.thread_id,
        channel=parent.channel,
        name=name,
        metadata=dict(metadata or {}),
    )
