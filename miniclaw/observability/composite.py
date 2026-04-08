from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from miniclaw.observability.contracts import NoopTracer, TraceContext, Tracer
from miniclaw.observability.contracts import build_run_context, build_span_context


class CompositeTracer:
    def __init__(self, tracers: Sequence[Tracer]) -> None:
        self._tracers = tuple(tracers)

    def start_run(
        self,
        *,
        name: str,
        thread_id: str | None = None,
        channel: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        context: TraceContext | None = None,
    ) -> TraceContext:
        shared_context = context or build_run_context(
            name=name,
            thread_id=thread_id,
            channel=channel,
            metadata=metadata,
        )
        for tracer in self._tracers:
            tracer.start_run(
                name=name,
                thread_id=thread_id,
                channel=channel,
                metadata=metadata,
                context=shared_context,
            )
        if not self._tracers:
            return NoopTracer().start_run(
                name=name,
                thread_id=thread_id,
                channel=channel,
                metadata=metadata,
            )
        return shared_context

    def finish_run(
        self,
        context: TraceContext,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        for tracer in self._tracers:
            tracer.finish_run(context, status=status, output=output, metadata=metadata)

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
        shared_context = context or build_span_context(parent, name=name, metadata=metadata)
        for tracer in self._tracers:
            tracer.start_span(
                parent, name=name, metadata=metadata, context=shared_context,
                inputs=inputs, run_type=run_type,
            )
        if not self._tracers:
            return NoopTracer().start_span(parent, name=name, metadata=metadata)
        return shared_context

    def finish_span(
        self,
        context: TraceContext,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
    ) -> None:
        for tracer in self._tracers:
            tracer.finish_span(context, status=status, output=output, metadata=metadata, outputs=outputs)

    def record_event(
        self,
        context: TraceContext,
        *,
        name: str,
        payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        status: str | None = None,
    ) -> None:
        for tracer in self._tracers:
            tracer.record_event(
                context,
                name=name,
                payload=payload,
                metadata=metadata,
                status=status,
            )
