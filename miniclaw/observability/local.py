from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from miniclaw.observability.contracts import (
    TraceContext,
    TraceRecord,
    build_run_context,
    build_span_context,
)


class JsonlTracer:
    def __init__(self, path: Path, *, full_content: bool, max_chars: int) -> None:
        self.path = Path(path)
        self.full_content = full_content
        self.max_chars = max(0, max_chars)

    def start_run(
        self,
        *,
        name: str,
        thread_id: str | None = None,
        channel: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        context: TraceContext | None = None,
    ) -> TraceContext:
        context = context or build_run_context(
            name=name,
            thread_id=thread_id,
            channel=channel,
            metadata=metadata,
        )
        self._write_record(
            TraceRecord(
                kind="run_start",
                trace_id=context.trace_id,
                run_id=context.run_id,
                thread_id=context.thread_id,
                channel=context.channel,
                name=name,
                metadata=dict(metadata or {}),
            )
        )
        return context

    def finish_run(
        self,
        context: TraceContext,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._write_record(
            TraceRecord(
                kind="run_finish",
                trace_id=context.trace_id,
                run_id=context.run_id,
                thread_id=context.thread_id,
                channel=context.channel,
                name=context.name or "",
                status=status,
                metadata=dict(metadata or {}),
                output=dict(output or {}),
            )
        )

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
        context = context or build_span_context(parent, name=name, metadata=metadata)
        record_metadata = dict(metadata or {})
        if run_type is not None:
            record_metadata["run_type"] = run_type
        self._write_record(
            TraceRecord(
                kind="span_start",
                trace_id=context.trace_id,
                run_id=context.run_id,
                span_id=context.span_id,
                parent_span_id=context.parent_span_id,
                thread_id=context.thread_id,
                channel=context.channel,
                name=name,
                metadata=record_metadata,
                payload=dict(inputs or {}),
            )
        )
        return context

    def finish_span(
        self,
        context: TraceContext,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_output = dict(outputs) if outputs is not None else dict(output or {})
        self._write_record(
            TraceRecord(
                kind="span_finish",
                trace_id=context.trace_id,
                run_id=context.run_id,
                span_id=context.span_id,
                parent_span_id=context.parent_span_id,
                thread_id=context.thread_id,
                channel=context.channel,
                name=context.name or "",
                status=status,
                metadata=dict(metadata or {}),
                output=resolved_output,
            )
        )

    def record_event(
        self,
        context: TraceContext,
        *,
        name: str,
        payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        status: str | None = None,
    ) -> None:
        self._write_record(
            TraceRecord(
                kind="event",
                trace_id=context.trace_id,
                run_id=context.run_id,
                span_id=context.span_id,
                parent_span_id=context.parent_span_id,
                thread_id=context.thread_id,
                channel=context.channel,
                name=name,
                status=status,
                metadata=dict(metadata or {}),
                payload=dict(payload or {}),
            )
        )

    def _write_record(self, record: TraceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._truncate_record(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def _truncate_record(self, record: TraceRecord) -> dict[str, Any]:
        data = record.as_dict()
        for key in ("metadata", "payload", "output"):
            data[key] = self._truncate_value(data[key])
        return data

    def _truncate_value(self, value: Any) -> Any:
        if isinstance(value, str):
            if len(value) <= self.max_chars:
                return value
            preview = value[: self.max_chars]
            if self.full_content:
                return f"{preview}...<truncated {len(value) - self.max_chars} chars>"
            return {
                "length": len(value),
                "preview": preview,
                "truncated": len(value) - self.max_chars,
            }
        if isinstance(value, Mapping):
            return {str(key): self._truncate_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._truncate_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._truncate_value(item) for item in value]
        return value
