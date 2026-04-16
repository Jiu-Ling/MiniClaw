"""Exception-swallowing wrappers for tracer calls.

A broken tracer must never crash the main agent path, subagent loop,
or background worker. All functions in this module catch Exception and
either return a fallback value or silently discard the failure.

Replaces the 5+ duplicate _safe_start_span / _safe_finish_span /
_safe_record_event / _safe_tracer_call definitions that were scattered
across runtime/nodes.py, runtime/service.py, and runtime/subagent.py.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def safe_start_span(
    tracer: Any,
    parent: Any,
    *,
    name: str,
    metadata: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    run_type: str | None = None,
) -> Any:
    """Start a trace span. Returns None if tracer/parent is None or on error."""
    if tracer is None or parent is None:
        return None
    try:
        return tracer.start_span(
            parent, name=name, metadata=metadata, inputs=inputs, run_type=run_type,
        )
    except Exception:
        return None


def safe_finish_span(
    tracer: Any,
    span: Any,
    *,
    status: str,
    metadata: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
) -> None:
    """Finish a trace span. No-op if tracer/span is None or on error."""
    if tracer is None or span is None:
        return
    try:
        tracer.finish_span(span, status=status, metadata=metadata, outputs=outputs)
    except Exception:
        pass


def safe_record_event(
    tracer: Any,
    context: Any,
    *,
    name: str,
    payload: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    status: str | None = None,
) -> None:
    """Record a trace event. No-op if tracer/context is None or on error."""
    if tracer is None or context is None:
        return
    try:
        tracer.record_event(
            context, name=name, payload=payload, metadata=metadata, status=status,
        )
    except Exception:
        pass


def safe_start_run(
    tracer: Any,
    *,
    name: str,
    thread_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Start a trace run. Returns None if tracer is None or on error."""
    if tracer is None:
        return None
    try:
        return tracer.start_run(name=name, thread_id=thread_id, metadata=metadata)
    except Exception:
        return None


def safe_finish_run(
    tracer: Any,
    context: Any,
    *,
    status: str,
    output: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Finish a trace run. No-op if tracer/context is None or on error."""
    if tracer is None or context is None:
        return
    try:
        tracer.finish_run(context, status=status, output=output, metadata=metadata)
    except Exception:
        pass
