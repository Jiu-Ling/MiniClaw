from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from miniclaw.observability.contracts import TraceContext, build_run_context, build_span_context


def _infer_run_type(name: str, explicit: str | None) -> str:
    """Derive a LangSmith run_type from span name when no explicit type is given.

    Mapping:
      tool.*  / tool.call.*   → "tool"
      provider.*              → "llm"
      memory.retrieve / memory.search → "retriever"
      everything else         → "chain"
    """
    if explicit is not None:
        return explicit
    lower = name.lower()
    if lower.startswith("tool."):
        return "tool"
    if lower.startswith("provider."):
        return "llm"
    if lower in {"memory.retrieve", "memory.search"} or lower.startswith("memory.retrieve") or lower.startswith("memory.search"):
        return "retriever"
    return "chain"


class LangSmithTracer:
    def __init__(
        self,
        client: Any | None = None,
        *,
        project: str | None = None,
        full_content: bool = False,
        max_chars: int = 4000,
    ) -> None:
        _ensure_tracing_env()
        self.client = client or _build_default_client()
        self.project = project or _default_project_name()
        self.full_content = full_content
        self.max_chars = max(0, max_chars)
        self._start_times: dict[str, datetime] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._start_metadata: dict[str, dict[str, Any]] = {}

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
        self._start_times[context.run_id] = _utc_now()
        self._create_run(
            run_id=context.run_id,
            trace_id=context.trace_id,
            parent_run_id=None,
            name=name,
            run_type="chain",
            inputs=self._truncate_mapping(
                {
                    "thread_id": thread_id,
                    "channel": channel,
                    "metadata": dict(metadata or {}),
                }
            ),
            extra=self._base_extra(context, metadata=metadata, payload=None),
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
        run_id = context.run_id
        events = self._events.pop(run_id, [])
        extra = self._base_extra(context, metadata=metadata, payload=output, status=status)
        if events:
            extra["events"] = events
        self._update_run(
            run_id=run_id,
            name=context.name,
            run_type="chain",
            start_time=self._start_times.pop(run_id, None),
            end_time=_utc_now(),
            error=_status_to_error(status),
            outputs=self._truncate_mapping(dict(output or {})),
            extra=extra,
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
        run_id = context.span_id or context.run_id
        self._start_times[run_id] = _utc_now()
        self._start_metadata[run_id] = dict(metadata or {})
        resolved_run_type = _infer_run_type(name, run_type)
        resolved_inputs = (
            dict(inputs)
            if inputs is not None
            else self._truncate_mapping(
                {
                    "parent_span_id": parent.span_id,
                    "parent_run_id": parent.run_id,
                    "metadata": dict(metadata or {}),
                }
            )
        )
        self._create_run(
            run_id=run_id,
            trace_id=context.trace_id,
            parent_run_id=parent.span_id or parent.run_id,
            name=name,
            run_type=resolved_run_type,
            inputs=resolved_inputs,
            extra=self._base_extra(context, metadata=metadata, payload=None),
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
        run_id = context.span_id or context.run_id
        events = self._events.pop(run_id, [])
        start_metadata = self._start_metadata.pop(run_id, {})
        resolved_outputs = (
            dict(outputs)
            if outputs is not None
            else self._truncate_mapping(dict(output or {}))
        )
        span_name = context.name or ""
        run_type = _infer_run_type(span_name, None)
        extra = self._base_extra(context, metadata=metadata, payload=output, status=status)
        if events:
            extra["events"] = events
        if run_type == "llm":
            extra = _enrich_llm_extra(extra, start_metadata, events)
        self._update_run(
            run_id=run_id,
            name=span_name,
            run_type=run_type,
            start_time=self._start_times.pop(run_id, None),
            end_time=_utc_now(),
            error=_status_to_error(status),
            outputs=resolved_outputs,
            extra=extra,
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
        run_id = context.span_id or context.run_id
        event = {
            "name": name,
            "status": status,
            "timestamp": _utc_now().isoformat(),
            "trace_id": context.trace_id,
            "run_id": context.run_id,
            "span_id": context.span_id,
            "parent_span_id": context.parent_span_id,
            "thread_id": context.thread_id,
            "channel": context.channel,
            "metadata": self._truncate_mapping(dict(metadata or {})),
            "payload": self._truncate_mapping(dict(payload or {})),
        }
        bucket = self._events.setdefault(run_id, [])
        bucket.append(event)

    def _truncate_mapping(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return {str(key): self._truncate_value(item) for key, item in value.items()}

    def _truncate_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return _truncate_string(value, max_chars=self.max_chars, full_content=self.full_content)
        if isinstance(value, Mapping):
            return self._truncate_mapping(value)
        if isinstance(value, list):
            return [self._truncate_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._truncate_value(item) for item in value]
        return value

    def _base_extra(
        self,
        context: TraceContext,
        *,
        metadata: Mapping[str, Any] | None,
        payload: Mapping[str, Any] | None,
        status: str | None = None,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "trace_id": context.trace_id,
            "run_id": context.run_id,
            "span_id": context.span_id,
            "parent_span_id": context.parent_span_id,
            "thread_id": context.thread_id,
            "channel": context.channel,
            "metadata": self._truncate_mapping(dict(metadata or {})),
        }
        if payload is not None:
            extra["payload"] = self._truncate_mapping(dict(payload))
        if status is not None:
            extra["status"] = status
        if context.name is not None:
            extra["context_name"] = context.name
        return extra

    def _create_run(
        self,
        *,
        run_id: str,
        trace_id: str,
        parent_run_id: str | None,
        name: str,
        run_type: str,
        inputs: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            self.client.create_run(
                id=_to_uuid(run_id),
                trace_id=_to_uuid(trace_id),
                parent_run_id=_to_uuid(parent_run_id) if parent_run_id else None,
                project_name=self.project,
                name=name,
                run_type=run_type,
                inputs=dict(inputs),
                extra=dict(extra or {}),
            )
        except Exception:
            return None

    def _update_run(
        self,
        *,
        run_id: str,
        name: str | None = None,
        run_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        error: str | None = None,
        outputs: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            self.client.update_run(
                _to_uuid(run_id),
                name=name,
                run_type=run_type,
                start_time=start_time,
                end_time=end_time,
                error=error,
                outputs=dict(outputs or {}),
                extra=dict(extra or {}),
            )
        except Exception:
            return None


def _enrich_llm_extra(
    extra: dict[str, Any],
    metadata: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Lift model name and token usage into LangSmith-recognised extra keys."""
    model = metadata.get("model") or metadata.get("provider.model")
    if model:
        invocation = dict(extra.get("invocation_params") or {})
        invocation["model"] = model
        extra = {**extra, "invocation_params": invocation}
    for ev in events:
        if ev.get("name") == "prompt.cache.usage":
            extra = {**extra, "token_usage": ev.get("payload") or {}}
            break
    return extra


def _ensure_tracing_env() -> None:
    """Load .env and ensure LANGSMITH_TRACING=true so the SDK sends data.

    pydantic-settings reads .env into the Settings model but does NOT export
    values to ``os.environ``.  The LangSmith ``Client`` reads API key,
    endpoint, and project directly from ``os.environ``, so we must load .env
    explicitly here.
    """
    import os

    from miniclaw.config.settings import _repo_root

    dotenv_path = _repo_root() / ".env"
    if dotenv_path.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path, override=False)
        except ImportError:
            pass

    if not os.environ.get("LANGSMITH_TRACING"):
        os.environ["LANGSMITH_TRACING"] = "true"


def _build_default_client() -> Any:
    from langsmith import Client

    return Client()


def _default_project_name() -> str:
    from os import environ

    return (
        environ.get("LANGSMITH_PROJECT", "").strip()
        or environ.get("LANGCHAIN_PROJECT", "").strip()
        or "miniclaw"
    )


def _to_uuid(value: str) -> str:
    """Convert a prefixed ID (e.g. 'run_abc123') to a proper UUID string.

    LangSmith requires valid UUID-format IDs.  Our internal IDs use
    ``prefix_<hex>`` where ``<hex>`` is ``uuid4().hex`` (32 hex chars).
    This strips the prefix and formats as a standard UUID.
    """
    import re
    import uuid

    hex_part = re.sub(r"^[a-z]+_", "", value)
    try:
        return str(uuid.UUID(hex_part))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _status_to_error(status: str) -> str | None:
    normalized = status.strip().lower()
    if normalized in {"ok", "success", "completed", "done"}:
        return None
    return status


def _truncate_string(value: str, *, max_chars: int, full_content: bool) -> Any:
    limit = max(0, max_chars)
    if len(value) <= limit:
        return value
    preview = value[:limit]
    if full_content:
        return f"{preview}...<truncated {len(value) - limit} chars>"
    return {
        "length": len(value),
        "preview": preview,
        "truncated": len(value) - limit,
    }
