from __future__ import annotations

from pathlib import Path
from typing import Any

from miniclaw.config.settings import Settings
from miniclaw.observability.composite import CompositeTracer
from miniclaw.observability.contracts import NoopTracer, Tracer
from miniclaw.observability.langsmith import LangSmithTracer
from miniclaw.observability.local import JsonlTracer


def build_tracer(
    settings: Settings | None = None,
    *,
    langsmith_client: Any | None = None,
) -> Tracer:
    resolved_settings = settings or Settings()
    mode = resolved_settings.trace_mode
    if mode == "off":
        return NoopTracer()

    tracers: list[Tracer] = []
    if mode in {"local", "both"}:
        tracers.append(
            JsonlTracer(
                _trace_file_path(resolved_settings.trace_dir),
                full_content=resolved_settings.trace_full_content,
                max_chars=resolved_settings.trace_max_chars,
            )
        )
    if mode in {"langsmith", "both"}:
        tracers.append(
            LangSmithTracer(
                client=langsmith_client,
                project=_resolve_langsmith_project(),
                full_content=resolved_settings.trace_full_content,
                max_chars=resolved_settings.trace_max_chars,
            )
        )

    if len(tracers) == 1:
        return tracers[0]
    return CompositeTracer(tracers)


def _trace_file_path(trace_dir: Path) -> Path:
    return Path(trace_dir) / "miniclaw.jsonl"


def _resolve_langsmith_project() -> str:
    from os import environ

    return (
        environ.get("LANGSMITH_PROJECT", "").strip()
        or environ.get("LANGCHAIN_PROJECT", "").strip()
        or "miniclaw"
    )
