"""Pure rendering helpers for `miniclaw trace tail`.

No file I/O, no typer — only stdlib so these functions are easily unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SpanState:
    """Mutable-free snapshot of an open span."""

    start_ts: str
    kind: str  # "run_start" | "span_start"
    name: str
    parent_span_id: str | None
    metadata: dict[str, Any]
    depth: int


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------


def parse_iso_ts(s: str) -> datetime:
    """Parse an ISO-8601 timestamp string (UTC, with or without trailing Z)."""
    s = s.rstrip("Z")
    # Handle fractional seconds of varying length
    if "." in s:
        date_part, frac = s.rsplit(".", 1)
        frac = frac[:6].ljust(6, "0")
        s = f"{date_part}.{frac}"
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
    else:
        fmt = "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)


def format_duration(ms: float) -> str:
    """Format a duration in milliseconds to a human-readable string."""
    if ms < 1000:
        return f"{int(ms)}ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 60_000:.1f}m"


def _compute_duration_ms(start_ts: str, finish_ts: str) -> float | None:
    """Return the duration in milliseconds between two ISO timestamps, or None on error."""
    try:
        start = parse_iso_ts(start_ts)
        finish = parse_iso_ts(finish_ts)
        return max(0.0, (finish - start).total_seconds() * 1000)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Inline detail extractors
# ---------------------------------------------------------------------------


_MAX_ARG_LEN = 40


def _truncate(value: str, max_len: int = _MAX_ARG_LEN) -> str:
    if len(value) <= max_len:
        return value
    return value[:max_len] + "…"


def extract_tool_inline(metadata: dict[str, Any], payload: dict[str, Any]) -> str:
    """Build an inline detail string for tool spans.

    Returns something like: ``cmd="ls /tmp" (builtin)``
    """
    parts: list[str] = []

    # First argument key=value pair from payload.arguments
    arguments: dict[str, Any] = payload.get("arguments") or {}
    if arguments:
        first_key = next(iter(arguments))
        raw_val = str(arguments[first_key])
        parts.append(f'{first_key}="{_truncate(raw_val)}"')

    # Source tag
    source: str = metadata.get("tool_source") or metadata.get("tool.source") or ""
    if source:
        parts.append(f"({source})")

    return "  ".join(parts)


def extract_chat_inline(metadata: dict[str, Any]) -> str:
    """Build an inline detail string for provider.chat spans.

    Returns something like: ``model=claude-sonnet-4-6``
    """
    model: str = metadata.get("model") or ""
    if model:
        return f"model={model}"
    return ""


# ---------------------------------------------------------------------------
# Tree-connector helpers
# ---------------------------------------------------------------------------

_MARKERS = {
    "run_start": "▶",
    "run_finish": "■",
    "span_start": "├",
    "span_finish": "└",
    "event": "•",
}


def _indent(depth: int) -> str:
    return "  " * depth


# ---------------------------------------------------------------------------
# Timestamp prefix
# ---------------------------------------------------------------------------


def _ts_prefix(timestamp: str) -> str:
    """Return HH:MM:SS from an ISO timestamp, or empty string on failure."""
    try:
        dt = parse_iso_ts(timestamp)
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Color helpers (applied only when color=True)
# ---------------------------------------------------------------------------


def _colorize(text: str, *, color: bool, fg: str) -> str:
    """Wrap *text* in ANSI escape sequences when *color* is True."""
    if not color:
        return text
    codes = {
        "green": "\033[32m",
        "red": "\033[31m",
        "yellow": "\033[33m",
        "cyan": "\033[36m",
        "reset": "\033[0m",
    }
    return f"{codes.get(fg, '')}{text}{codes['reset']}"


def _status_color(status: str, *, color: bool) -> str:
    if not color:
        return status
    if status == "ok":
        return _colorize(status, color=True, fg="green")
    if status in ("error", "failed"):
        return _colorize(status, color=True, fg="red")
    return status


# ---------------------------------------------------------------------------
# Main format_record entry point
# ---------------------------------------------------------------------------


def format_record(
    rec: dict[str, Any],
    span_state: dict[str, SpanState],
    *,
    color: bool,
    min_ms: int,
    tool_filter: str | None = None,
    kind_filter: set[str] | None = None,
    status_filter: str | None = None,
) -> str | None:
    """Render a single trace record to a printable line, or None if filtered out.

    Side-effects on *span_state*:
    - ``run_start`` / ``span_start``: inserts a new ``SpanState``.
    - ``span_finish`` / ``run_finish``: pops the matching ``SpanState``.

    The function returns ``None`` (suppressed) when any active filter rejects the
    record, but the *span_state* dict is still updated so tree depth stays
    consistent.
    """
    kind: str = rec.get("kind") or ""
    name: str = rec.get("name") or ""
    span_id: str = rec.get("span_id") or ""
    parent_span_id: str | None = rec.get("parent_span_id")
    timestamp: str = rec.get("timestamp") or ""
    status: str = rec.get("status") or ""
    metadata: dict[str, Any] = rec.get("metadata") or {}
    payload: dict[str, Any] = rec.get("payload") or {}

    # ------------------------------------------------------------------ depth
    depth = _compute_depth(kind, span_id, parent_span_id, span_state)

    # ------------------------------------------- register open spans in state
    if kind in ("run_start", "span_start"):
        span_state[span_id] = SpanState(
            start_ts=timestamp,
            kind=kind,
            name=name,
            parent_span_id=parent_span_id,
            metadata=metadata,
            depth=depth,
        )

    # ------------------------------------------------------ compute duration
    duration_ms: float | None = None
    open_span: SpanState | None = None
    if kind in ("span_finish", "run_finish"):
        open_span = span_state.pop(span_id, None)
        if open_span is not None:
            duration_ms = _compute_duration_ms(open_span.start_ts, timestamp)
            # Use registered depth from open span for consistent indentation
            depth = open_span.depth

    # ---------------------------------------------------- event parent depth
    if kind == "event" and parent_span_id and parent_span_id in span_state:
        depth = span_state[parent_span_id].depth + 1

    # ---------------------------------------------------------------- filters
    if kind_filter and kind not in kind_filter:
        return None

    if status_filter and status and status != status_filter:
        return None

    if tool_filter:
        target = f"tool.{tool_filter}"
        if name != target and not name.startswith(f"tool.call.{tool_filter}"):
            return None

    if min_ms > 0 and duration_ms is not None and duration_ms < min_ms:
        return None

    # ----------------------------------------------------------- build output
    ts_pfx = _ts_prefix(timestamp)
    marker = _MARKERS.get(kind, "·")
    indent = _indent(depth)

    if kind == "event":
        return _render_event(name, payload, metadata, ts_pfx, indent, marker, color=color)

    inline = _build_inline(name, metadata, payload, open_span=open_span)

    # Status + duration suffix
    suffix_parts: list[str] = []
    if status:
        suffix_parts.append(_status_color(status, color=color))
    if duration_ms is not None:
        suffix_parts.append(format_duration(duration_ms))
    suffix = f"[{', '.join(suffix_parts)}]" if suffix_parts else ""

    # Colorize name for tool spans
    display_name = name
    if name.startswith("tool.") and color:
        display_name = _colorize(name, color=True, fg="yellow")

    # Align columns: name+inline left-padded to 50 chars, then suffix
    core = f"{display_name}{('  ' + inline) if inline else ''}"
    line = f"{ts_pfx}  {indent}{marker} {core}"
    if suffix:
        # Right-pad to column 72 before appending suffix
        padded = line.ljust(72)
        line = f"{padded} {suffix}"

    return line


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_depth(
    kind: str,
    span_id: str,
    parent_span_id: str | None,
    span_state: dict[str, SpanState],
) -> int:
    if parent_span_id and parent_span_id in span_state:
        return span_state[parent_span_id].depth + 1
    if kind == "run_start":
        return 0
    return span_state[span_id].depth if span_id in span_state else 0


def _build_inline(
    name: str,
    metadata: dict[str, Any],
    payload: dict[str, Any],
    *,
    open_span: SpanState | None,
) -> str:
    """Return the inline detail string for a span (tool args, model name, etc.)."""
    # For finish records, prefer the metadata from the open span (richer on start)
    effective_meta = (open_span.metadata if open_span else None) or metadata

    if name.startswith("tool."):
        return extract_tool_inline(effective_meta, payload)
    if name.startswith("provider."):
        return extract_chat_inline(effective_meta)
    return ""


def _render_event(
    name: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    ts_pfx: str,
    indent: str,
    marker: str,
    *,
    color: bool,
) -> str:
    colored_marker = _colorize(marker, color=color, fg="cyan")
    if name == "prompt.cache.usage":
        cached = int(payload.get("cached_tokens") or 0)
        prompt = int(payload.get("prompt_tokens") or 1)
        rate = payload.get("cache_hit_rate")
        if rate is None and prompt:
            rate = cached / prompt
        pct = f"{rate * 100:.0f}%" if rate is not None else "?"
        detail = f"cache_hit={pct} ({cached}/{prompt})"
        return f"{ts_pfx}  {indent}{colored_marker} {name}  {detail}"

    status: str = metadata.get("status") or ""
    suffix = f" [{status}]" if status else ""
    return f"{ts_pfx}  {indent}{colored_marker} {name}{suffix}"
