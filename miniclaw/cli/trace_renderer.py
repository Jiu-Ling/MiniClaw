"""Pure rendering helpers for `miniclaw trace tail` and `miniclaw trace summary`.

No file I/O, no typer — only stdlib so these functions are easily unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable


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


# ---------------------------------------------------------------------------
# Aggregation data types
# ---------------------------------------------------------------------------


@dataclass
class ToolStat:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


@dataclass
class SpanStat:
    count: int = 0
    total_ms: float = 0.0
    errors: int = 0


@dataclass
class CacheStat:
    calls: int = 0
    total_prompt: int = 0
    total_cached: int = 0

    @property
    def hit_rate(self) -> float:
        return self.total_cached / self.total_prompt if self.total_prompt else 0.0


@dataclass
class ErrorEntry:
    name: str
    count: int = 0
    sample_msg: str = ""


@dataclass
class TraceSummary:
    runs: int = 0
    spans_ok: int = 0
    spans_err: int = 0
    events: int = 0
    tools: dict[str, ToolStat] = field(default_factory=dict)
    spans: dict[str, SpanStat] = field(default_factory=dict)
    cache: CacheStat = field(default_factory=CacheStat)
    errors: dict[str, ErrorEntry] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------------


def aggregate_trace(records: Iterable[dict]) -> TraceSummary:
    """Walk JSONL records and return aggregate stats."""
    summary = TraceSummary()
    open_spans: dict[str, dict] = {}  # span_id -> start record

    for rec in records:
        kind = rec.get("kind", "")
        if kind == "run_start":
            summary.runs += 1
        elif kind == "span_start":
            sid = rec.get("span_id")
            if sid:
                open_spans[sid] = rec
        elif kind == "span_finish":
            sid = rec.get("span_id")
            start = open_spans.pop(sid, None) if sid else None
            duration_ms = _duration_ms(start, rec) if start else 0.0
            name = rec.get("name", "")
            status = rec.get("status", "")

            if status == "error":
                summary.spans_err += 1
                err_msg = _extract_error_msg(rec)
                err = summary.errors.setdefault(name, ErrorEntry(name=name))
                err.count += 1
                if not err.sample_msg and err_msg:
                    err.sample_msg = err_msg
            else:
                summary.spans_ok += 1

            stat = summary.spans.setdefault(name, SpanStat())
            stat.count += 1
            stat.total_ms += duration_ms
            if status == "error":
                stat.errors += 1

            if name.startswith("tool.") or name.startswith("tool.call."):
                tool_name = (rec.get("metadata") or {}).get("tool.name") or name.split(".", 1)[-1]
                tstat = summary.tools.setdefault(tool_name, ToolStat())
                tstat.count += 1
                tstat.total_ms += duration_ms
                tstat.max_ms = max(tstat.max_ms, duration_ms)
        elif kind == "event":
            summary.events += 1
            if rec.get("name") == "prompt.cache.usage":
                payload = rec.get("payload") or {}
                pt = int(payload.get("prompt_tokens") or 0)
                ct = int(payload.get("cached_tokens") or 0)
                if pt > 0:
                    summary.cache.calls += 1
                    summary.cache.total_prompt += pt
                    summary.cache.total_cached += ct

    return summary


def _duration_ms(start: dict, finish: dict) -> float:
    """Compute milliseconds between start and finish ISO timestamps."""
    try:
        start_ts = parse_iso_ts(start.get("timestamp", ""))
        finish_ts = parse_iso_ts(finish.get("timestamp", ""))
        return max(0.0, (finish_ts - start_ts).total_seconds() * 1000.0)
    except Exception:
        return 0.0


def _extract_error_msg(rec: dict) -> str:
    """Pull first non-empty error string from finish record's output or metadata."""
    output = rec.get("output") or {}
    md = rec.get("metadata") or {}
    for source in (output, md):
        for k in ("error", "message", "reason"):
            v = source.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()[:80]
    return ""


# ---------------------------------------------------------------------------
# Summary renderer
# ---------------------------------------------------------------------------


def render_summary(summary: TraceSummary, *, top_n: int = 10) -> str:
    """Render the summary as a multi-line string for terminal display."""
    lines: list[str] = []

    total_spans = summary.spans_ok + summary.spans_err
    lines.append("Trace summary:")
    lines.append(f"  Runs:    {summary.runs}")
    lines.append(f"  Spans:   {total_spans}  ({summary.spans_ok} ok, {summary.spans_err} error)")
    lines.append(f"  Events:  {summary.events}")

    if summary.tools:
        lines.append("")
        lines.append("Top tools by count:")
        ranked_tools = sorted(summary.tools.items(), key=lambda x: x[1].count, reverse=True)[:top_n]
        name_w = max(len(n) for n, _ in ranked_tools) + 2
        for name, stat in ranked_tools:
            lines.append(
                f"  {name:<{name_w}}{stat.count:>3}  "
                f"total {format_duration(stat.total_ms):<8}"
                f"avg {format_duration(stat.avg_ms):<8}"
                f"max {format_duration(stat.max_ms)}"
            )

    if summary.spans:
        lines.append("")
        lines.append("Top spans by total duration:")
        ranked_spans = sorted(summary.spans.items(), key=lambda x: x[1].total_ms, reverse=True)[:top_n]
        name_w = max(len(n) for n, _ in ranked_spans) + 2
        for name, stat in ranked_spans:
            call_label = "call" if stat.count == 1 else "calls"
            lines.append(
                f"  {name:<{name_w}}{format_duration(stat.total_ms):<8}"
                f"({stat.count} {call_label})"
            )

    if summary.cache.calls > 0:
        rate_pct = summary.cache.hit_rate * 100
        lines.append("")
        lines.append("Cache:")
        lines.append(f"  Calls:           {summary.cache.calls}")
        lines.append(f"  Avg hit rate:    {rate_pct:.1f}%")
        lines.append(f"  Total prompt:    {summary.cache.total_prompt:,} tokens")
        lines.append(f"  Total cached:    {summary.cache.total_cached:,} tokens")

    if summary.errors:
        lines.append("")
        lines.append(f"Errors ({summary.spans_err}):")
        ranked_errors = sorted(summary.errors.items(), key=lambda x: x[1].count, reverse=True)[:top_n]
        name_w = max(len(n) for n, _ in ranked_errors) + 2
        for name, e in ranked_errors:
            sample = f"  ('{e.sample_msg}')" if e.sample_msg else ""
            lines.append(f"  {name:<{name_w}}{e.count}{sample}")

    return "\n".join(lines) + "\n"
