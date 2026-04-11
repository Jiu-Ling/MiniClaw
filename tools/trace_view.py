#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""MiniClaw trace log visualizer.

Reads a JSONL trace file produced by `miniclaw.observability.local.JsonlTracer`
and renders it as a single self-contained interactive HTML page. No external
dependencies — pure Python stdlib + inline HTML/CSS/JS.

Usage:
    python tools/trace_view.py                                  # reads ~/.miniclaw/traces/miniclaw.jsonl
    python tools/trace_view.py path/to/trace.jsonl              # custom input
    python tools/trace_view.py trace.jsonl -o out.html          # custom output
    python tools/trace_view.py trace.jsonl --no-open            # don't auto-open browser

The generated HTML file embeds all trace data — shareable as a single file.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warning: line {line_no} is not valid JSON: {exc}")
    return records


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    v = value.rstrip("Z")
    try:
        return datetime.fromisoformat(v).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def duration_ms(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds() * 1000.0


def kind_prefix(name: str) -> str:
    if "." in name:
        return name.split(".", 1)[0]
    return name


# ----------------------------------------------------------------------
# Trace building
# ----------------------------------------------------------------------

def _new_trace(tid: str, rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": tid,
        "run_id": rec.get("run_id") or "",
        "thread_id": rec.get("thread_id"),
        "channel": rec.get("channel"),
        "start_ts": None,
        "end_ts": None,
        "duration_ms": None,
        "status": None,
        "name": "",
        "metadata": {},
        "output": {},
        "spans_by_id": {},
        "roots": [],
        "orphan_events": [],
    }


def _new_span(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "span_id": rec.get("span_id") or "",
        "parent_span_id": rec.get("parent_span_id"),
        "name": rec.get("name") or "",
        "kind_prefix": kind_prefix(rec.get("name") or ""),
        "start_ts": rec.get("timestamp"),
        "end_ts": None,
        "duration_ms": None,
        "status": None,
        "start_metadata": dict(rec.get("metadata") or {}),
        "end_metadata": {},
        "payload": dict(rec.get("payload") or {}),
        "output": {},
        "children": [],
        "events": [],
    }


def build_traces(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}

    for rec in records:
        tid = rec.get("trace_id") or "unknown"
        if tid not in traces:
            traces[tid] = _new_trace(tid, rec)
        trace = traces[tid]

        kind = rec.get("kind")
        if kind == "run_start":
            trace["start_ts"] = rec.get("timestamp")
            trace["name"] = rec.get("name") or trace["name"]
            trace["metadata"].update(rec.get("metadata") or {})
            trace["thread_id"] = trace["thread_id"] or rec.get("thread_id")
            trace["channel"] = trace["channel"] or rec.get("channel")
        elif kind == "run_finish":
            trace["end_ts"] = rec.get("timestamp")
            trace["status"] = rec.get("status") or trace["status"]
            trace["output"] = dict(rec.get("output") or {})
            trace["metadata"].update(rec.get("metadata") or {})
        elif kind == "span_start":
            sid = rec.get("span_id") or ""
            if sid not in trace["spans_by_id"]:
                trace["spans_by_id"][sid] = _new_span(rec)
        elif kind == "span_finish":
            sid = rec.get("span_id") or ""
            span = trace["spans_by_id"].get(sid)
            if span is None:
                span = _new_span(rec)
                span["start_ts"] = None
                trace["spans_by_id"][sid] = span
            span["end_ts"] = rec.get("timestamp")
            span["status"] = rec.get("status")
            span["end_metadata"].update(rec.get("metadata") or {})
            span["output"] = dict(rec.get("output") or {})
        elif kind == "event":
            event = {
                "name": rec.get("name") or "",
                "timestamp": rec.get("timestamp"),
                "status": rec.get("status"),
                "metadata": dict(rec.get("metadata") or {}),
                "payload": dict(rec.get("payload") or {}),
            }
            parent_sid = rec.get("span_id")
            if parent_sid and parent_sid in trace["spans_by_id"]:
                trace["spans_by_id"][parent_sid]["events"].append(event)
            else:
                trace["orphan_events"].append(event)

    for trace in traces.values():
        spans = trace["spans_by_id"]
        for span in spans.values():
            parent_sid = span["parent_span_id"]
            if parent_sid and parent_sid in spans:
                spans[parent_sid]["children"].append(span)
            else:
                trace["roots"].append(span)

        trace["duration_ms"] = duration_ms(
            parse_ts(trace["start_ts"]),
            parse_ts(trace["end_ts"]),
        )
        for span in spans.values():
            span["duration_ms"] = duration_ms(
                parse_ts(span["start_ts"]),
                parse_ts(span["end_ts"]),
            )

        trace["stats"] = compute_stats(trace)

    return traces


def compute_stats(trace: dict[str, Any]) -> dict[str, Any]:
    spans = list(trace["spans_by_id"].values())
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    models_used: set[str] = set()
    fleet_ids: set[str] = set()
    subagent_count = 0
    error_count = 0
    running_count = 0

    for s in spans:
        merged_meta = {**s.get("start_metadata", {}), **s.get("end_metadata", {})}
        usage = merged_meta.get("provider.usage")
        if isinstance(usage, dict):
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
        model = merged_meta.get("provider.model")
        if model:
            models_used.add(str(model))
        fleet = merged_meta.get("subagent.fleet_id")
        if fleet:
            fleet_ids.add(str(fleet))
        if s.get("name") == "subagent.run":
            subagent_count += 1
        if s.get("status") == "error":
            error_count += 1
        if s.get("status") is None:
            running_count += 1

    return {
        "span_count": len(spans),
        "error_count": error_count,
        "running_count": running_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "models_used": sorted(models_used),
        "fleet_count": len(fleet_ids),
        "subagent_count": subagent_count,
    }


# ----------------------------------------------------------------------
# HTML generation
# ----------------------------------------------------------------------

def serialize_traces_for_html(traces: dict[str, dict[str, Any]]) -> str:
    serialized: dict[str, Any] = {}
    for tid, trace in traces.items():
        spans_flat: dict[str, Any] = {}
        for sid, span in trace["spans_by_id"].items():
            flat = {k: v for k, v in span.items() if k != "children"}
            flat["children_ids"] = [c["span_id"] for c in span["children"]]
            spans_flat[sid] = flat
        serialized[tid] = {
            "trace_id": trace["trace_id"],
            "run_id": trace["run_id"],
            "thread_id": trace["thread_id"],
            "channel": trace["channel"],
            "start_ts": trace["start_ts"],
            "end_ts": trace["end_ts"],
            "duration_ms": trace["duration_ms"],
            "status": trace["status"],
            "name": trace["name"],
            "metadata": trace["metadata"],
            "output": trace["output"],
            "stats": trace["stats"],
            "orphan_events": trace["orphan_events"],
            "spans_flat": spans_flat,
            "root_ids": [s["span_id"] for s in trace["roots"]],
        }
    return json.dumps(serialized, ensure_ascii=False, default=str)


_HTML_PLACEHOLDER = "__MINICLAW_TRACE_DATA_PLACEHOLDER__"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniClaw Trace Viewer</title>
<style>
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background: #0d1117; color: #c9d1d9; font-size: 13px; }
code, pre, .mono { font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace; }
a { color: #58a6ff; }

header { display: flex; align-items: center; gap: 16px; padding: 10px 16px; border-bottom: 1px solid #30363d; background: #161b22; flex-wrap: wrap; }
header h1 { margin: 0; font-size: 15px; font-weight: 600; color: #f0f6fc; }
header .badge { padding: 2px 8px; background: #21262d; border-radius: 10px; font-size: 11px; color: #8b949e; }
header .search { flex: 1; min-width: 200px; }
header .search input { width: 100%; padding: 6px 10px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; color: #c9d1d9; font-size: 12px; }
header .search input:focus { outline: none; border-color: #58a6ff; }
header .filter-chip { display: inline-flex; align-items: center; gap: 4px; padding: 4px 8px; background: #21262d; border: 1px solid #30363d; border-radius: 12px; cursor: pointer; font-size: 11px; user-select: none; }
header .filter-chip.active { background: #1f6feb; border-color: #58a6ff; color: #f0f6fc; }
header .filter-chip:hover { border-color: #58a6ff; }

main { display: grid; grid-template-columns: 220px 1fr 380px; height: calc(100vh - 50px - 180px); overflow: hidden; }
aside.traces { border-right: 1px solid #30363d; overflow-y: auto; background: #0d1117; }
aside.traces .trace-item { padding: 10px 12px; border-bottom: 1px solid #21262d; cursor: pointer; }
aside.traces .trace-item:hover { background: #161b22; }
aside.traces .trace-item.active { background: #1f2937; border-left: 3px solid #58a6ff; padding-left: 9px; }
aside.traces .trace-item .tid { font-family: "JetBrains Mono", monospace; font-size: 11px; color: #8b949e; word-break: break-all; }
aside.traces .trace-item .meta { font-size: 11px; color: #6e7681; margin-top: 4px; }
aside.traces .trace-item .stats { margin-top: 4px; font-size: 10px; color: #8b949e; }
aside.traces .trace-item.error { border-left-color: #f85149; }

section.tree { overflow: auto; padding: 8px 12px; background: #0d1117; }
section.tree .span-node { padding: 3px 0; user-select: none; }
section.tree .span-row { display: flex; align-items: center; gap: 6px; padding: 3px 6px; border-radius: 4px; cursor: pointer; min-width: 0; }
section.tree .span-row:hover { background: #161b22; }
section.tree .span-row.selected { background: #1f2937; outline: 1px solid #58a6ff; }
section.tree .toggle { display: inline-block; width: 12px; text-align: center; color: #6e7681; font-size: 10px; }
section.tree .toggle.empty { color: transparent; }
section.tree .status-icon { width: 14px; text-align: center; font-size: 11px; }
section.tree .status-ok { color: #3fb950; }
section.tree .status-error { color: #f85149; }
section.tree .status-running { color: #d29922; }
section.tree .span-name { font-family: "JetBrains Mono", monospace; font-size: 12px; white-space: nowrap; }
section.tree .span-name.prefix-graph { color: #58a6ff; }
section.tree .span-name.prefix-agent { color: #7ee787; }
section.tree .span-name.prefix-subagent { color: #d2a8ff; }
section.tree .span-name.prefix-tool { color: #ffa657; }
section.tree .span-name.prefix-provider { color: #a5d6ff; }
section.tree .span-name.prefix-memory { color: #f0883e; }
section.tree .span-name.prefix-event { color: #e3b341; }
section.tree .span-duration { font-size: 10px; color: #6e7681; margin-left: auto; white-space: nowrap; }
section.tree .span-tags { display: flex; gap: 4px; }
section.tree .tag { font-size: 10px; padding: 1px 6px; background: #21262d; border-radius: 8px; color: #8b949e; white-space: nowrap; }
section.tree .tag.fleet { background: #30363d; color: #d2a8ff; }
section.tree .tag.role-researcher { background: #1f3a5f; color: #79c0ff; }
section.tree .tag.role-executor { background: #3b2314; color: #ffa657; }
section.tree .tag.role-reviewer { background: #2d1b3d; color: #d2a8ff; }
section.tree .tag.model { background: #0d2818; color: #7ee787; }
section.tree .tag.tokens { background: #1a1a2e; color: #a5d6ff; }
section.tree .children { margin-left: 16px; border-left: 1px dashed #30363d; padding-left: 4px; }
section.tree .span-node.hidden { display: none; }
section.tree .event-row { padding: 2px 6px 2px 22px; font-size: 11px; color: #e3b341; font-family: "JetBrains Mono", monospace; }
section.tree .event-row::before { content: "◆ "; color: #d29922; }

aside.details { border-left: 1px solid #30363d; overflow-y: auto; padding: 12px 14px; background: #0d1117; }
aside.details h2 { margin: 0 0 6px 0; font-size: 13px; font-weight: 600; color: #f0f6fc; font-family: "JetBrains Mono", monospace; word-break: break-all; }
aside.details .subtitle { color: #8b949e; font-size: 11px; margin-bottom: 14px; }
aside.details .section-title { margin: 16px 0 6px 0; font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
aside.details .kv { display: grid; grid-template-columns: 140px 1fr; gap: 2px 8px; font-size: 11px; font-family: "JetBrains Mono", monospace; }
aside.details .kv .k { color: #8b949e; }
aside.details .kv .v { color: #c9d1d9; word-break: break-word; }
aside.details .kv .v.status-ok { color: #3fb950; }
aside.details .kv .v.status-error { color: #f85149; }
aside.details .json { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 8px 10px; font-family: "JetBrains Mono", monospace; font-size: 11px; white-space: pre-wrap; word-break: break-word; color: #c9d1d9; max-height: 300px; overflow: auto; }
aside.details .json .key { color: #79c0ff; }
aside.details .json .string { color: #a5d6ff; }
aside.details .json .number { color: #79c0ff; }
aside.details .json .bool { color: #ff7b72; }
aside.details .json .null { color: #6e7681; }
aside.details .json .truncated { color: #d29922; font-style: italic; }
aside.details .placeholder { color: #6e7681; font-style: italic; text-align: center; margin-top: 40px; }
aside.details .jump-btn { display: inline-block; padding: 2px 6px; background: #21262d; border: 1px solid #30363d; border-radius: 4px; color: #58a6ff; font-size: 10px; cursor: pointer; text-decoration: none; }
aside.details .jump-btn:hover { background: #30363d; }
aside.details .event-item { padding: 6px 8px; background: #161b22; border-left: 2px solid #d29922; margin-bottom: 4px; font-size: 11px; font-family: "JetBrains Mono", monospace; }
aside.details .event-item .ename { color: #e3b341; }
aside.details .event-item .etime { color: #6e7681; font-size: 10px; }

footer.timeline { border-top: 1px solid #30363d; background: #0d1117; padding: 10px 16px 12px; height: 180px; overflow: auto; }
footer.timeline .tl-title { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
footer.timeline .tl-rows { display: flex; flex-direction: column; gap: 2px; }
footer.timeline .tl-row { position: relative; height: 16px; display: flex; align-items: center; }
footer.timeline .tl-bar { position: absolute; height: 14px; border-radius: 2px; cursor: pointer; overflow: hidden; font-size: 10px; color: #0d1117; padding: 0 4px; display: flex; align-items: center; white-space: nowrap; font-family: "JetBrains Mono", monospace; }
footer.timeline .tl-bar:hover { outline: 1px solid #f0f6fc; }
footer.timeline .tl-bar.selected { outline: 2px solid #58a6ff; }
footer.timeline .tl-bar.prefix-graph { background: #58a6ff; }
footer.timeline .tl-bar.prefix-agent { background: #7ee787; }
footer.timeline .tl-bar.prefix-subagent { background: #d2a8ff; }
footer.timeline .tl-bar.prefix-tool { background: #ffa657; }
footer.timeline .tl-bar.prefix-provider { background: #a5d6ff; }
footer.timeline .tl-bar.prefix-memory { background: #f0883e; }
footer.timeline .tl-bar.status-error { background: #f85149; color: #f0f6fc; }

.summary-bar { display: flex; gap: 14px; font-size: 11px; flex-wrap: wrap; }
.summary-bar .metric { display: flex; align-items: center; gap: 4px; }
.summary-bar .metric .label { color: #8b949e; }
.summary-bar .metric .val { color: #f0f6fc; font-weight: 600; font-family: "JetBrains Mono", monospace; }
.summary-bar .metric .val.error { color: #f85149; }
</style>
</head>
<body>
<header>
<h1>MiniClaw Trace Viewer</h1>
<span class="badge" id="trace-count-badge">-</span>
<div class="search"><input id="search-input" type="search" placeholder="Search span name, metadata, payload..."></div>
<span class="filter-chip" id="filter-errors" data-kind="error">errors only</span>
<span class="filter-chip" id="filter-subagents" data-kind="subagent">subagents only</span>
<div class="summary-bar" id="summary-bar"></div>
</header>
<main>
<aside class="traces" id="traces-list"></aside>
<section class="tree" id="tree-container"></section>
<aside class="details" id="details-container">
<div class="placeholder">Select a span to see its metadata, payload, and output.</div>
</aside>
</main>
<footer class="timeline">
<div class="tl-title">Timeline (flame graph)</div>
<div class="tl-rows" id="timeline-container"></div>
</footer>
<script>
window.TRACE_DATA = __MINICLAW_TRACE_DATA_PLACEHOLDER__;

const state = {
  activeTraceId: null,
  selectedSpanId: null,
  filterQuery: "",
  filterErrors: false,
  filterSubagents: false,
  collapsed: new Set(),
};

function el(tag, attrs, children) {
  const e = document.createElement(tag);
  if (attrs) {
    for (const k in attrs) {
      if (k === "class") e.className = attrs[k];
      else if (k === "text") e.textContent = attrs[k];
      else if (k === "html") e.innerHTML = attrs[k];
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    }
  }
  if (children) {
    for (const c of children) {
      if (c == null) continue;
      if (typeof c === "string") e.appendChild(document.createTextNode(c));
      else e.appendChild(c);
    }
  }
  return e;
}

function formatDuration(ms) {
  if (ms == null) return "";
  if (ms < 1) return ms.toFixed(2) + "ms";
  if (ms < 1000) return ms.toFixed(1) + "ms";
  if (ms < 60000) return (ms / 1000).toFixed(2) + "s";
  return Math.floor(ms / 60000) + "m" + Math.round((ms % 60000) / 1000) + "s";
}

function formatTokens(n) {
  if (n < 1000) return n.toString();
  if (n < 1000000) return (n / 1000).toFixed(1) + "k";
  return (n / 1000000).toFixed(2) + "M";
}

function statusIcon(status) {
  if (status === "ok") return { text: "●", cls: "status-ok" };
  if (status === "error") return { text: "✗", cls: "status-error" };
  return { text: "◌", cls: "status-running" };
}

function renderJson(obj, depth) {
  depth = depth || 0;
  if (obj === null) return '<span class="null">null</span>';
  if (typeof obj === "boolean") return '<span class="bool">' + obj + '</span>';
  if (typeof obj === "number") return '<span class="number">' + obj + '</span>';
  if (typeof obj === "string") return '<span class="string">' + JSON.stringify(obj) + '</span>';
  if (Array.isArray(obj)) {
    if (obj.length === 0) return "[]";
    const indent = "  ".repeat(depth + 1);
    const close = "  ".repeat(depth);
    return "[\n" + obj.map(x => indent + renderJson(x, depth + 1)).join(",\n") + "\n" + close + "]";
  }
  if (typeof obj === "object") {
    // Truncation marker
    if ("length" in obj && "preview" in obj && "truncated" in obj
        && typeof obj.length === "number" && typeof obj.preview === "string") {
      return '<span class="truncated">[truncated string: ' + obj.length + ' chars, showing preview]</span>\n'
        + '<span class="string">' + JSON.stringify(obj.preview) + '</span>';
    }
    const keys = Object.keys(obj);
    if (keys.length === 0) return "{}";
    const indent = "  ".repeat(depth + 1);
    const close = "  ".repeat(depth);
    const lines = keys.map(k =>
      indent + '<span class="key">' + JSON.stringify(k) + '</span>: ' + renderJson(obj[k], depth + 1)
    );
    return "{\n" + lines.join(",\n") + "\n" + close + "}";
  }
  return String(obj);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function renderSummary(trace) {
  const s = trace.stats;
  const parts = [
    `<div class="metric"><span class="label">spans</span><span class="val">${s.span_count}</span></div>`,
    `<div class="metric"><span class="label">duration</span><span class="val">${formatDuration(trace.duration_ms)}</span></div>`,
    `<div class="metric"><span class="label">tokens</span><span class="val">${formatTokens(s.total_tokens)}</span></div>`,
    `<div class="metric"><span class="label">subagents</span><span class="val">${s.subagent_count}</span></div>`,
    `<div class="metric"><span class="label">fleets</span><span class="val">${s.fleet_count}</span></div>`,
    `<div class="metric"><span class="label">errors</span><span class="val ${s.error_count ? 'error' : ''}">${s.error_count}</span></div>`,
  ];
  if (s.models_used.length) {
    parts.push(`<div class="metric"><span class="label">models</span><span class="val">${s.models_used.join(", ")}</span></div>`);
  }
  document.getElementById("summary-bar").innerHTML = parts.join("");
}

function renderTracesList() {
  const container = document.getElementById("traces-list");
  container.innerHTML = "";
  const traces = Object.values(window.TRACE_DATA);
  traces.sort((a, b) => (a.start_ts || "").localeCompare(b.start_ts || ""));
  document.getElementById("trace-count-badge").textContent = traces.length + " traces";
  for (const trace of traces) {
    const isActive = trace.trace_id === state.activeTraceId;
    const hasError = trace.stats.error_count > 0;
    const item = el("div", {
      class: "trace-item" + (isActive ? " active" : "") + (hasError ? " error" : ""),
      onclick: () => selectTrace(trace.trace_id),
    }, [
      el("div", { class: "tid mono", text: trace.trace_id.slice(0, 20) + (trace.trace_id.length > 20 ? "…" : "") }),
      el("div", { class: "meta" }, [
        (trace.thread_id || "—") + " · " + (trace.channel || "—"),
      ]),
      el("div", { class: "stats" }, [
        trace.stats.span_count + " spans · " +
        formatDuration(trace.duration_ms) +
        (trace.stats.total_tokens ? " · " + formatTokens(trace.stats.total_tokens) + " tok" : "") +
        (trace.stats.error_count ? " · " + trace.stats.error_count + " err" : "")
      ]),
    ]);
    container.appendChild(item);
  }
}

function buildSpanTags(span) {
  const tags = [];
  const meta = Object.assign({}, span.start_metadata || {}, span.end_metadata || {});
  const fleet = meta["subagent.fleet_id"];
  if (fleet) tags.push({ cls: "tag fleet", text: fleet });
  const role = meta["subagent.role"];
  if (role) tags.push({ cls: "tag role-" + role, text: role });
  const model = meta["provider.model"];
  if (model) tags.push({ cls: "tag model", text: model });
  const usage = meta["provider.usage"];
  if (usage && typeof usage === "object") {
    const tot = usage.total_tokens || ((usage.prompt_tokens || 0) + (usage.completion_tokens || 0));
    if (tot) tags.push({ cls: "tag tokens", text: tot + " tok" });
  }
  const rounds = meta["rounds_used"];
  if (rounds) tags.push({ cls: "tag", text: rounds + " rounds" });
  return tags;
}

function matchesFilter(span, trace) {
  const q = state.filterQuery.trim().toLowerCase();
  if (state.filterErrors && span.status !== "error") return false;
  if (state.filterSubagents) {
    const meta = Object.assign({}, span.start_metadata || {}, span.end_metadata || {});
    if (!meta["subagent.fleet_id"] && !String(span.name).startsWith("subagent")) return false;
  }
  if (!q) return true;
  if (String(span.name).toLowerCase().includes(q)) return true;
  const meta = Object.assign({}, span.start_metadata || {}, span.end_metadata || {});
  for (const k in meta) {
    if (k.toLowerCase().includes(q)) return true;
    if (JSON.stringify(meta[k]).toLowerCase().includes(q)) return true;
  }
  if (JSON.stringify(span.payload || {}).toLowerCase().includes(q)) return true;
  if (JSON.stringify(span.output || {}).toLowerCase().includes(q)) return true;
  return false;
}

function subtreeMatches(span, trace) {
  if (matchesFilter(span, trace)) return true;
  for (const cid of span.children_ids || []) {
    const child = trace.spans_flat[cid];
    if (child && subtreeMatches(child, trace)) return true;
  }
  return false;
}

function renderSpanNode(span, trace) {
  const hasChildren = (span.children_ids || []).length > 0;
  const isCollapsed = state.collapsed.has(span.span_id);
  const icon = statusIcon(span.status);
  const prefix = span.kind_prefix || "";
  const tags = buildSpanTags(span);
  const row = el("div", {
    class: "span-row" + (state.selectedSpanId === span.span_id ? " selected" : ""),
    onclick: (e) => {
      if (e.target.classList.contains("toggle")) return;
      selectSpan(span.span_id);
    },
  }, [
    el("span", {
      class: "toggle" + (hasChildren ? "" : " empty"),
      text: hasChildren ? (isCollapsed ? "▶" : "▼") : "·",
      onclick: (e) => { e.stopPropagation(); toggleCollapse(span.span_id); },
    }),
    el("span", { class: "status-icon " + icon.cls, text: icon.text }),
    el("span", { class: "span-name prefix-" + prefix, text: span.name }),
    ...tags.map(t => el("span", { class: t.cls, text: t.text })),
    el("span", { class: "span-duration", text: formatDuration(span.duration_ms) }),
  ]);
  const node = el("div", { class: "span-node", "data-span-id": span.span_id }, [row]);

  if (!matchesFilter(span, trace) && !subtreeMatches(span, trace)) {
    node.classList.add("hidden");
  }

  // Events inside the span
  for (const ev of (span.events || [])) {
    node.appendChild(el("div", { class: "event-row", text: ev.name + (ev.status ? " [" + ev.status + "]" : "") }));
  }

  if (hasChildren && !isCollapsed) {
    const childrenEl = el("div", { class: "children" });
    for (const cid of span.children_ids) {
      const child = trace.spans_flat[cid];
      if (child) childrenEl.appendChild(renderSpanNode(child, trace));
    }
    node.appendChild(childrenEl);
  }
  return node;
}

function renderTree() {
  const container = document.getElementById("tree-container");
  container.innerHTML = "";
  const trace = window.TRACE_DATA[state.activeTraceId];
  if (!trace) {
    container.innerHTML = '<div class="placeholder">No trace selected.</div>';
    return;
  }
  for (const rid of trace.root_ids) {
    const root = trace.spans_flat[rid];
    if (root) container.appendChild(renderSpanNode(root, trace));
  }
}

function renderDetails() {
  const container = document.getElementById("details-container");
  container.innerHTML = "";
  const trace = window.TRACE_DATA[state.activeTraceId];
  if (!trace) { container.innerHTML = '<div class="placeholder">No trace selected.</div>'; return; }
  const span = state.selectedSpanId ? trace.spans_flat[state.selectedSpanId] : null;

  if (!span) {
    // Show trace-level info
    container.appendChild(el("h2", { text: trace.name || trace.trace_id }));
    container.appendChild(el("div", { class: "subtitle", text: "trace · " + trace.trace_id }));
    const kvTrace = el("div", { class: "kv" });
    const kvPairs = [
      ["thread_id", trace.thread_id || "—"],
      ["channel", trace.channel || "—"],
      ["status", trace.status || "—"],
      ["start", trace.start_ts || "—"],
      ["end", trace.end_ts || "—"],
      ["duration", formatDuration(trace.duration_ms)],
      ["run_id", trace.run_id || "—"],
    ];
    for (const [k, v] of kvPairs) {
      kvTrace.appendChild(el("div", { class: "k", text: k }));
      const cls = "v" + (k === "status" && v === "error" ? " status-error" : (k === "status" && v === "ok" ? " status-ok" : ""));
      kvTrace.appendChild(el("div", { class: cls, text: String(v) }));
    }
    container.appendChild(kvTrace);

    if (Object.keys(trace.metadata || {}).length) {
      container.appendChild(el("div", { class: "section-title", text: "Trace metadata" }));
      container.appendChild(el("pre", { class: "json", html: renderJson(trace.metadata) }));
    }
    if (Object.keys(trace.output || {}).length) {
      container.appendChild(el("div", { class: "section-title", text: "Trace output" }));
      container.appendChild(el("pre", { class: "json", html: renderJson(trace.output) }));
    }
    return;
  }

  container.appendChild(el("h2", { text: span.name }));
  container.appendChild(el("div", { class: "subtitle", text: "span · " + span.span_id }));
  const kv = el("div", { class: "kv" });
  const mergedMeta = Object.assign({}, span.start_metadata || {}, span.end_metadata || {});
  const pairs = [
    ["status", span.status || "running"],
    ["duration", formatDuration(span.duration_ms)],
    ["start", span.start_ts || "—"],
    ["end", span.end_ts || "—"],
    ["kind", span.kind_prefix],
  ];
  if (mergedMeta["provider.model"]) pairs.push(["model", mergedMeta["provider.model"]]);
  if (mergedMeta["subagent.role"]) pairs.push(["role", mergedMeta["subagent.role"]]);
  if (mergedMeta["subagent.fleet_id"]) pairs.push(["fleet_id", mergedMeta["subagent.fleet_id"]]);
  if (mergedMeta["subagent.sub_id"]) pairs.push(["sub_id", mergedMeta["subagent.sub_id"]]);
  if (span.parent_span_id) pairs.push(["parent", span.parent_span_id.slice(0, 12) + "…"]);
  for (const [k, v] of pairs) {
    kv.appendChild(el("div", { class: "k", text: k }));
    let cls = "v";
    if (k === "status" && v === "ok") cls += " status-ok";
    if (k === "status" && v === "error") cls += " status-error";
    kv.appendChild(el("div", { class: cls, text: String(v) }));
  }
  container.appendChild(kv);

  if (span.parent_span_id) {
    const btn = el("a", {
      class: "jump-btn",
      href: "#",
      text: "↑ parent",
      onclick: (e) => { e.preventDefault(); selectSpan(span.parent_span_id); },
    });
    container.appendChild(el("div", { class: "section-title", text: "Navigation" }));
    container.appendChild(btn);
  }

  if (Object.keys(mergedMeta).length) {
    container.appendChild(el("div", { class: "section-title", text: "Metadata" }));
    container.appendChild(el("pre", { class: "json", html: renderJson(mergedMeta) }));
  }
  if (Object.keys(span.payload || {}).length) {
    container.appendChild(el("div", { class: "section-title", text: "Payload (inputs)" }));
    container.appendChild(el("pre", { class: "json", html: renderJson(span.payload) }));
  }
  if (Object.keys(span.output || {}).length) {
    container.appendChild(el("div", { class: "section-title", text: "Output" }));
    container.appendChild(el("pre", { class: "json", html: renderJson(span.output) }));
  }
  if ((span.events || []).length) {
    container.appendChild(el("div", { class: "section-title", text: "Events (" + span.events.length + ")" }));
    for (const ev of span.events) {
      const item = el("div", { class: "event-item" }, [
        el("div", {}, [
          el("span", { class: "ename", text: ev.name }),
          " ",
          el("span", { class: "etime", text: (ev.timestamp || "").replace("T", " ").replace("Z", "") }),
        ]),
      ]);
      if (Object.keys(ev.payload || {}).length) {
        item.appendChild(el("pre", { class: "json", html: renderJson(ev.payload) }));
      }
      container.appendChild(item);
    }
  }
}

function renderTimeline() {
  const container = document.getElementById("timeline-container");
  container.innerHTML = "";
  const trace = window.TRACE_DATA[state.activeTraceId];
  if (!trace) return;

  // Collect visible spans with valid timestamps
  const spans = Object.values(trace.spans_flat).filter(s => s.start_ts && s.end_ts && matchesFilter(s, trace));
  if (spans.length === 0) return;

  // Find earliest start and latest end
  const starts = spans.map(s => new Date(s.start_ts).getTime());
  const ends = spans.map(s => new Date(s.end_ts).getTime());
  const t0 = Math.min(...starts);
  const t1 = Math.max(...ends);
  const range = Math.max(t1 - t0, 1);

  // Assign each span to a row (simple greedy: find first row where its time window fits)
  const rows = [];
  const spanRow = new Map();
  // Sort by depth ascending so parents go above children
  const depthMap = new Map();
  function computeDepth(sid) {
    if (depthMap.has(sid)) return depthMap.get(sid);
    const s = trace.spans_flat[sid];
    if (!s) return 0;
    const d = s.parent_span_id && trace.spans_flat[s.parent_span_id] ? computeDepth(s.parent_span_id) + 1 : 0;
    depthMap.set(sid, d);
    return d;
  }
  for (const sid in trace.spans_flat) computeDepth(sid);
  spans.sort((a, b) => computeDepth(a.span_id) - computeDepth(b.span_id) || new Date(a.start_ts) - new Date(b.start_ts));

  for (const s of spans) {
    const depth = depthMap.get(s.span_id) || 0;
    while (rows.length <= depth) rows.push([]);
    rows[depth].push(s);
    spanRow.set(s.span_id, depth);
  }

  for (let i = 0; i < rows.length; i++) {
    const rowEl = el("div", { class: "tl-row" });
    for (const s of rows[i]) {
      const startMs = new Date(s.start_ts).getTime() - t0;
      const durMs = new Date(s.end_ts).getTime() - new Date(s.start_ts).getTime();
      const leftPct = (startMs / range) * 100;
      const widthPct = Math.max((durMs / range) * 100, 0.5);
      const isSelected = s.span_id === state.selectedSpanId;
      const bar = el("div", {
        class: "tl-bar prefix-" + (s.kind_prefix || "") + (s.status === "error" ? " status-error" : "") + (isSelected ? " selected" : ""),
        style: `left: ${leftPct}%; width: ${widthPct}%;`,
        title: s.name + " · " + formatDuration(s.duration_ms),
        onclick: () => selectSpan(s.span_id),
      });
      bar.textContent = s.name.split(".").slice(-1)[0];
      rowEl.appendChild(bar);
    }
    container.appendChild(rowEl);
  }
}

function selectTrace(tid) {
  state.activeTraceId = tid;
  state.selectedSpanId = null;
  state.collapsed.clear();
  renderTracesList();
  renderTree();
  renderDetails();
  renderTimeline();
  renderSummary(window.TRACE_DATA[tid]);
}

function selectSpan(sid) {
  state.selectedSpanId = sid;
  renderTree();
  renderDetails();
  renderTimeline();
}

function toggleCollapse(sid) {
  if (state.collapsed.has(sid)) state.collapsed.delete(sid);
  else state.collapsed.add(sid);
  renderTree();
}

function refresh() {
  renderTracesList();
  renderTree();
  renderTimeline();
}

document.getElementById("search-input").addEventListener("input", (e) => {
  state.filterQuery = e.target.value;
  refresh();
});
document.getElementById("filter-errors").addEventListener("click", (e) => {
  state.filterErrors = !state.filterErrors;
  e.target.classList.toggle("active", state.filterErrors);
  refresh();
});
document.getElementById("filter-subagents").addEventListener("click", (e) => {
  state.filterSubagents = !state.filterSubagents;
  e.target.classList.toggle("active", state.filterSubagents);
  refresh();
});

// Init: select first trace
const firstTid = Object.keys(window.TRACE_DATA)[0];
if (firstTid) selectTrace(firstTid);
else {
  document.getElementById("tree-container").innerHTML = '<div class="placeholder">No traces found in the input file.</div>';
}
</script>
</body>
</html>
"""


def generate_html(traces: dict[str, dict[str, Any]]) -> str:
    data_json = serialize_traces_for_html(traces)
    return HTML_TEMPLATE.replace(_HTML_PLACEHOLDER, data_json)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def default_trace_path() -> Path:
    return Path.home() / ".miniclaw" / "traces" / "miniclaw.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize a miniclaw JSONL trace file as interactive HTML.",
    )
    parser.add_argument(
        "path", nargs="?",
        help=f"Path to JSONL trace file (default: {default_trace_path()})",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output HTML path (default: same directory, .html suffix)",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Do not auto-open the generated HTML in a browser",
    )
    args = parser.parse_args()

    input_path = Path(args.path) if args.path else default_trace_path()
    if not input_path.exists():
        print(f"error: trace file not found: {input_path}")
        return 1

    records = parse_jsonl(input_path)
    if not records:
        print(f"error: no records found in {input_path}")
        return 1

    traces = build_traces(records)
    html = generate_html(traces)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".html")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    total_spans = sum(t["stats"]["span_count"] for t in traces.values())
    print(f"✓ parsed {len(records)} records → {len(traces)} trace(s), {total_spans} span(s)")
    print(f"✓ HTML written to {output_path}")

    if not args.no_open:
        try:
            webbrowser.open(output_path.as_uri())
        except Exception as exc:
            print(f"warning: could not auto-open browser: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
