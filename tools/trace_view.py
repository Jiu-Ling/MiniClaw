#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""MiniClaw trace log visualizer.

Reads a JSONL trace file produced by `miniclaw.observability.local.JsonlTracer`
and renders it as either:

  1. A self-contained interactive HTML page with all data embedded (static mode).
  2. A live server that tails the trace file and pushes new records to the
     browser via Server-Sent Events (serve mode).

Pure Python stdlib — no runtime dependencies. Inline HTML/CSS/JS in the
template. Dark/light theme toggle persisted in localStorage. Click any
metadata / payload / output panel to pop a dialog with the full untruncated
value.

Usage:
    # Static export (default): write .html next to input, open browser.
    python tools/trace_view.py                                  # reads ~/.miniclaw/traces/miniclaw.jsonl
    python tools/trace_view.py path/to/trace.jsonl
    python tools/trace_view.py trace.jsonl -o out.html
    python tools/trace_view.py trace.jsonl --no-open

    # Serve mode: start a live-updating HTTP server (reads default path,
    # auto-tails the file, pushes new records to the browser over SSE).
    python tools/trace_view.py --serve                          # default path, auto port
    python tools/trace_view.py --serve --port 8787
    python tools/trace_view.py path/to/trace.jsonl --serve

The generated HTML file (static mode) embeds all trace data — shareable as
a single file.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
/* ---- Theme variables ---- */
:root {
  --bg:         #0d1117;
  --bg-panel:   #161b22;
  --bg-sunken:  #0d1117;
  --bg-hover:   #1f2937;
  --bg-chip:    #21262d;
  --bg-chip-2:  #30363d;
  --fg:         #c9d1d9;
  --fg-strong:  #f0f6fc;
  --fg-muted:   #8b949e;
  --fg-dim:     #6e7681;
  --border:     #30363d;
  --border-soft:#21262d;
  --link:       #58a6ff;
  --link-soft:  #1f6feb;
  --accent:     #58a6ff;
  --ok:         #3fb950;
  --error:      #f85149;
  --warn:       #d29922;
  --role-r:     #79c0ff;
  --role-e:     #ffa657;
  --role-v:     #d2a8ff;
  --span-graph: #58a6ff;
  --span-agent: #7ee787;
  --span-sub:   #d2a8ff;
  --span-tool:  #ffa657;
  --span-prov:  #a5d6ff;
  --span-mem:   #f0883e;
  --span-event: #e3b341;
  --dialog-backdrop: rgba(0, 0, 0, 0.6);
}
:root[data-theme="light"] {
  --bg:         #ffffff;
  --bg-panel:   #f6f8fa;
  --bg-sunken:  #ffffff;
  --bg-hover:   #eaeef2;
  --bg-chip:    #eaeef2;
  --bg-chip-2:  #d0d7de;
  --fg:         #1f2328;
  --fg-strong:  #0d1117;
  --fg-muted:   #656d76;
  --fg-dim:     #848f99;
  --border:     #d0d7de;
  --border-soft:#eaeef2;
  --link:       #0969da;
  --link-soft:  #dbeafe;
  --accent:     #0969da;
  --ok:         #1a7f37;
  --error:      #cf222e;
  --warn:       #9a6700;
  --role-r:     #0550ae;
  --role-e:     #bc4c00;
  --role-v:     #6639ba;
  --span-graph: #0969da;
  --span-agent: #1a7f37;
  --span-sub:   #6639ba;
  --span-tool:  #bc4c00;
  --span-prov:  #0969da;
  --span-mem:   #bc4c00;
  --span-event: #9a6700;
  --dialog-backdrop: rgba(30, 35, 40, 0.5);
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background: var(--bg); color: var(--fg); font-size: 13px; transition: background 0.2s, color 0.2s; }
code, pre, .mono { font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace; }
a { color: var(--link); }

header { display: flex; align-items: center; gap: 16px; padding: 10px 16px; border-bottom: 1px solid var(--border); background: var(--bg-panel); flex-wrap: wrap; }
header h1 { margin: 0; font-size: 15px; font-weight: 600; color: var(--fg-strong); }
header .badge { padding: 2px 8px; background: var(--bg-chip); border-radius: 10px; font-size: 11px; color: var(--fg-muted); }
header .status-pill { padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
header .status-pill.live { background: var(--ok); color: #ffffff; }
header .status-pill.paused { background: var(--warn); color: #ffffff; }
header .status-pill.offline { background: var(--fg-dim); color: var(--bg); }
header .search { flex: 1; min-width: 200px; }
header .search input { width: 100%; padding: 6px 10px; background: var(--bg-sunken); border: 1px solid var(--border); border-radius: 6px; color: var(--fg); font-size: 12px; }
header .search input:focus { outline: none; border-color: var(--accent); }
header .filter-chip { display: inline-flex; align-items: center; gap: 4px; padding: 4px 8px; background: var(--bg-chip); border: 1px solid var(--border); border-radius: 12px; cursor: pointer; font-size: 11px; user-select: none; color: var(--fg); }
header .filter-chip.active { background: var(--link-soft); border-color: var(--accent); color: var(--fg-strong); }
header .filter-chip:hover { border-color: var(--accent); }
header .icon-btn { cursor: pointer; background: var(--bg-chip); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: 14px; line-height: 1; color: var(--fg); user-select: none; }
header .icon-btn:hover { border-color: var(--accent); }

main { display: grid; grid-template-columns: 220px 1fr 380px; height: calc(100vh - 50px - 180px); overflow: hidden; }
aside.traces { border-right: 1px solid var(--border); overflow-y: auto; background: var(--bg); }
aside.traces .trace-item { padding: 10px 12px; border-bottom: 1px solid var(--border-soft); cursor: pointer; }
aside.traces .trace-item:hover { background: var(--bg-panel); }
aside.traces .trace-item.active { background: var(--bg-hover); border-left: 3px solid var(--accent); padding-left: 9px; }
aside.traces .trace-item .tid { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--fg-muted); word-break: break-all; }
aside.traces .trace-item .meta { font-size: 11px; color: var(--fg-dim); margin-top: 4px; }
aside.traces .trace-item .stats { margin-top: 4px; font-size: 10px; color: var(--fg-muted); }
aside.traces .trace-item.error { border-left-color: var(--error); }

section.tree { overflow: auto; padding: 8px 12px; background: var(--bg); }
section.tree .span-node { padding: 3px 0; user-select: none; }
section.tree .span-row { display: flex; align-items: center; gap: 6px; padding: 3px 6px; border-radius: 4px; cursor: pointer; min-width: 0; }
section.tree .span-row:hover { background: var(--bg-panel); }
section.tree .span-row.selected { background: var(--bg-hover); outline: 1px solid var(--accent); }
section.tree .toggle { display: inline-block; width: 12px; text-align: center; color: var(--fg-dim); font-size: 10px; }
section.tree .toggle.empty { color: transparent; }
section.tree .status-icon { width: 14px; text-align: center; font-size: 11px; }
section.tree .status-ok { color: var(--ok); }
section.tree .status-error { color: var(--error); }
section.tree .status-running { color: var(--warn); }
section.tree .span-name { font-family: "JetBrains Mono", monospace; font-size: 12px; white-space: nowrap; }
section.tree .span-name.prefix-graph { color: var(--span-graph); }
section.tree .span-name.prefix-agent { color: var(--span-agent); }
section.tree .span-name.prefix-subagent { color: var(--span-sub); }
section.tree .span-name.prefix-tool { color: var(--span-tool); }
section.tree .span-name.prefix-provider { color: var(--span-prov); }
section.tree .span-name.prefix-memory { color: var(--span-mem); }
section.tree .span-name.prefix-event { color: var(--span-event); }
section.tree .span-duration { font-size: 10px; color: var(--fg-dim); margin-left: auto; white-space: nowrap; }
section.tree .span-tags { display: flex; gap: 4px; }
section.tree .tag { font-size: 10px; padding: 1px 6px; background: var(--bg-chip); border-radius: 8px; color: var(--fg-muted); white-space: nowrap; }
section.tree .tag.fleet { background: var(--bg-chip-2); color: var(--role-v); }
section.tree .tag.role-researcher { color: var(--role-r); }
section.tree .tag.role-executor  { color: var(--role-e); }
section.tree .tag.role-reviewer  { color: var(--role-v); }
section.tree .tag.model  { color: var(--span-agent); }
section.tree .tag.tokens { color: var(--span-prov); }
section.tree .children { margin-left: 16px; border-left: 1px dashed var(--border); padding-left: 4px; }
section.tree .span-node.hidden { display: none; }
section.tree .event-row { padding: 2px 6px 2px 22px; font-size: 11px; color: var(--span-event); font-family: "JetBrains Mono", monospace; }
section.tree .event-row::before { content: "◆ "; color: var(--warn); }

aside.details { border-left: 1px solid var(--border); overflow-y: auto; padding: 12px 14px; background: var(--bg); }
aside.details h2 { margin: 0 0 6px 0; font-size: 13px; font-weight: 600; color: var(--fg-strong); font-family: "JetBrains Mono", monospace; word-break: break-all; }
aside.details .subtitle { color: var(--fg-muted); font-size: 11px; margin-bottom: 14px; }
aside.details .section-title { margin: 16px 0 6px 0; font-size: 11px; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
aside.details .section-title .expand-hint { font-size: 9px; color: var(--fg-dim); text-transform: none; letter-spacing: 0; font-weight: normal; opacity: 0; transition: opacity 0.15s; }
aside.details .kv { display: grid; grid-template-columns: 140px 1fr; gap: 2px 8px; font-size: 11px; font-family: "JetBrains Mono", monospace; }
aside.details .kv .k { color: var(--fg-muted); }
aside.details .kv .v { color: var(--fg); word-break: break-word; }
aside.details .kv .v.status-ok { color: var(--ok); }
aside.details .kv .v.status-error { color: var(--error); }
aside.details .json { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; font-family: "JetBrains Mono", monospace; font-size: 11px; white-space: pre-wrap; word-break: break-word; color: var(--fg); max-height: 240px; overflow: auto; cursor: zoom-in; position: relative; transition: border-color 0.15s, box-shadow 0.15s; }
aside.details .json:hover { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
aside.details .json:hover + .section-title .expand-hint,
aside.details .section-title:hover .expand-hint { opacity: 1; }
aside.details .json .key { color: var(--link); }
aside.details .json .string { color: var(--span-prov); }
aside.details .json .number { color: var(--link); }
aside.details .json .bool { color: var(--error); }
aside.details .json .null { color: var(--fg-dim); }
aside.details .json .truncated { color: var(--warn); font-style: italic; }
aside.details .placeholder { color: var(--fg-dim); font-style: italic; text-align: center; margin-top: 40px; }
aside.details .jump-btn { display: inline-block; padding: 2px 6px; background: var(--bg-chip); border: 1px solid var(--border); border-radius: 4px; color: var(--link); font-size: 10px; cursor: pointer; text-decoration: none; }
aside.details .jump-btn:hover { background: var(--bg-chip-2); }
aside.details .event-item { padding: 6px 8px; background: var(--bg-panel); border-left: 2px solid var(--warn); margin-bottom: 4px; font-size: 11px; font-family: "JetBrains Mono", monospace; }
aside.details .event-item .ename { color: var(--span-event); }
aside.details .event-item .etime { color: var(--fg-dim); font-size: 10px; }

footer.timeline { border-top: 1px solid var(--border); background: var(--bg); padding: 10px 16px 12px; height: 180px; overflow: auto; }
footer.timeline .tl-title { font-size: 11px; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
footer.timeline .tl-rows { display: flex; flex-direction: column; gap: 2px; }
footer.timeline .tl-row { position: relative; height: 16px; display: flex; align-items: center; }
footer.timeline .tl-bar { position: absolute; height: 14px; border-radius: 2px; cursor: pointer; overflow: hidden; font-size: 10px; color: var(--bg); padding: 0 4px; display: flex; align-items: center; white-space: nowrap; font-family: "JetBrains Mono", monospace; }
footer.timeline .tl-bar:hover { outline: 1px solid var(--fg-strong); }
footer.timeline .tl-bar.selected { outline: 2px solid var(--accent); }
footer.timeline .tl-bar.prefix-graph { background: var(--span-graph); }
footer.timeline .tl-bar.prefix-agent { background: var(--span-agent); }
footer.timeline .tl-bar.prefix-subagent { background: var(--span-sub); }
footer.timeline .tl-bar.prefix-tool { background: var(--span-tool); }
footer.timeline .tl-bar.prefix-provider { background: var(--span-prov); }
footer.timeline .tl-bar.prefix-memory { background: var(--span-mem); }
footer.timeline .tl-bar.status-error { background: var(--error); color: #ffffff; }

.summary-bar { display: flex; gap: 14px; font-size: 11px; flex-wrap: wrap; }
.summary-bar .metric { display: flex; align-items: center; gap: 4px; }
.summary-bar .metric .label { color: var(--fg-muted); }
.summary-bar .metric .val { color: var(--fg-strong); font-weight: 600; font-family: "JetBrains Mono", monospace; }
.summary-bar .metric .val.error { color: var(--error); }

/* ---- Detail dialog ---- */
#detail-dialog { padding: 0; max-width: min(90vw, 1100px); width: 80vw; max-height: 85vh; background: var(--bg-panel); color: var(--fg); border: 1px solid var(--border); border-radius: 10px; box-shadow: 0 20px 60px rgba(0,0,0,0.35); }
#detail-dialog::backdrop { background: var(--dialog-backdrop); backdrop-filter: blur(2px); }
#detail-dialog .dialog-head { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--bg-panel); }
#detail-dialog .dialog-head h3 { margin: 0; font-size: 14px; font-weight: 600; color: var(--fg-strong); flex: 1; font-family: "JetBrains Mono", monospace; }
#detail-dialog .dialog-head .dialog-subtitle { color: var(--fg-muted); font-size: 11px; }
#detail-dialog .dialog-head button { background: var(--bg-chip); border: 1px solid var(--border); border-radius: 6px; color: var(--fg); padding: 5px 12px; cursor: pointer; font-size: 12px; }
#detail-dialog .dialog-head button:hover { border-color: var(--accent); color: var(--fg-strong); }
#detail-dialog .dialog-body { padding: 14px 16px; font-family: "JetBrains Mono", monospace; font-size: 12px; color: var(--fg); white-space: pre-wrap; word-break: break-word; overflow: auto; max-height: calc(85vh - 60px); margin: 0; background: var(--bg); }
#detail-dialog .dialog-body .key { color: var(--link); }
#detail-dialog .dialog-body .string { color: var(--span-prov); }
#detail-dialog .dialog-body .number { color: var(--link); }
#detail-dialog .dialog-body .bool { color: var(--error); }
#detail-dialog .dialog-body .null { color: var(--fg-dim); }
#detail-dialog .dialog-body .truncated { color: var(--warn); font-style: italic; }
#copy-toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); background: var(--bg-panel); color: var(--fg-strong); border: 1px solid var(--border); padding: 8px 16px; border-radius: 6px; font-size: 12px; opacity: 0; transition: opacity 0.2s; pointer-events: none; z-index: 1000; }
#copy-toast.visible { opacity: 1; }
</style>
</head>
<body>
<header>
<h1>MiniClaw Trace Viewer</h1>
<span class="badge" id="trace-count-badge">-</span>
<span class="status-pill offline" id="live-status" title="Connection status">static</span>
<div class="search"><input id="search-input" type="search" placeholder="Search span name, metadata, payload..."></div>
<span class="filter-chip" id="filter-errors" data-kind="error">errors only</span>
<span class="filter-chip" id="filter-subagents" data-kind="subagent">subagents only</span>
<span class="icon-btn" id="theme-toggle" title="Toggle dark/light theme">🌙</span>
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

<dialog id="detail-dialog">
  <div class="dialog-head">
    <h3 id="dialog-title">Details</h3>
    <span class="dialog-subtitle" id="dialog-subtitle"></span>
    <button id="dialog-copy" type="button">Copy JSON</button>
    <button id="dialog-close" type="button">Close ✕</button>
  </div>
  <pre class="dialog-body" id="dialog-body"></pre>
</dialog>
<div id="copy-toast">Copied to clipboard</div>

<script>
window.TRACE_DATA = __MINICLAW_TRACE_DATA_PLACEHOLDER__;

const SERVE_MODE = !!(window.TRACE_DATA && window.TRACE_DATA.__SERVE_MODE__);
if (SERVE_MODE) window.TRACE_DATA = {};

const state = {
  activeTraceId: null,
  selectedSpanId: null,
  filterQuery: "",
  filterErrors: false,
  filterSubagents: false,
  collapsed: new Set(),
};

// ---- Theme ----
const THEME_KEY = "miniclaw-trace-viewer-theme";
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = theme === "light" ? "☀️" : "🌙";
}
(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch (e) {}
  if (!saved) {
    const prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
    saved = prefersLight ? "light" : "dark";
  }
  applyTheme(saved);
})();

// ---- Detail dialog ----
const dialogEl = document.getElementById("detail-dialog");
const dialogTitle = document.getElementById("dialog-title");
const dialogSubtitle = document.getElementById("dialog-subtitle");
const dialogBody = document.getElementById("dialog-body");
let dialogCurrentObj = null;

function openDetailDialog(title, subtitle, obj) {
  dialogCurrentObj = obj;
  dialogTitle.textContent = title;
  dialogSubtitle.textContent = subtitle || "";
  dialogBody.innerHTML = renderJson(obj);
  if (typeof dialogEl.showModal === "function") {
    try { dialogEl.showModal(); } catch (e) { dialogEl.setAttribute("open", ""); }
  } else {
    dialogEl.setAttribute("open", "");
  }
}
function closeDetailDialog() {
  if (typeof dialogEl.close === "function") { try { dialogEl.close(); } catch (e) {} }
  dialogEl.removeAttribute("open");
}
document.getElementById("dialog-close").addEventListener("click", closeDetailDialog);
dialogEl.addEventListener("click", (e) => {
  // Click on backdrop (dialog element itself, not a descendant) closes.
  if (e.target === dialogEl) closeDetailDialog();
});
document.getElementById("dialog-copy").addEventListener("click", () => {
  const text = dialogCurrentObj == null ? "" : JSON.stringify(dialogCurrentObj, null, 2);
  const done = () => {
    const toast = document.getElementById("copy-toast");
    toast.classList.add("visible");
    setTimeout(() => toast.classList.remove("visible"), 1200);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done).catch(() => {
      const ta = document.createElement("textarea");
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); } catch (e) {}
      document.body.removeChild(ta); done();
    });
  } else {
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(ta); done();
  }
});

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

function traceActivityTs(trace) {
  // "Last activity" for sorting: prefer end_ts (finished traces) falling
  // back to start_ts (running traces). Running traces with a recent
  // start_ts naturally stay at top; when they finish, end_ts updates to
  // the last span's end and they remain at top.
  return trace.end_ts || trace.start_ts || "";
}

function renderTracesList() {
  const container = document.getElementById("traces-list");
  container.innerHTML = "";
  const traces = Object.values(window.TRACE_DATA);
  // Newest first: sort by last-activity timestamp, descending.
  traces.sort((a, b) => traceActivityTs(b).localeCompare(traceActivityTs(a)));
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
      appendJsonPanel(container, "Trace metadata", trace.metadata, trace.trace_id);
    }
    if (Object.keys(trace.output || {}).length) {
      appendJsonPanel(container, "Trace output", trace.output, trace.trace_id);
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
    appendJsonPanel(container, "Metadata", mergedMeta, span.name);
  }
  if (Object.keys(span.payload || {}).length) {
    appendJsonPanel(container, "Payload (inputs)", span.payload, span.name);
  }
  if (Object.keys(span.output || {}).length) {
    appendJsonPanel(container, "Output", span.output, span.name);
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
        const pre = el("pre", {
          class: "json",
          html: renderJson(ev.payload),
          title: "Click to expand",
          onclick: () => openDetailDialog("Event payload · " + ev.name, (ev.timestamp || ""), ev.payload),
        });
        item.appendChild(pre);
      }
      container.appendChild(item);
    }
  }
}

function appendJsonPanel(container, title, obj, subtitle) {
  container.appendChild(el("div", { class: "section-title" }, [
    title,
    el("span", { class: "expand-hint", text: "click to expand" }),
  ]));
  const pre = el("pre", {
    class: "json",
    html: renderJson(obj),
    title: "Click to open full view",
    onclick: () => openDetailDialog(title, subtitle || "", obj),
  });
  container.appendChild(pre);
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

function selectTrace(tid, options) {
  const preserveState = options && options.preserveState;
  const sameTrace = state.activeTraceId === tid;
  state.activeTraceId = tid;
  // Preserve selection + collapse state on auto-refresh of the same trace.
  // Only reset when the user explicitly clicks a different trace.
  if (!preserveState && !sameTrace) {
    state.selectedSpanId = null;
    state.collapsed.clear();
  }
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

document.getElementById("theme-toggle").addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
});

// Close dialog on Escape even if default handler is disabled.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && dialogEl.hasAttribute("open")) closeDetailDialog();
});

// ---- Init and (optionally) live refresh via SSE ----
function setLiveStatus(kind, label) {
  const pill = document.getElementById("live-status");
  pill.classList.remove("live", "paused", "offline");
  pill.classList.add(kind);
  pill.textContent = label;
}

function initFromData() {
  // Pick the newest trace by last-activity ts (not insertion order).
  const traceList = Object.values(window.TRACE_DATA);
  traceList.sort((a, b) => traceActivityTs(b).localeCompare(traceActivityTs(a)));
  const newestTid = traceList.length ? traceList[0].trace_id : null;

  if (newestTid) {
    if (state.activeTraceId && window.TRACE_DATA[state.activeTraceId]) {
      // Same trace still present: auto-refresh path — preserve folded/
      // selected state so the view doesn't jump around when SSE fires.
      selectTrace(state.activeTraceId, { preserveState: true });
    } else {
      selectTrace(newestTid);
    }
  } else {
    document.getElementById("tree-container").innerHTML =
      '<div class="placeholder">No traces yet. Waiting for records...</div>';
    document.getElementById("traces-list").innerHTML = "";
    document.getElementById("details-container").innerHTML =
      '<div class="placeholder">No traces yet.</div>';
    document.getElementById("timeline-container").innerHTML = "";
    document.getElementById("summary-bar").innerHTML = "";
    document.getElementById("trace-count-badge").textContent = "0 traces";
  }
}

async function fetchInitial() {
  try {
    const r = await fetch("/api/initial", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    window.TRACE_DATA = await r.json();
  } catch (e) {
    setLiveStatus("offline", "offline");
    return false;
  }
  initFromData();
  return true;
}

let refreshTimer = null;
function scheduleRefresh() {
  if (refreshTimer) clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => { fetchInitial(); }, 400);
}

function openLiveStream() {
  if (typeof EventSource === "undefined") return;
  try {
    const es = new EventSource("/api/stream");
    es.onopen = () => setLiveStatus("live", "live");
    es.onmessage = () => scheduleRefresh();
    es.onerror = () => {
      // Browser will auto-reconnect; show paused while we wait.
      setLiveStatus("paused", "reconnecting");
    };
  } catch (e) {
    setLiveStatus("offline", "offline");
  }
}

if (SERVE_MODE) {
  setLiveStatus("paused", "loading");
  fetchInitial().then((ok) => {
    if (ok) openLiveStream();
  });
} else {
  setLiveStatus("offline", "static");
  initFromData();
}
</script>
</body>
</html>
"""


def generate_html(traces: dict[str, dict[str, Any]]) -> str:
    data_json = serialize_traces_for_html(traces)
    return HTML_TEMPLATE.replace(_HTML_PLACEHOLDER, data_json)


def generate_html_for_server() -> str:
    """HTML served in serve mode. Data is empty; the browser fetches via API.

    Injects a sentinel ``{"__SERVE_MODE__": true}`` so the client-side JS
    knows to fetch from `/api/initial` and open an SSE connection instead
    of reading from the embedded payload.
    """
    return HTML_TEMPLATE.replace(_HTML_PLACEHOLDER, '{"__SERVE_MODE__": true}')


# ----------------------------------------------------------------------
# Serve mode: file tailer + HTTP server
# ----------------------------------------------------------------------

class TraceTailer:
    """Background thread that tails a JSONL file and fans new lines out to
    subscribers over thread-safe queues.

    Subscribers receive only lines added AFTER they subscribe; they should
    fetch the initial snapshot (``get_initial_records``) atomically with
    subscribing by calling ``snapshot_and_subscribe``.
    """

    POLL_INTERVAL_S = 0.25
    SUBSCRIBER_MAX_QUEUE = 10_000

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: list[str] = []
        self._subscribers: list[queue.Queue[str]] = []
        self._lock = threading.Lock()
        self._last_pos = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Catch up the existing file contents synchronously, then start the
        background tail thread.

        The synchronous initial read closes a race where a client's
        ``/api/initial`` request arrives before the background thread's
        first poll (250ms later) and ends up seeing an empty snapshot
        even though the file is full of existing records.
        """
        if self._running:
            return
        try:
            self._read_new_lines()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"trace_view: initial catch-up error: {exc}")
        self._running = True
        self._thread = threading.Thread(target=self._tail_loop, daemon=True, name="trace-tailer")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def snapshot_and_subscribe(self) -> tuple[list[str], queue.Queue[str]]:
        """Return (current records snapshot, new subscriber queue) atomically.

        The queue will start receiving lines appended after this call, with
        no overlap and no gap relative to the returned snapshot.
        """
        q: queue.Queue[str] = queue.Queue(maxsize=self.SUBSCRIBER_MAX_QUEUE)
        with self._lock:
            snapshot = list(self._records)
            self._subscribers.append(q)
        return snapshot, q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _tail_loop(self) -> None:
        while self._running:
            try:
                self._read_new_lines()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"trace_view: tailer error: {exc}")
            time.sleep(self.POLL_INTERVAL_S)

    def _read_new_lines(self) -> None:
        if not self.path.exists():
            return
        new_lines: list[str] = []
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                # File may have been rotated / truncated. If current EOF is
                # smaller than our stored position, reset.
                fh.seek(0, 2)
                file_end = fh.tell()
                if file_end < self._last_pos:
                    self._last_pos = 0
                fh.seek(self._last_pos)
                for raw in fh:
                    line = raw.rstrip("\n\r")
                    if line:
                        new_lines.append(line)
                self._last_pos = fh.tell()
        except OSError:
            return
        if not new_lines:
            return
        with self._lock:
            self._records.extend(new_lines)
            subs = list(self._subscribers)
        for q in subs:
            for line in new_lines:
                try:
                    q.put_nowait(line)
                except queue.Full:
                    # Drop the oldest item and retry. A slow subscriber is
                    # better than a blocked tailer.
                    try:
                        q.get_nowait()
                        q.put_nowait(line)
                    except (queue.Empty, queue.Full):
                        pass


def _build_traces_from_raw(raw_lines: list[str]) -> dict[str, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return build_traces(records)


def _make_request_handler(tailer: TraceTailer, html_body: bytes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # quiet
            return

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            path = self.path.split("?", 1)[0]
            if path == "/" or path == "/index.html":
                self._send_bytes(html_body, "text/html; charset=utf-8")
            elif path == "/api/initial":
                self._send_initial()
            elif path == "/api/stream":
                self._send_stream()
            elif path == "/api/ping":
                self._send_bytes(b"ok", "text/plain; charset=utf-8")
            else:
                self.send_error(404)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_initial(self) -> None:
            snapshot, q = tailer.snapshot_and_subscribe()
            # Immediately unsubscribe — /api/initial is one-shot. The
            # /api/stream endpoint is used for live updates.
            tailer.unsubscribe(q)
            traces = _build_traces_from_raw(snapshot)
            body = serialize_traces_for_html(traces).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")

        def _send_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            _, q = tailer.snapshot_and_subscribe()
            try:
                # Tell the client its connection is live.
                try:
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                while True:
                    try:
                        line = q.get(timeout=15)
                    except queue.Empty:
                        # Heartbeat to keep proxies and browsers from timing out.
                        try:
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                        continue
                    # Escape newlines inside the payload so SSE framing stays valid.
                    safe = line.replace("\r", "").replace("\n", "\\n")
                    try:
                        self.wfile.write(f"data: {safe}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
            finally:
                tailer.unsubscribe(q)

    return Handler


def run_server(
    path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> int:
    """Start the live trace server. Blocks until interrupted.

    Returns 0 on clean shutdown, non-zero on startup error.
    """
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Create the file so the tailer has something to poll; a real run
        # will append to it. Empty file is fine — initial snapshot is [].
        path.touch()

    tailer = TraceTailer(path)
    tailer.start()

    html = generate_html_for_server().encode("utf-8")
    handler_cls = _make_request_handler(tailer, html)

    try:
        server = ThreadingHTTPServer((host, port), handler_cls)
    except OSError as exc:
        print(f"trace_view: could not bind {host}:{port}: {exc}")
        return 2

    server.daemon_threads = True
    actual_host, actual_port = server.server_address[0], server.server_address[1]
    url = f"http://{actual_host}:{actual_port}/"
    print(f"trace_view: serving {path}")
    print(f"trace_view: open {url}")
    print("trace_view: new records will stream live. Ctrl+C to stop.")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\ntrace_view: shutting down.")
    finally:
        server.server_close()
        tailer.stop()
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def default_trace_path() -> Path:
    return Path(__file__).parent.parent / ".miniclaw" / "traces" / "miniclaw.jsonl"


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
        help="Output HTML path for static mode (default: alongside input, .html suffix)",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Do not auto-open the browser",
    )
    parser.add_argument(
        "--serve", "-s", action="store_true",
        help="Start a live server that tails the file and pushes updates via SSE.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Serve mode: bind host (default 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=0,
        help="Serve mode: bind port (default: auto-pick)",
    )
    args = parser.parse_args()

    input_path = Path(args.path) if args.path else default_trace_path()

    if args.serve:
        return run_server(
            input_path,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )

    # Static export mode.
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
