# tools/

Standalone utilities that live alongside miniclaw but have no runtime dependency
on the `miniclaw` package. Each script should be runnable on its own and should
state its dependencies inline (PEP 723 script metadata) so that `uv run` can
execute it without any project-level setup.

## `trace_view.py` — Trace log visualizer

A single-file visualizer for `~/.miniclaw/traces/miniclaw.jsonl` (the JSONL file
produced by `miniclaw.observability.local.JsonlTracer`). Parses the trace,
builds the span tree from `parent_span_id` chains, and either:

- **Static mode** — generates a self-contained HTML file with all data
  embedded. Shareable as one file.
- **Serve mode** — starts a tiny local HTTP server that tails the trace
  file and pushes new records to the browser over Server-Sent Events, so
  the view updates live while a miniclaw run is writing records.

**Zero runtime dependencies** — pure Python stdlib + inline HTML/CSS/JS.

### Usage — static mode (default)

```bash
# default: read ~/.miniclaw/traces/miniclaw.jsonl, write .html next to it, auto-open
python tools/trace_view.py

# explicit input
python tools/trace_view.py path/to/trace.jsonl

# custom output path
python tools/trace_view.py trace.jsonl -o report.html

# generate only, do not open the browser
python tools/trace_view.py trace.jsonl --no-open
```

### Usage — serve mode (live)

```bash
# Default path + auto port + auto-open browser
python tools/trace_view.py --serve

# Explicit path
python tools/trace_view.py path/to/trace.jsonl --serve

# Fixed host/port (for remote / WSL)
python tools/trace_view.py --serve --host 0.0.0.0 --port 8787

# Serve without auto-opening the browser (handy over SSH)
python tools/trace_view.py --serve --port 8787 --no-open
```

While the server is running:

- `GET /` → the HTML UI (loads initial data via API, subscribes to SSE).
- `GET /api/initial` → full snapshot of every trace currently in the file,
  as one JSON object keyed by `trace_id`.
- `GET /api/stream` → Server-Sent Events; each `data:` event is one raw
  JSONL line that was just appended. The browser debounces multiple events
  and re-fetches `/api/initial` to avoid mirroring the Python parser in JS.
- `GET /api/ping` → `ok` health check.

The status pill in the header shows the connection state: `live` (SSE
connected), `reconnecting` (browser auto-retrying), `static` (file mode).

Equivalently with `uv` (no venv needed):

```bash
uv run tools/trace_view.py tools/sample_trace.jsonl
uv run tools/trace_view.py --serve
```

### UI features

- **Dark / light theme** — click the 🌙 / ☀️ button in the header. Preference
  is saved in `localStorage`; initial theme follows `prefers-color-scheme`.
- **Click any JSON panel** (Metadata / Payload / Output / Trace metadata /
  Trace output / Event payload) to pop a modal dialog with the full,
  untruncated JSON pretty-printed with syntax highlight. The dialog has a
  **Copy JSON** button that writes the un-highlighted JSON to the clipboard,
  and you can dismiss with **Esc**, the close button, or by clicking outside.
- **Search box** filters span tree by substring (name / metadata / payload /
  output).
- **Filter chips** — "errors only" hides non-error spans; "subagents only"
  keeps just the subagent subtree.

### What it shows

- **Trace list (left)** — one entry per `trace_id` found in the file, with
  `thread_id`, `channel`, span count, duration, total tokens, error count.
  Click to switch traces.
- **Span tree (center)** — recursive collapsible tree rooted at each top-level
  span for the active trace. Each node shows:
  - Status icon (`●` ok / `✗` error / `◌` running)
  - Span name, colored by prefix (`graph.*` blue, `agent.*` green,
    `subagent.*` purple, `tool.*` orange, `provider.*` light blue,
    `memory.*` amber)
  - Inline tags: `fleet_id`, `role`, `provider.model`, total tokens,
    `rounds_used`
  - Duration (right-aligned)
  - Events attached to the span are rendered inline as `◆ event_name`
- **Details panel (right)** — for the selected span (or for the trace itself
  if no span is selected):
  - Key-value summary (status, duration, timestamps, parent, model, role,
    fleet_id, sub_id)
  - Full merged metadata pretty-printed as JSON
  - Payload (inputs) pretty-printed as JSON
  - Output pretty-printed as JSON
  - Events list with per-event payload
  - "↑ parent" jump link
- **Timeline footer** — flame-graph-style horizontal bars, one row per depth,
  positioned by start timestamp and width by duration. Click a bar to select
  the corresponding span; the currently selected span is outlined.
- **Top bar** — summary metrics (spans, duration, tokens, subagents, fleets,
  errors, models) for the active trace. Search box matches across span name,
  metadata keys/values, payload, output. Filter chips toggle "errors only"
  and "subagents only".

### JSONL schema expected

The viewer parses records that match `miniclaw.observability.contracts.TraceRecord`:

```json
{
  "kind": "run_start" | "run_finish" | "span_start" | "span_finish" | "event",
  "timestamp": "2026-04-11T20:15:03.123456Z",
  "trace_id": "trace_<hex>",
  "run_id": "run_<hex>",
  "span_id": "span_<hex>" | null,
  "parent_span_id": "span_<hex>" | null,
  "thread_id": "...",
  "channel": "cli" | "telegram" | ...,
  "name": "graph.agent" | "subagent.run" | "provider.achat" | "tool.read_file" | ...,
  "status": "ok" | "error" | null,
  "metadata": { ... },
  "payload": { ... },
  "output":  { ... }
}
```

**Truncated strings**: `JsonlTracer` may replace long strings with
`{"length": N, "preview": "...", "truncated": M}` objects when
`trace_full_content=False`. The viewer detects this pattern and renders it
with a "[truncated string: N chars]" marker instead of pretty-printing it
as a normal object.

**Incomplete traces**: if a `span_start` has no matching `span_finish`, the
span is shown with status `◌ running`. If a `span_finish` has no matching
`span_start`, the viewer creates a minimal span from the finish record.

### Span name conventions understood

The visualization colors and groups spans by the name prefix (everything
before the first `.`). The conventions used by miniclaw's tracer:

| Prefix | Meaning | Color |
|---|---|---|
| `graph.<node>` | LangGraph node execution | blue |
| `agent.tool_loop.round_<n>` | Main agent tool-loop round | green |
| `agent.execute_tool_calls` | Parallel batch dispatch | green |
| `subagent.run` | One subagent execution | purple |
| `subagent.tool_loop.round_<n>` | Subagent tool-loop round | purple |
| `tool.<tool_name>` | Single tool invocation | orange |
| `provider.achat` | One LLM completion call | light blue |
| `memory.retrieve` | Memory retrieval | amber |

### Metadata fields the viewer knows about

Badges and details are keyed off these metadata fields:

- `subagent.fleet_id`, `subagent.sub_id`, `subagent.role`,
  `subagent.task_summary`, `subagent.depth`, `subagent.tools_count`
- `provider.model`, `provider.usage.prompt_tokens`,
  `provider.usage.completion_tokens`, `provider.usage.total_tokens`
- `rounds_used`
- `tool_calls`, `tool_call_count`

Other metadata keys are still rendered in the JSON section — the viewer
just doesn't give them dedicated badges.

## `sample_trace.jsonl`

An 85-record hand-generated trace showing the full subagent dispatch flow:
planner → agent with 2 parallel researchers (same fleet) → 1 executor
(separate fleet) → final response. Useful for demoing `trace_view.py`
without needing a real run and for regression-checking the parser.

```bash
python tools/trace_view.py tools/sample_trace.jsonl
```

The file exercises: multiple `agent.tool_loop` rounds, parallel
`tool.spawn_subagent` calls sharing a fleet_id, sequential fleet separation,
nested `subagent.tool_loop` inside `subagent.run`, per-span `provider.usage`
aggregation, `tool.read_file` / `tool.web_search` / `tool.shell` invocations,
and inline `subagent_dispatched` / `_started` / `_completed` events.

## Notes

- `tools/*.html` is gitignored — generated artifacts should not be committed.
- Add new tools here as single-file scripts with PEP 723 inline deps (or
  stdlib-only) for maximum portability. Avoid reaching into `miniclaw`.
