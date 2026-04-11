# MiniClaw Trace JSONL Schema

Stable schema reference for the trace file produced by
`miniclaw.observability.local.JsonlTracer` and consumed by
`tools/trace_view.py`. Update this file whenever you add a new span name,
a new metadata key that the viewer should highlight, or a new record kind.

**File location**: `<settings.trace_dir>/miniclaw.jsonl`
(default: `~/.miniclaw/traces/miniclaw.jsonl`)

**Format**: one JSON object per line, UTF-8, newline-terminated, append-only.
Multiple traces from multiple turns / threads interleave in the same file;
group by `trace_id` at read time.

---

## 1. Record envelope

Every line is a JSON object with this shape. Fields marked `nullable` may be
`null` or absent; readers should default gracefully.

| Field | Type | Description |
|---|---|---|
| `kind` | `"run_start" \| "run_finish" \| "span_start" \| "span_finish" \| "event"` | Record discriminator. |
| `timestamp` | `string` (ISO-8601, UTC, trailing `Z`) | e.g. `"2026-04-11T20:15:03.123456Z"`. |
| `trace_id` | `string` | Stable per trace, e.g. `"trace_a3f2b7e4c9d1..."`. |
| `run_id` | `string` | Per-run id within a trace, e.g. `"run_5e8f2a1b..."`. |
| `span_id` | `string \| null` | `null` for `run_*` records; otherwise `"span_<hex>"`. |
| `parent_span_id` | `string \| null` | `null` for root spans of a run and for `run_*` records. |
| `thread_id` | `string \| null` | Channel-level conversation id (e.g. `"tg:12345"`, `"cli:local"`). |
| `channel` | `string \| null` | `"telegram"`, `"cli"`, `"test"`, ... |
| `name` | `string` | Span / run / event name. See §3 for naming conventions. |
| `status` | `"ok" \| "error" \| null` | Set on `_finish` and `event` records; `null` on `_start`. |
| `metadata` | `object` | Structured context attached to the span. See §4 for keys the viewer understands. |
| `payload` | `object` | Used on `span_start` for inputs and on `event` for event payloads. |
| `output` | `object` | Used on `span_finish` and `run_finish` for outputs. |

### 1.1 Field conventions per record kind

| Kind | Populated fields |
|---|---|
| `run_start` | `trace_id`, `run_id`, `thread_id?`, `channel?`, `name`, `metadata`, `timestamp` |
| `run_finish` | `trace_id`, `run_id`, `name`, `status`, `metadata`, `output`, `timestamp` |
| `span_start` | `trace_id`, `run_id`, `span_id`, `parent_span_id`, `name`, `metadata`, `payload` (inputs), `timestamp` |
| `span_finish` | same as `span_start` plus `status`, `output`, minus `payload` |
| `event` | `trace_id`, `run_id`, `span_id?`, `parent_span_id?`, `name`, `status?`, `metadata`, `payload`, `timestamp` |

---

## 2. Tree reconstruction

Readers build the span tree by pairing `span_start` with the matching
`span_finish` on `span_id`, and chaining parents via `parent_span_id`.

```
run_start (run_id=R, name=turn.handle_message)
 └─ span_start (span_id=S1, parent_span_id=null, name=graph.agent)
    └─ span_start (span_id=S2, parent_span_id=S1, name=agent.tool_loop.round_0)
       ├─ span_start (span_id=S3, parent_span_id=S2, name=provider.achat)
       └─ span_finish (span_id=S3)
       └─ span_start (span_id=S4, parent_span_id=S2, name=tool.read_file)
       └─ span_finish (span_id=S4)
    └─ span_finish (span_id=S2)
 └─ span_finish (span_id=S1)
run_finish (run_id=R)
```

**Edge cases to handle**:

- A `span_start` with no matching `span_finish` (span still in progress / crashed
  mid-run) → viewer renders as status `running`.
- A `span_finish` with no matching `span_start` (tracer partially lost a record,
  or process crashed between them) → viewer synthesizes a minimal span from
  the finish record with `start_ts = null`.
- `parent_span_id` referencing a `span_id` that doesn't exist in this trace →
  attach to the trace root.
- Events can be standalone (`span_id = null`) or attached to a span. Standalone
  events go into `trace.orphan_events`.

---

## 3. Span names (stable conventions)

These prefixes are load-bearing for the viewer's coloring, filtering, and
grouping. Keep them stable when adding new spans.

### 3.1 Graph level

| Name | Emitted by | Parent | Notes |
|---|---|---|---|
| `graph.run` | `_wrap_node_with_state_snapshot` at graph.py | None (top level) | Synthetic per-node run context. |
| `graph.ingest` | wrapped `ingest` | `graph.run` | State: message append. |
| `graph.classify` | wrapped `make_classify` | `graph.run` | State: route decision. |
| `graph.clarify` | wrapped `clarify` | `graph.run` | Short-circuit path. |
| `graph.load_context` | wrapped `make_load_context` | `graph.run` | Memory retrieval call inside. |
| `graph.planner` | wrapped `make_planner` | `graph.run` | submit_plan call inside. |
| `graph.agent` | wrapped `make_agent` | `graph.run` | Contains the full agent tool loop subtree. |
| `graph.error_handler` | wrapped `error_handler` | `graph.run` | Terminal error formatting. |
| `graph.complete` | wrapped `complete` | `graph.run` | Final state shaping. |

### 3.2 Agent / subagent tool loop

| Name | Emitted by | Parent | Notes |
|---|---|---|---|
| `agent.tool_loop.round_<n>` | `make_agent` | `graph.agent` (internal) | `<n>` = 0-indexed round. |
| `agent.execute_tool_calls` | *(reserved, not yet emitted)* | round span | Currently not opened by code — kept for forward compatibility if dispatch batching gains its own span. |
| `subagent.run` | `run_subagent` | `tool.spawn_subagent` | Wraps the whole subagent lifetime. |
| `subagent.tool_loop.round_<n>` | `run_subagent` | `subagent.run` | Same shape as agent round. |

### 3.3 Provider

| Name | Emitted by | Parent | Notes |
|---|---|---|---|
| `provider.achat` | `make_agent`, `run_subagent` | `agent.tool_loop.round_<n>` or `subagent.tool_loop.round_<n>` | One per round. Wraps the LLM call. |

### 3.4 Tool

| Name | Emitted by | Parent | Notes |
|---|---|---|---|
| `tool.<tool_name>` | `_exec_one_with_span` in tool_loop.py | `agent.tool_loop.round_<n>` or `subagent.tool_loop.round_<n>` (**not** `agent.execute_tool_calls` — that span is not currently opened) | `<tool_name>` is the registry name, e.g. `tool.read_file`, `tool.shell`, `tool.spawn_subagent`, `tool.write_file`. |
| `tool.spawn_subagent` | same | round span | Its child is `subagent.run`, completing the recursive dispatch tree. |

### 3.5 Memory

| Name | Emitted by | Parent | Notes |
|---|---|---|---|
| `memory.retrieve` | `make_load_context` (future) | `graph.load_context` | Hybrid FTS5 + vec retrieval. Currently not opened by the runtime; reserved. |

### 3.6 Events (kind=`event`)

| Name | Attached to | Payload |
|---|---|---|
| `subagent_dispatched` | `subagent.run` | `{role, task_summary, fleet_id, sub_id}` in metadata + event payload. |
| `subagent_started` | `subagent.run` | `{sub_id}` in metadata. |
| `subagent_completed` | `subagent.run` | `{status, rounds_used}` in metadata; `{result_summary, error}` in payload. |
| *(any prompt-cache event)* | various | Reserved for future use. |

---

## 4. Metadata keys the viewer understands

Any other keys still get shown in the raw Metadata JSON panel; these ones
additionally drive badges, filters, and coloring.

### 4.1 Subagent metadata

| Key | On | Type | Meaning |
|---|---|---|---|
| `subagent.fleet_id` | `subagent.run`, `tool.spawn_subagent` | `string` (e.g. `"fleet-a3f2e1"`) | Groups all subagents dispatched in the same main-agent tool-loop round. Viewer colors them together. |
| `subagent.sub_id` | `subagent.run`, `tool.spawn_subagent` | `string` (e.g. `"fleet-a3f2e1-1"`) | Globally unique within a fleet. |
| `subagent.role` | `subagent.run` | `"researcher" \| "executor" \| "reviewer" \| <custom>` | Viewer uses role-specific tag colors. |
| `subagent.task_summary` | `subagent.run` | `string` (≤120 chars) | Short human label for the subagent's task. |
| `subagent.depth` | `subagent.run` | `integer` | Recursion depth; currently always `1`. |
| `subagent.tools_count` | `subagent.run` | `integer` | Number of tools available to this subagent. |

### 4.2 Provider metadata

| Key | On | Type | Meaning |
|---|---|---|---|
| `provider.model` | `provider.achat`, round spans | `string` | Model name used for this call. Viewer renders as a tag. |
| `provider.usage` | `provider.achat`, round spans | `{prompt_tokens, completion_tokens, total_tokens}` | Aggregated in the summary bar. |

### 4.3 Round metadata

| Key | On | Type |
|---|---|---|
| `tool_calls` | `subagent.tool_loop.round_<n>` _finish_ | `integer` |
| `tool_call_count` | `agent.tool_loop.round_<n>` _finish_ | `integer` |
| `rounds_used` | `subagent.run` _finish_ | `integer` |
| `rounds` | `graph.agent` _finish_ | `integer` |

### 4.4 Classify / planner metadata

| Key | On | Type | Meaning |
|---|---|---|---|
| `intent` | `graph.classify` _finish_ | `"clarify" \| "simple" \| "planned"` | Model output from the classify call. |
| `route` | `graph.classify` _finish_ | same | Stored in state; usually matches `intent`. |
| `classify.provider` | `graph.classify` _start_ | `"mini" \| "main"` | Which provider was used. |
| `briefs_count` | `graph.planner` _finish_ | `integer` | Number of `subagent_briefs` produced. |
| `planner.provider` | `graph.planner` _start_ | `"main"` | Always main model. |

### 4.5 Error metadata

| Key | On | Type | Meaning |
|---|---|---|---|
| `error` | any `_finish` with `status=error` | `string` | Short error string. |

---

## 5. Payload (inputs) conventions

### 5.1 `provider.achat` _start_ payload

```jsonc
{
  "messages": [
    {"role": "system", "content": "...", "content_parts_count": 2},
    {"role": "user",   "content": "hi"},
    {"role": "assistant", "tool_call_count": 1},
    {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "..."}
  ],
  "tool_count": 12
}
```

Note: only the **last 20** messages are kept (`TRACE_MESSAGE_LIMIT`). Each
message is compacted — `content` is a string preview, `content_parts_count`
indicates multimodal parts, `tool_call_count` and `tool_call_id` help reference
tool invocations without duplicating their bodies.

### 5.2 `agent.tool_loop.round_<n>` / `subagent.tool_loop.round_<n>` _start_ payload

```jsonc
{
  "messages": [ ... same compact format as above ... ],
  "tools": ["read_file", "write_file", "shell", "web_search", "memory_search"],
  "model": "qwen3-coder-plus"
}
```

### 5.3 `tool.<tool_name>` _start_ payload

```jsonc
{
  "arguments": {"path": "auth/middleware.py"}
}
```

Private context keys (`_on_event`, `_parent_trace`, `_sub_index`,
`current_fleet_id`) are stripped from the arguments before they land in
the payload — they're internal wiring, not tool inputs.

### 5.4 Graph node state-snapshot _start_ payload

The whitelisted state keys, plus derived counts:

```jsonc
{
  "route": "planned",
  "needs_clarification": false,
  "clarification_reason": "",
  "plan_summary": "investigate auth issue then patch",
  "subagent_briefs": [
    {"role": "researcher", "task": "..."},
    {"role": "executor",   "task": "...", "depends_on": [0]}
  ],
  "executor_notes": "",
  "last_error": "",
  "response_text": "",
  "message_count": 7,
  "memory_context_length": 842,
  "active_capabilities": {
    "skills": [], "tools": [], "mcp_servers": [], "mcp_tools": []
  }
}
```

Empty string / empty list values are elided. `user_input` and `messages` are
**never** recorded verbatim at the graph level — message bodies belong in
the tool-loop round spans.

---

## 6. Output conventions

### 6.1 `provider.achat` _finish_ output

```jsonc
{
  "content": "truncated response text (≤ max_tool_result_chars)",
  "tool_calls": [
    {"id": "c1", "name": "read_file", "arguments": {"path": "a.py"}}
  ],
  "usage": {"prompt_tokens": 284, "completion_tokens": 72, "total_tokens": 356}
}
```

### 6.2 Round span _finish_ output

Same shape as `provider.achat` output (the round's model response).

### 6.3 `tool.<tool_name>` _finish_ output

```jsonc
{
  "content": "file body (truncated)",
  "is_error": false,
  "metadata": {"path": "a.py", "mode": "overwrite", "bytes": 42}
}
```

### 6.4 Graph node _finish_ output

Same whitelisted state snapshot shape as the _start_ payload, but reflecting
the **post-node** state (what the node returned). Readers can diff start vs
finish to see what changed.

---

## 7. Truncation behavior

`JsonlTracer` replaces any string longer than `settings.trace_max_chars`
(default `2048`) with one of:

- `"{preview}...<truncated N chars>"` if `trace_full_content=true`
- `{"length": N, "preview": "...", "truncated": M}` object if
  `trace_full_content=false` (default)

The viewer detects the object form and renders it as
`[truncated string: N chars, showing preview]` instead of pretty-printing
it as a regular object. Both forms are nested-safe: truncation applies
recursively to every string inside `metadata`, `payload`, and `output`.

**Exception**: the compact message list produced by `trace_messages` keeps
each message short (content is the preview the tracer further truncates if
needed), and keeps only the 20 most recent. This prevents span bloat even
for multi-hundred-message conversations.

---

## 8. Timing

All `timestamp` values are UTC ISO-8601 with microsecond precision and a
trailing `Z`. The viewer computes `duration_ms` for each span as
`end_ts - start_ts` with fractional precision.

**Parallel spans**: when the main agent dispatches multiple
`tool.spawn_subagent` calls in one round, each spawn runs in its own thread
and opens its own `tool.spawn_subagent` span with overlapping start/end
timestamps. The flame-graph view correctly reflects the overlap.

---

## 9. Reserved / future extensions

These names are anticipated but not yet emitted. Add them here first, then
update the emitter.

- `memory.retrieve` — span around FTS5+vec hybrid retrieval, with payload
  `{top_k, query}` and output `{fts_hits, vec_hits, fused, results_count}`.
- `agent.execute_tool_calls` — would open around the phase-2 parallel tool
  dispatch inside `execute_tool_calls`, parent of individual `tool.<name>`
  spans. Currently skipped for simplicity; `tool.<name>` spans parent
  directly to the round.
- `prompt.cache.hit` / `prompt.cache.miss` — events attached to
  `provider.achat` when `usage.cache_read_input_tokens` /
  `cache_creation_input_tokens` are non-zero.

---

## 10. Viewer responsibilities

`tools/trace_view.py` relies on this schema for:

1. **Tree building**: pair `_start` / `_finish` by `span_id`; chain by
   `parent_span_id`. Handle orphans and running spans gracefully.
2. **Duration**: compute per-span duration from timestamps (`end - start`).
   Display as `Xms` / `X.Ys`.
3. **Coloring**: by name prefix (`graph` / `agent` / `subagent` / `tool` /
   `provider` / `memory`). Stable hues in `_HTML_TEMPLATE`'s CSS.
4. **Badges**: from the metadata keys in §4 (model, role, fleet, tokens,
   rounds).
5. **Fleet grouping**: visually grouping subagents with matching
   `subagent.fleet_id`.
6. **Filtering**: search substring matches against span name, metadata
   keys and values, payload, output.
7. **Filter chips**: "errors only" (span.status == `"error"`); "subagents
   only" (subtree contains `subagent.fleet_id` or name starts with
   `subagent`).
8. **Timeline**: stacks spans by depth, positions horizontally by timestamp.
9. **Details panel**: shows full metadata/payload/output JSON, plus event
   list for the selected span.
10. **Truncated string rendering**: detects the `{length, preview,
    truncated}` shape and renders it with a marker instead of normal JSON.

When you change the schema, update this file first, then the viewer if the
change affects coloring / badges / filtering, then the emitter.

---

## 11. Example record

```jsonc
{
  "channel": "cli",
  "kind": "span_finish",
  "metadata": {
    "provider.usage": {
      "completion_tokens": 72,
      "prompt_tokens": 284,
      "total_tokens": 356
    }
  },
  "name": "provider.achat",
  "output": {
    "content": "I'll read the file first.",
    "tool_calls": [
      {
        "arguments": {"path": "auth/middleware.py"},
        "id": "call_abc123",
        "name": "read_file"
      }
    ],
    "usage": {
      "completion_tokens": 72,
      "prompt_tokens": 284,
      "total_tokens": 356
    }
  },
  "parent_span_id": "span_00090000",
  "payload": {},
  "run_id": "run_5e8f2a1b",
  "span_id": "span_000a0000",
  "status": "ok",
  "thread_id": "cli:local",
  "timestamp": "2026-04-11T20:15:04.967456Z",
  "trace_id": "trace_a3f2b7e4c9d1"
}
```

---

## 12. Versioning

This schema is currently unversioned. If backward-incompatible changes
are needed (renaming a top-level field, dropping a kind), introduce a
`schema_version` top-level field first, bump it, and teach the viewer to
route by version. Adding new optional fields / new span names is always
safe — readers must tolerate unknown keys.
