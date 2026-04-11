# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniClaw is a minimal agent runtime built on LangGraph. It provides LLM-driven task planning, hybrid memory retrieval (FTS5 + sqlite-vec), multi-channel streaming output, and tool orchestration. Supports OpenAI-compatible model providers, thread-level checkpoint persistence, and pluggable channels (Telegram, CLI).

## Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/

# Run a single test
uv run pytest tests/test_runtime_graph.py -v

# Initialize local storage (~/.miniclaw/)
uv run miniclaw init

# Single-turn chat
uv run miniclaw chat "hello"

# Interactive REPL with Rich rendering
uv run miniclaw repl

# Start Telegram bot
uv run miniclaw telegram polling
```

## Architecture

### LangGraph State Machine (`miniclaw/runtime/`)

The core is a LangGraph `StateGraph` with `RuntimeState`. Flow:

```
ingest → classify → load_context → (clarify | agent | planner → agent) → complete
```

- **classify**: Mini-model routes intent to `clarify`, `simple`, or `planned` paths
- **planner** (planned only): LLM emits a lightweight `submit_plan` with `plan_summary` and an advisory `subagent_briefs` list. Empty briefs is valid — it tells the main agent to execute directly.
- **agent**: Unified tool-loop node (`make_agent` in `runtime/nodes.py`). The main agent decides at runtime whether to dispatch subagents via the `spawn_subagent` tool. Multiple spawn calls in one turn run in parallel via the existing thread-pool dispatch in `tool_loop.execute_tool_calls`. Subagents run in clean-room context (`runtime/subagent.py`) with role-default tool whitelists. Recursion depth is 1.
- **error_handler**: Any node that writes `last_error` is routed here via global conditional edges (`route_on_error` in `runtime/nodes.py`).

### Channel Abstraction (`miniclaw/channels/`)

`Channel` is a `@runtime_checkable` Protocol with `StreamCapability` metadata (NATIVE/EDIT/BUFFER). Implementations:
- **Telegram** (`channels/telegram/`): Uses `sendMessageDraft` for native streaming
- **CLI** (`channels/cli/`): Rich-based markdown rendering

`MessageLoop` (`channels/loop.py`) is the channel-agnostic orchestrator: access control → command interception → attachment processing → runtime invocation → 5-stage progressive streaming.

Subagent lifecycle emits fleet events (`subagent_dispatched`, `subagent_started`, `subagent_completed`) via `on_event`. The CLI channel does not yet render a fleet panel; events are available for future channel integration.

### Configuration (`miniclaw/config/settings.py`)

Pydantic `BaseSettings` with `MINICLAW_` prefix. All env vars documented in `.env.example`. Key groups: API/model config, Telegram, embedding/Ollama, search, runtime limits, tracing.

### Tool System (`miniclaw/tools/`)

Registry-based with two visibility levels:
- `always_active`: Available in every turn (e.g., send, memory_search)
- Discoverable: Activated on demand via `ActiveCapabilities`

Tools carry a `worker_visible` metadata flag (default `True`). Tools with `worker_visible: False` are filtered out before subagents see the tool list: `spawn_subagent`, `send`, `cron`, and `manage_heartbeat`. This prevents recursion and channel side-effects from subagents.

`spawn_subagent` is a builtin tool (not a graph node). It dispatches a `SubagentBrief` to `run_subagent` in `runtime/subagent.py`. Parallel spawns within the same agent turn share a fleet ID and execute concurrently via the thread-pool dispatch in `tool_loop.py`.

`tool_loop.py` (`runtime/tool_loop.py`) is the shared tool execution module used by both `make_agent` (main agent) and `run_subagent` (subagents).

Builtin tools in `tools/builtin/`: filesystem, shell, web search, cron, spawn_subagent, heartbeat, memory_search, skill activation, send message.

### Command System (`miniclaw/commands/`)

`@command` decorator auto-registers handlers. `CommandRegistry` matches input patterns, executes at channel layer with zero checkpoint pollution. Builtins: `/help`, `/status`, `/model`, `/clear`, `/stop`, `/new`, `/resume_run`, `/retry`, `/ping`.

### Memory (`miniclaw/memory/`)

Hybrid retrieval: FTS5 (BM25) + sqlite-vec (BGE-M3 1024-dim embeddings via Ollama) fused with RRF. Components: `retriever.py` (search), `indexer.py` (ingest), `chunker.py` (splitting), `embedding.py` (Ollama client), `context.py` (compression with dual trigger: 30K chars + 20 messages).

### MCP Integration (`miniclaw/mcp/`)

MCP server registry with stdio/SSE adapters. Config-driven (`mcp/config.py`), tools exposed through the main tool registry.

### Bootstrap (`miniclaw/bootstrap.py`)

Factory functions compose all dependencies: `build_runtime_service()`, `build_tool_registry()`, `build_mcp_registry()`. Optional components (Ollama, MCP, cron) degrade gracefully if unavailable.

### Access Control (`miniclaw/channels/access.py`)

Pairing system: 6-digit random code → `miniclaw pair <code>` → authorized. CLI is exempt from pairing.

## CLI Commands

- `miniclaw graph --format {mermaid,ascii,png}` — exports the current LangGraph topology
- `miniclaw trace tail <path> [--no-follow]` — pretty-prints a JSONL trace file with hierarchical indentation (depth via `parent_span_id`)
- `miniclaw chat`, `miniclaw repl`, `miniclaw telegram polling`, `miniclaw init`, `miniclaw pair` — existing commands unchanged

## Key Patterns

- **Async throughout**: All I/O is non-blocking async/await
- **Graceful degradation**: Optional components (Ollama embeddings, MCP servers, cron) don't break core functionality
- **Immutable state**: RuntimeState flows through graph nodes; new state returned per step
- **Protocol-based channels**: Declarative capability negotiation with automatic fallback
- **HTML parse_mode**: Telegram uses HTML (not MarkdownV2) for reliability
