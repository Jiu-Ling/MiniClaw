from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from miniclaw.observability.contracts import NoopTracer
from miniclaw.runtime.tool_loop import resolve_turn_trace
from miniclaw.runtime.nodes import (
    clarify,
    complete,
    error_handler,
    ingest,
    make_agent,
    make_classify,
    make_load_context,
    make_planner,
    route_after_classify,
    route_after_load_context,
    route_after_planner,
    route_on_error,
)
from miniclaw.runtime.state import RuntimeState

if TYPE_CHECKING:
    from miniclaw.config.settings import Settings
    from miniclaw.memory.indexer import MemoryIndexer
    from miniclaw.memory.retriever import HybridRetriever
    from miniclaw.observability.contracts import TraceContext, Tracer
    from miniclaw.persistence.memory_store import MemoryStore
    from miniclaw.providers.contracts import ChatProvider
    from miniclaw.tools.registry import ToolRegistry


# State keys recorded into graph.<node> span inputs and outputs. Chosen for
# debuggability (what routing / state transitions happened) without bloating
# the trace file with full message history or memory context bodies.
_STATE_SNAPSHOT_KEYS: tuple[str, ...] = (
    "route",
    "needs_clarification",
    "clarification_reason",
    "plan_summary",
    "subagent_briefs",
    "executor_notes",
    "last_error",
    "response_text",
)


def _snapshot_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract a compact, loggable snapshot of a RuntimeState.

    Uses a whitelist of keys (routing signals, plan output, error state,
    response). Also derives counts for `messages` and `memory_context` so
    readers can see how big those were without storing their contents.
    """
    if not isinstance(state, Mapping):
        return {}
    snapshot: dict[str, Any] = {}
    for key in _STATE_SNAPSHOT_KEYS:
        if key in state:
            value = state.get(key)
            if value is None or value == "" or value == []:
                continue
            snapshot[key] = value

    messages = state.get("messages")
    if isinstance(messages, list):
        snapshot["message_count"] = len(messages)
    memory_context = state.get("memory_context")
    if isinstance(memory_context, str):
        snapshot["memory_context_length"] = len(memory_context)
    caps = state.get("active_capabilities")
    if caps is not None and hasattr(caps, "model_dump"):
        try:
            snapshot["active_capabilities"] = caps.model_dump()
        except Exception:
            pass
    return snapshot


def _wrap_node_with_state_snapshot(
    tracer: "Tracer",
    name: str,
    fn: Callable[[RuntimeState], RuntimeState | Mapping[str, Any]],
) -> Callable[[RuntimeState], RuntimeState | Mapping[str, Any]]:
    """Wrap a LangGraph node so every invocation opens a `graph.<name>` span
    with an input state snapshot and closes with an output snapshot (of the
    delta returned by the node). Failures propagate but always close the span.

    All wrapped invocations within the same turn share one ``trace_id`` by
    reading ``_turn_trace_id`` / ``_turn_run_id`` from ``runtime_metadata``
    (seeded by the service layer). This is what lets the trace viewer group
    every node + agent round + tool span of a turn under one logical trace.
    """
    span_name = f"graph.{name}"

    def wrapped(state: RuntimeState) -> RuntimeState | Mapping[str, Any]:
        parent = resolve_turn_trace(state, "graph.run")
        try:
            span = tracer.start_span(
                parent,
                name=span_name,
                inputs=_snapshot_state(state),
            )
        except Exception:
            span = None
        try:
            result = fn(state)
        except Exception as exc:
            if span is not None:
                try:
                    tracer.finish_span(
                        span,
                        status="error",
                        outputs={"error": str(exc)},
                    )
                except Exception:
                    pass
            raise
        if span is not None:
            try:
                tracer.finish_span(
                    span,
                    status="ok",
                    outputs=_snapshot_state(result) if isinstance(result, Mapping) else {},
                )
            except Exception:
                pass
        return result

    wrapped.__name__ = f"{name}_with_snapshot"
    return wrapped


def build_graph(
    *,
    settings: "Settings",
    provider: "ChatProvider",
    mini_provider: "ChatProvider | None" = None,
    memory_store: "MemoryStore",
    tool_registry: "ToolRegistry | None" = None,
    retriever: "HybridRetriever | None" = None,
    indexer: "MemoryIndexer | None" = None,
    memory_token_budget: int = 2000,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    tracer: "Tracer | None" = None,
    on_compression: "Callable[[Any], None] | None" = None,
) -> StateGraph:
    graph = StateGraph(RuntimeState)
    resolved_tracer = tracer or NoopTracer()

    def wrap(name: str, node: Callable[[RuntimeState], Any]) -> Callable[[RuntimeState], Any]:
        return _wrap_node_with_state_snapshot(resolved_tracer, name, node)

    # Nodes (wrapped with graph.<name> spans that capture state snapshots)
    graph.add_node("ingest", wrap("ingest", ingest))
    graph.add_node("classify", wrap(
        "classify",
        make_classify(mini_provider=mini_provider, main_provider=provider),
    ))
    graph.add_node("clarify", wrap("clarify", clarify))
    graph.add_node("load_context", wrap(
        "load_context",
        make_load_context(
            memory_store,
            retriever=retriever,
            indexer=indexer,
            memory_token_budget=memory_token_budget,
            mini_provider=mini_provider,
            main_provider=provider,
            settings=settings,
            tracer=resolved_tracer,
        ),
    ))
    graph.add_node("planner", wrap(
        "planner",
        make_planner(provider=provider, tool_registry=tool_registry),
    ))
    graph.add_node("agent", wrap(
        "agent",
        make_agent(
            settings=settings,
            provider=provider,
            tool_registry=tool_registry,
            on_event=on_event,
            tracer=tracer,
            on_compression=on_compression,
        ),
    ))
    graph.add_node("error_handler", wrap("error_handler", error_handler))
    graph.add_node("complete", wrap("complete", complete))

    # Edges
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "classify")

    graph.add_conditional_edges("classify", route_after_classify, {
        "clarify": "clarify",
        "load_context": "load_context",
        "error_handler": "error_handler",
    })
    graph.add_edge("clarify", "complete")

    graph.add_conditional_edges("load_context", route_after_load_context, {
        "agent": "agent",
        "planner": "planner",
        "error_handler": "error_handler",
    })

    graph.add_conditional_edges("planner", route_after_planner, {
        "agent": "agent",
        "error_handler": "error_handler",
    })

    graph.add_conditional_edges("agent", route_on_error("complete"), {
        "complete": "complete",
        "error_handler": "error_handler",
    })

    graph.add_edge("error_handler", "complete")
    graph.add_edge("complete", END)
    return graph
