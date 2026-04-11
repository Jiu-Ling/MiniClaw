from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

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
    from miniclaw.observability.contracts import Tracer
    from miniclaw.persistence.memory_store import MemoryStore
    from miniclaw.providers.contracts import ChatProvider
    from miniclaw.tools.registry import ToolRegistry


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
) -> StateGraph:
    graph = StateGraph(RuntimeState)

    # Nodes
    graph.add_node("ingest", ingest)
    graph.add_node("classify", make_classify(
        mini_provider=mini_provider,
        main_provider=provider,
    ))
    graph.add_node("clarify", clarify)
    graph.add_node("load_context", make_load_context(
        memory_store,
        retriever=retriever,
        indexer=indexer,
        memory_token_budget=memory_token_budget,
    ))
    graph.add_node("planner", make_planner(
        provider=provider,
        tool_registry=tool_registry,
    ))
    graph.add_node("agent", make_agent(
        settings=settings,
        provider=provider,
        tool_registry=tool_registry,
        on_event=on_event,
        tracer=tracer,
    ))
    graph.add_node("error_handler", error_handler)
    graph.add_node("complete", complete)

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
