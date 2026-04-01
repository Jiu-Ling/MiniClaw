from __future__ import annotations

from typing import Any

from miniclaw.memory.retriever import HybridRetriever
from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool
from miniclaw.utils.async_bridge import run_sync as _run_sync


def build_search_memory_tool(
    *,
    retriever: HybridRetriever,
    default_top_k: int = 10,
) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        query = str(call.arguments.get("query", "")).strip()
        if not query:
            return ToolResult(content="query is required", is_error=True)

        try:
            top_k = int(call.arguments.get("top_k", default_top_k) or default_top_k)
        except (TypeError, ValueError):
            return ToolResult(content="top_k must be an integer", is_error=True)

        thread_id = call.arguments.get("thread_id")
        if isinstance(thread_id, str):
            thread_id = thread_id.strip() or None
        else:
            thread_id = None

        date_range = _parse_date_range(call.arguments.get("date_range"))

        try:
            chunks = _run_sync(
                retriever.search(
                    query,
                    top_k=top_k,
                    thread_id=thread_id,
                    date_range=date_range,
                )
            )
        except Exception as exc:
            return ToolResult(content=f"memory search failed: {exc}", is_error=True)

        if not chunks:
            return ToolResult(content="No matching memories found.")

        lines = [f"Found {len(chunks)} relevant memories:\n"]
        for chunk in chunks:
            lines.append(
                f"[{chunk.created_at[:10]}] ({chunk.source_file}) score={chunk.score:.4f}\n"
                f"{chunk.content}\n"
            )
        return ToolResult(
            content="\n".join(lines),
            metadata={"query": query, "top_k": top_k, "result_count": len(chunks)},
        )

    return RegisteredTool(
        spec=ToolSpec(
            name="search_memory",
            description=(
                "Search long-term memory and daily logs using hybrid retrieval (keyword + semantic). "
                "Use this to recall past conversations, decisions, user preferences, or project context."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Number of results to return (default: {default_top_k})",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Filter results to a specific thread (optional)",
                    },
                    "date_range": {
                        "type": "object",
                        "description": "Filter by date range (optional)",
                        "properties": {
                            "start": {"type": "string", "description": "Start date, YYYY-MM-DD"},
                            "end": {"type": "string", "description": "End date, YYYY-MM-DD"},
                        },
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            source="builtin",
        ),
        executor=execute,
    )


def _parse_date_range(raw: Any) -> tuple[str, str] | None:
    if not isinstance(raw, dict):
        return None
    start = str(raw.get("start", "")).strip()
    end = str(raw.get("end", "")).strip()
    if start and end:
        return (start, end)
    return None



# _run_sync imported from miniclaw.utils.async_bridge
