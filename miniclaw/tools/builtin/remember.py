"""`remember` tool — agent writes a discovered fact to long-term memory.

Phase 6 deliverable. Always-active and worker-visible so subagents can also
remember findings (paths, API endpoints, decisions). Per-turn rate limit
prevents fact spam. Critical-tier writes require a `reason` to discourage
overclassification of stable preferences.
"""
from __future__ import annotations

from miniclaw.memory.files import MemoryFileStore
from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool


def build_remember_tool(
    *,
    memory_store: MemoryFileStore,
    max_per_turn: int = 5,
) -> RegisteredTool:
    """Construct a fresh remember tool.

    The closure holds the per-turn counter, so a new instance must be built
    each turn (or the counter would persist across turns).
    """
    state = {"used": 0}

    def execute(call: ToolCall) -> ToolResult:
        fact = str(call.arguments.get("fact", "")).strip()
        if not fact:
            return ToolResult(content="ERROR: fact is required", is_error=True)

        tier_raw = str(call.arguments.get("tier", "normal")).strip().lower()
        tier = "critical" if tier_raw == "critical" else "normal"
        reason = str(call.arguments.get("reason", "")).strip()

        if tier == "critical" and not reason:
            return ToolResult(
                content="ERROR: reason is required when tier=critical",
                is_error=True,
            )

        if state["used"] >= max_per_turn:
            return ToolResult(
                content=f"ERROR: rate limit reached (max {max_per_turn} remember calls per turn)",
                is_error=True,
            )
        state["used"] += 1

        added = memory_store.add_fact(
            fact=fact,
            tier=tier,
            source="agent",
            reason=reason,
            dedup=True,
        )
        preview = fact[:80]
        if not added:
            return ToolResult(
                content=f"Already known (skipped duplicate): {preview}",
                metadata={"tier": tier, "added": False},
            )
        return ToolResult(
            content=f"Remembered ({tier}): {preview}",
            metadata={"tier": tier, "added": True},
        )

    return RegisteredTool(
        spec=ToolSpec(
            name="remember",
            description=(
                "Save a discovered fact to long-term memory so it persists across "
                "conversations. Call this proactively when you discover non-obvious "
                "facts (file paths, API endpoints, decisions, user preferences, "
                "bug locations). Don't wait until context is compressed — save facts "
                "when you find them. Duplicates are auto-deduplicated, so calling "
                "extra times is safe."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember. Be specific and self-contained.",
                        "maxLength": 300,
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["critical", "normal"],
                        "description": (
                            "critical = stable user preference (requires reason); "
                            "normal = useful project fact (default)"
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "Required if tier=critical. Why this is critical.",
                        "maxLength": 200,
                    },
                },
                "required": ["fact"],
                "additionalProperties": False,
            },
            source="builtin",
            metadata={"always_active": True, "worker_visible": True},
        ),
        executor=execute,
    )
