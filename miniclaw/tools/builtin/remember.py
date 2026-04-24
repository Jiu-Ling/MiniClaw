"""`remember` tool — agent writes a discovered fact to long-term memory.

Phase 6 deliverable. Always-active and worker-visible so subagents can also
remember findings (paths, API endpoints, decisions). Critical-tier writes
require a `reason` to discourage overclassification of stable preferences.

The `max_calls` parameter is a **process-wide soft cap** against runaway
loops, not a per-turn limit (the registry is built once per RuntimeService
and shared across turns). The real anti-spam defense is `add_fact`'s dedup.
True per-turn reset would require threading turn context through ToolCall,
which is V2 work.
"""
from __future__ import annotations

from miniclaw.memory.files import MemoryFileStore
from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool


def build_remember_tool(
    *,
    memory_store: MemoryFileStore,
    max_calls: int = 50,
) -> RegisteredTool:
    """Build the remember tool. `max_calls` is a process-wide soft cap.

    Set high enough that legitimate use never trips it; rely on `add_fact`'s
    dedup to prevent spam from being persisted. Default 50 means even a very
    chatty agent process won't hit the cap in normal operation.
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

        if state["used"] >= max_calls:
            return ToolResult(
                content=f"ERROR: process-wide soft cap reached ({max_calls} remember calls). Restart process or rely on dedup.",
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
