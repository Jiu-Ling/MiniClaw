from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

from miniclaw.runtime.subagent import (
    ROLE_DEFAULTS,
    SubagentBrief,
    run_subagent,
)
from miniclaw.observability.contracts import NoopTracer, build_run_context
from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool

if TYPE_CHECKING:
    from miniclaw.config.settings import Settings
    from miniclaw.observability.contracts import Tracer
    from miniclaw.providers.contracts import ChatProvider
    from miniclaw.tools.registry import ToolRegistry


PRIVATE_KEYS = {"_on_event", "_parent_trace", "_sub_index", "current_fleet_id"}


def build_spawn_subagent_tool(
    *,
    provider: "ChatProvider",
    settings: "Settings",
    tool_registry: "ToolRegistry",
    tracer: "Tracer | None" = None,
) -> RegisteredTool:
    resolved_tracer = tracer or NoopTracer()

    def execute(call: ToolCall) -> ToolResult:
        role = str(call.arguments.get("role", "")).strip()
        task = str(call.arguments.get("task", "")).strip()
        if not role:
            return ToolResult(content="role is required", is_error=True)
        if not task:
            return ToolResult(content="task is required", is_error=True)

        expected = str(call.arguments.get("expected_output", "")).strip()
        raw_tools = call.arguments.get("tools")
        tools_tuple: tuple[str, ...] | None = None
        if isinstance(raw_tools, list):
            tools_tuple = tuple(str(t) for t in raw_tools if isinstance(t, str))
        if role not in ROLE_DEFAULTS and not tools_tuple:
            return ToolResult(
                content=f"custom role '{role}' requires explicit tools list",
                is_error=True,
            )

        raw_context = call.arguments.get("context")
        context_tuple: tuple[str, ...] = ()
        if isinstance(raw_context, list):
            context_tuple = tuple(str(c) for c in raw_context if isinstance(c, str))

        brief = SubagentBrief(
            role=role,
            task=task,
            expected_output=expected,
            tools=tools_tuple,
            context=context_tuple,
        )

        runtime_context = dict(call.context or {})
        fleet_id = str(runtime_context.get("current_fleet_id", "")).strip()
        if not fleet_id:
            fleet_id = f"fleet-{uuid4().hex[:6]}"
        sub_index = int(runtime_context.get("_sub_index", 0)) + 1
        sub_id = f"{fleet_id}-{sub_index}"

        on_event = runtime_context.get("_on_event")
        parent_trace = runtime_context.get("_parent_trace") or build_run_context(name="spawn_subagent")

        subagent_runtime_metadata = {k: v for k, v in runtime_context.items() if k not in PRIVATE_KEYS}

        try:
            result = asyncio.run(run_subagent(
                brief=brief,
                fleet_id=fleet_id,
                sub_id=sub_id,
                provider=provider,
                settings=settings,
                tool_registry=tool_registry,
                runtime_metadata=subagent_runtime_metadata,
                tracer=resolved_tracer,
                parent_trace=parent_trace,
                on_event=on_event,
            ))
        except Exception as exc:
            return ToolResult(
                content=f"subagent failed: {exc}",
                is_error=True,
                metadata={"sub_id": sub_id, "fleet_id": fleet_id},
            )

        if result.status == "failed":
            return ToolResult(
                content=f"status: failed\nerror: {result.error}\nsummary: {result.summary}",
                is_error=True,
                metadata={
                    "sub_id": result.sub_id,
                    "fleet_id": result.fleet_id,
                    "role": result.role,
                    "rounds_used": result.rounds_used,
                },
            )

        return ToolResult(
            content=f"status: completed\nresult: {result.result}",
            metadata={
                "sub_id": result.sub_id,
                "fleet_id": result.fleet_id,
                "role": result.role,
                "rounds_used": result.rounds_used,
                "summary": result.summary,
            },
        )

    return RegisteredTool(
        spec=ToolSpec(
            name="spawn_subagent",
            description=(
                "Dispatch a subagent to handle a focused subtask in an isolated context. "
                "Use roles 'researcher' (read-only investigation), 'executor' (carry out work), "
                "or 'reviewer' (verify). Multiple spawn calls in one turn run in parallel."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "task": {"type": "string"},
                    "expected_output": {"type": "string"},
                    "tools": {"type": "array", "items": {"type": "string"}},
                    "context": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["role", "task"],
                "additionalProperties": False,
            },
            source="builtin",
            metadata={"worker_visible": False, "discoverable": True},
        ),
        executor=execute,
    )


__all__ = ["build_spawn_subagent_tool"]
