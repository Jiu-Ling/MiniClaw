from __future__ import annotations

from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool

SUPPORTED_WORKER_ROLES = frozenset({"researcher", "executor", "reviewer"})


def build_spawn_worker_tool(*, worker_manager: object) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        role = str(call.arguments.get("role", "")).strip()
        task = str(call.arguments.get("task", "")).strip()
        expected_output = call.arguments.get("expected_output")
        if role not in SUPPORTED_WORKER_ROLES:
            return ToolResult(content=f"unsupported worker role: {role or '<empty>'}", is_error=True)
        if not task:
            return ToolResult(content="task is required", is_error=True)

        spawn = getattr(worker_manager, "spawn", None)
        if not callable(spawn):
            return ToolResult(content="worker manager is not configured", is_error=True)

        run = spawn(role=role, task=task, expected_output=expected_output)
        if not isinstance(run, dict):
            return ToolResult(content="worker manager returned an invalid result", is_error=True)

        status = str(run.get("status", "")).strip()
        content = str(run.get("result", "")).strip() or str(run.get("summary", "")).strip()
        error = str(run.get("error", "")).strip()
        if status == "failed":
            return ToolResult(
                content=error or content or f"worker {role} failed",
                is_error=True,
                metadata={"worker_run": dict(run)},
            )
        return ToolResult(
            content=content or f"worker {role} completed",
            metadata={"worker_run": dict(run)},
        )

    return RegisteredTool(
        spec=ToolSpec(
            name="spawn_worker",
            description=(
                "Delegate one bounded task to a researcher, executor, or reviewer worker. "
                "Workers have access to the same tools as the main agent (shell, read_file, "
                "web_search, load_skill_tools, load_mcp_tools, etc.) and can call tools "
                "autonomously during their execution."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": sorted(SUPPORTED_WORKER_ROLES),
                    },
                    "task": {"type": "string"},
                    "expected_output": {"type": "string"},
                },
                "required": ["role", "task"],
                "additionalProperties": False,
            },
            source="builtin",
            metadata={"discoverable": True},
        ),
        executor=execute,
    )


__all__ = ["SUPPORTED_WORKER_ROLES", "build_spawn_worker_tool"]
