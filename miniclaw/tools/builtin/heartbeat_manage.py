from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool

_TASK_PATTERN = re.compile(
    r"- \[hb-(\d+)\|(\d{4}-\d{2}-\d{2})\|last:([^\]]+)\] (.+)"
)


@dataclass(frozen=True)
class HeartbeatTask:
    task_id: str
    added: str
    last_executed: str
    description: str


def parse_heartbeat_tasks(heartbeat_file: Path) -> list[HeartbeatTask]:
    if not heartbeat_file.is_file():
        return []
    text = heartbeat_file.read_text(encoding="utf-8")
    tasks: list[HeartbeatTask] = []
    for match in _TASK_PATTERN.finditer(text):
        tasks.append(
            HeartbeatTask(
                task_id=f"hb-{match.group(1)}",
                added=match.group(2),
                last_executed=match.group(3).strip(),
                description=match.group(4).strip(),
            )
        )
    return tasks


def update_last_executed(heartbeat_file: Path, task_id: str) -> None:
    if not heartbeat_file.is_file():
        return
    text = heartbeat_file.read_text(encoding="utf-8")
    now = _local_now().strftime("%Y-%m-%dT%H:%M")
    id_num = task_id.removeprefix("hb-")

    def _replace(m: re.Match) -> str:
        if m.group(1) == id_num:
            return f"- [hb-{m.group(1)}|{m.group(2)}|last:{now}] {m.group(4)}"
        return m.group(0)

    updated = _TASK_PATTERN.sub(_replace, text)
    heartbeat_file.write_text(updated, encoding="utf-8")


def build_manage_heartbeat_tool(*, heartbeat_file: Path) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        action = str(call.arguments.get("action", "")).strip()

        if action == "add":
            return _handle_add(heartbeat_file, call)
        if action == "remove":
            return _handle_remove(heartbeat_file, call)
        if action == "list":
            return _handle_list(heartbeat_file)
        return ToolResult(content=f"unknown action: {action}", is_error=True)

    return RegisteredTool(
        spec=ToolSpec(
            name="manage_heartbeat",
            description=(
                "Add, remove, or list periodic tasks in HEARTBEAT.md. "
                "These tasks are reviewed by the heartbeat agent on each tick (every 15 minutes)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove", "list"],
                        "description": "Action to perform",
                    },
                    "task": {
                        "type": "string",
                        "description": "Task description (required for add)",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to remove, e.g. 'hb-001' (required for remove)",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
            source="builtin",
            metadata={"worker_visible": False},
        ),
        executor=execute,
    )


def _handle_add(heartbeat_file: Path, call: ToolCall) -> ToolResult:
    task_desc = str(call.arguments.get("task", "")).strip()
    if not task_desc:
        return ToolResult(content="task description is required for add", is_error=True)

    tasks = parse_heartbeat_tasks(heartbeat_file)
    max_id = max((int(t.task_id.removeprefix("hb-")) for t in tasks), default=0)
    next_id = max_id + 1
    task_id = f"hb-{next_id:03d}"
    added = _local_now().strftime("%Y-%m-%d")

    line = f"- [{task_id}|{added}|last:never] {task_desc}\n"

    _ensure_file(heartbeat_file)
    with heartbeat_file.open("a", encoding="utf-8") as f:
        f.write(line)

    return ToolResult(
        content=f"Added task [{task_id}]: {task_desc}",
        metadata={"task_id": task_id, "action": "add"},
    )


def _handle_remove(heartbeat_file: Path, call: ToolCall) -> ToolResult:
    task_id = str(call.arguments.get("task_id", "")).strip()
    if not task_id:
        return ToolResult(content="task_id is required for remove", is_error=True)

    tasks = parse_heartbeat_tasks(heartbeat_file)
    target = next((t for t in tasks if t.task_id == task_id), None)
    if target is None:
        return ToolResult(content=f"task {task_id} not found", is_error=True)

    text = heartbeat_file.read_text(encoding="utf-8")
    id_num = task_id.removeprefix("hb-")
    lines = text.splitlines(keepends=True)
    filtered = [
        line for line in lines
        if not re.match(rf"- \[hb-{re.escape(id_num)}\|", line)
    ]
    heartbeat_file.write_text("".join(filtered), encoding="utf-8")

    return ToolResult(
        content=f"Removed task [{task_id}]: {target.description}",
        metadata={"task_id": task_id, "action": "remove"},
    )


def _handle_list(heartbeat_file: Path) -> ToolResult:
    tasks = parse_heartbeat_tasks(heartbeat_file)
    if not tasks:
        return ToolResult(content="No heartbeat tasks configured.")

    lines: list[str] = []
    for t in tasks:
        last = f"last run: {t.last_executed}"
        lines.append(f"[{t.task_id}] {t.description} (added: {t.added}, {last})")
    return ToolResult(content="\n".join(lines))


def _ensure_file(heartbeat_file: Path) -> None:
    if not heartbeat_file.is_file():
        heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_file.write_text(
            "# Heartbeat\n\nAdd periodic tasks below that MiniClaw should review on each heartbeat tick.\n\n",
            encoding="utf-8",
        )


def _local_now() -> datetime:
    return datetime.now().astimezone()
