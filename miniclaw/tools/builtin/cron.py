from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from miniclaw.cron.service import CronService
from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool


def build_cron_tool(*, cron_service: CronService | None) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        if cron_service is None:
            return ToolResult(content="cron service is not configured", is_error=True)
        action = str(call.arguments.get("action", "")).strip().lower()
        if action == "add":
            channel = _ctx_str(call.context, "channel")
            chat_id = _ctx_str(call.context, "chat_id")
            message_thread_id = _ctx_str(call.context, "message_thread_id")

            at_value = _optional_str(call.arguments.get("at"))
            delay_seconds = _optional_int(call.arguments.get("delay_seconds"))
            if delay_seconds is not None and delay_seconds > 0 and not at_value:
                at_value = (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat()

            try:
                job = cron_service.add_job(
                    name=str(call.arguments.get("name", "")).strip() or "scheduled-task",
                    message=str(call.arguments.get("message", "")).strip(),
                    every_seconds=_optional_int(call.arguments.get("every_seconds")),
                    cron_expr=_optional_str(call.arguments.get("cron_expr")),
                    tz=_optional_str(call.arguments.get("tz")),
                    at=at_value,
                    deliver=True,
                    channel=channel,
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                )
            except ValueError as exc:
                return ToolResult(content=str(exc), is_error=True)
            return ToolResult(
                content=f"scheduled job {job.id}: {job.payload.message} (channel={channel}, chat_id={chat_id})",
                metadata={"job_id": job.id, "kind": job.schedule.kind},
            )
        if action == "list":
            jobs = cron_service.list_jobs()
            if not jobs:
                return ToolResult(content="No scheduled jobs.")
            lines = ["Scheduled jobs:"]
            for job in jobs:
                lines.append(f"- {job.id}: {job.name} [{job.schedule.kind}] {job.payload.message}")
            return ToolResult(content="\n".join(lines), metadata={"count": len(jobs)})
        if action == "remove":
            job_id = str(call.arguments.get("job_id", "")).strip()
            if not job_id:
                return ToolResult(content="job_id is required", is_error=True)
            if not cron_service.remove_job(job_id):
                return ToolResult(content=f"job not found: {job_id}", is_error=True)
            return ToolResult(content=f"removed job {job_id}", metadata={"job_id": job_id})
        return ToolResult(content="action must be one of: add, list, remove", is_error=True)

    return RegisteredTool(
        spec=ToolSpec(
            name="cron",
            description=(
                "Schedule reminders or recurring MiniClaw tasks. Use this when work must happen later "
                "or repeatedly instead of during the current turn. "
                "For one-time tasks use delay_seconds (e.g. 60 for 1 minute) — no need to compute ISO datetime. "
                "Delivery channel and chat_id are auto-filled from the current conversation context."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "list", "remove"]},
                    "name": {"type": "string"},
                    "message": {"type": "string"},
                    "delay_seconds": {"type": ["integer", "null"], "description": "Run once after this many seconds from now."},
                    "every_seconds": {"type": ["integer", "null"]},
                    "cron_expr": {"type": ["string", "null"]},
                    "tz": {"type": ["string", "null"]},
                    "at": {"type": ["string", "null"], "description": "ISO datetime for one-time execution. Prefer delay_seconds for relative times."},
                    "job_id": {"type": ["string", "null"]},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
            source="builtin",
        ),
        executor=execute,
    )


def _ctx_str(context: dict[str, Any], key: str) -> str | None:
    value = str(context.get(key, "") or "").strip()
    return value or None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
