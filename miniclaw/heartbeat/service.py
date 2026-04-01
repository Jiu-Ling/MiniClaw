from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Coroutine

from miniclaw.providers.contracts import ChatMessage


_HEARTBEAT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "heartbeat",
            "description": "Report heartbeat decision after reviewing tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["skip", "run"],
                        "description": "skip = nothing to do, run = has active tasks",
                    },
                    "tasks": {
                        "type": "string",
                        "description": "Natural-language summary of active tasks",
                    },
                },
                "required": ["action"],
            },
        },
    }
]


class HeartbeatService:
    def __init__(
        self,
        workspace: Path,
        provider: object,
        model: str,
        heartbeat_model: str | None = None,
        heartbeat_provider: object | None = None,
        on_execute: Callable[[str], Coroutine[Any, Any, str | None]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        interval_s: int = 30 * 60,
        enabled: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        self.provider = provider
        self.model = model
        self.heartbeat_model = heartbeat_model
        self.heartbeat_provider = heartbeat_provider
        self.on_execute = on_execute
        self.on_notify = on_notify
        self.interval_s = interval_s
        self.enabled = enabled
        self._running = False
        self._task: asyncio.Task[None] | None = None

    @staticmethod
    def _wrap_achat(achat: Callable) -> Callable:
        async def _chat_with_retry(messages, tools=None, model=None):
            normalized = [
                ChatMessage(**msg) if isinstance(msg, dict) else msg
                for msg in messages
            ]
            kwargs: dict[str, Any] = {}
            if model is not None:
                kwargs["model"] = model
            if tools is not None:
                kwargs["tools"] = tools
            return await achat(normalized, **kwargs)

        return _chat_with_retry

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    def _read_heartbeat_file(self) -> str | None:
        if not self.heartbeat_file.exists():
            return None
        content = self.heartbeat_file.read_text(encoding="utf-8").strip()
        return content or None

    async def _decide(self, content: str) -> tuple[str, str]:
        # Use independent heartbeat provider if configured, otherwise fall back to main provider
        decide_provider = self.heartbeat_provider or self.provider
        chat_with_retry = getattr(decide_provider, "chat_with_retry", None)
        if not callable(chat_with_retry):
            # Fallback to achat if chat_with_retry not available
            achat = getattr(decide_provider, "achat", None)
            if callable(achat):
                chat_with_retry = self._wrap_achat(achat)
            else:
                return "skip", ""

        from datetime import datetime
        now_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

        response = await chat_with_retry(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a heartbeat agent. Current time: {now_local}. "
                    "Call the heartbeat tool to report your decision.",
                },
                {
                    "role": "user",
                    "content": "Review the following HEARTBEAT.md and decide whether there are active tasks "
                    "that need execution now. Consider when each task was last executed.\n\n"
                    + content,
                },
            ],
            tools=_HEARTBEAT_TOOL,
            model=self.heartbeat_model or self.model,
        )
        if not getattr(response, "has_tool_calls", False):
            return "skip", ""
        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            return "skip", ""
        arguments = getattr(tool_calls[0], "arguments", {}) or {}
        return str(arguments.get("action", "skip")), str(arguments.get("tasks", ""))

    async def start(self) -> None:
        if not self.enabled or self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break

    async def _tick(self) -> None:
        content = self._read_heartbeat_file()
        if not content:
            return
        action, tasks = await self._decide(content)
        if action != "run" or not tasks.strip() or self.on_execute is None:
            return
        response = await self.on_execute(tasks)
        self._update_executed_tasks()
        if response and self.on_notify is not None:
            await self.on_notify(response)

    def _update_executed_tasks(self) -> None:
        try:
            from miniclaw.tools.builtin.heartbeat_manage import parse_heartbeat_tasks, update_last_executed

            hb_tasks = parse_heartbeat_tasks(self.heartbeat_file)
            for task in hb_tasks:
                update_last_executed(self.heartbeat_file, task.task_id)
        except Exception:
            pass  # Non-critical: don't break heartbeat if update fails

    async def trigger_now(self) -> str | None:
        content = self._read_heartbeat_file()
        if not content:
            return None
        action, tasks = await self._decide(content)
        if action != "run" or not tasks.strip() or self.on_execute is None:
            return None
        response = await self.on_execute(tasks)
        self._update_executed_tasks()
        if response and self.on_notify is not None:
            await self.on_notify(response)
        return response
