from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from miniclaw.providers.contracts import ChatMessage, ChatProvider
from miniclaw.tools.contracts import ToolCall
from miniclaw.utils.async_bridge import run_sync as _run_sync

if TYPE_CHECKING:
    from miniclaw.config.settings import Settings
    from miniclaw.tools.registry import ToolRegistry

_PHASE_ORDER = ("research", "execution", "review")
_WORKER_MAX_TOOL_ROUNDS = 8
_ROLE_TO_PHASE = {
    "researcher": "research",
    "executor": "execution",
    "reviewer": "review",
}
_ROLE_TO_KIND = {
    "researcher": "research",
    "executor": "execution",
    "reviewer": "review",
}
_KIND_TO_PHASE = {
    "research": "research",
    "execution": "execution",
    "review": "review",
}
_ROLE_PROMPTS = {
    "researcher": (
        "You are a researcher worker. Gather the most relevant context, constraints, and risks for the task. "
        "You have access to tools — use them when they are the most reliable way to get information. "
        "Prefer specific tools (e.g., shell with curl for weather) over general web search when possible."
    ),
    "executor": (
        "You are an executor worker. Carry out the task directly and describe the concrete result. "
        "You have access to tools — use them to execute actions and gather data."
    ),
    "reviewer": (
        "You are a reviewer worker. Evaluate the work, identify gaps, and suggest follow-up checks. "
        "You have access to tools — use them to verify results when needed."
    ),
}

WorkerRunner = Callable[[dict[str, object]], Awaitable[dict[str, object]]]
_WorkerRunnerWithContext = Callable[[dict[str, object], str, str], Awaitable[dict[str, object]]]


class WorkerManager:
    def __init__(
        self,
        *,
        provider: ChatProvider | None,
        settings: Settings | None,
        tool_registry: ToolRegistry | None,
        tracer: object | None,
        worker_runner: WorkerRunner | _WorkerRunnerWithContext | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.tool_registry = tool_registry
        self.tracer = tracer
        self._worker_runner = worker_runner
        self._worker_runner_accepts_context = self._accepts_context(worker_runner)

    def plan_batches(self, tasks: list[Mapping[str, Any]]) -> list[dict[str, object]]:
        phases: dict[str, list[dict[str, object]]] = {phase: [] for phase in _PHASE_ORDER}
        for raw_task in tasks:
            task = self._coerce_task(raw_task)
            phase = self._phase_name(task)
            if phase is None:
                continue
            phases[phase].append(task)

        return [
            {"name": phase_name, "tasks": phase_tasks}
            for phase_name in _PHASE_ORDER
            if (phase_tasks := phases[phase_name])
        ]

    def _coerce_task(self, task: Mapping[str, Any]) -> dict[str, object]:
        return {
            "id": str(task.get("id", "")).strip(),
            "title": str(task.get("title", "")).strip(),
            "kind": str(task.get("kind", "")).strip(),
            "status": str(task.get("status", "")).strip(),
            "worker_role": str(task.get("worker_role", "")).strip(),
            "parallel_group": str(task.get("parallel_group", "")).strip(),
            "expected_output": str(task.get("expected_output", "")).strip(),
        }

    def _phase_name(self, task: Mapping[str, object]) -> str | None:
        worker_role = str(task.get("worker_role", "")).strip()
        if worker_role:
            phase = _ROLE_TO_PHASE.get(worker_role)
            if phase is not None:
                return phase

        kind = str(task.get("kind", "")).strip()
        if kind:
            phase = _KIND_TO_PHASE.get(kind)
            if phase is not None:
                return phase
        return None

    def run(
        self,
        state: Mapping[str, Any] | None = None,
        *,
        tasks: list[Mapping[str, Any]] | None = None,
        user_input: str | None = None,
    ) -> dict[str, object]:
        resolved_state = state or {}
        raw_tasks = tasks if tasks is not None else resolved_state.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raw_tasks = []
        resolved_user_input = user_input if user_input is not None else str(resolved_state.get("user_input", ""))
        return _run_sync(self.arun(tasks=raw_tasks, user_input=resolved_user_input))

    async def arun(
        self,
        *,
        tasks: list[Mapping[str, Any]],
        user_input: str = "",
    ) -> dict[str, object]:
        prepared_tasks = [self._coerce_task(task) for task in tasks if isinstance(task, Mapping)]
        batches = self.plan_batches(prepared_tasks)
        if prepared_tasks and not batches:
            batches = [{"name": "execution", "tasks": prepared_tasks}]
        if not prepared_tasks:
            return {
                "worker_runs": [],
                "aggregated_worker_context": "",
                "orchestration_status": "idle",
            }

        worker_runs: list[dict[str, object]] = []
        shared_context = ""
        failed = False
        for batch in batches:
            batch_tasks = batch.get("tasks", [])
            if not isinstance(batch_tasks, list) or not batch_tasks:
                continue
            batch_runs = await self._run_batch(batch_tasks, user_input=user_input, shared_context=shared_context)
            worker_runs.extend(batch_runs)
            shared_context = self._format_completed_runs(worker_runs)
            if any(str(run.get("status", "")).strip() == "failed" for run in batch_runs):
                failed = True

        return {
            "worker_runs": worker_runs,
            "aggregated_worker_context": self._format_worker_context(batches, worker_runs),
            "orchestration_status": "failed" if failed else "complete",
        }

    def spawn(self, *, role: str, task: str, expected_output: object | None = None) -> dict[str, object]:
        batch = self.run(
            tasks=[
                {
                    "id": "spawned-task",
                    "title": task,
                    "kind": _ROLE_TO_KIND.get(role, ""),
                    "status": "pending",
                    "worker_role": role,
                    "parallel_group": _ROLE_TO_PHASE.get(role, ""),
                    "expected_output": "" if expected_output is None else str(expected_output),
                }
            ],
            user_input=task,
        )
        worker_runs = batch.get("worker_runs", [])
        if isinstance(worker_runs, list) and worker_runs:
            first_run = worker_runs[0]
            if isinstance(first_run, dict):
                return dict(first_run)
        return {
            "id": "worker-run-spawned-task",
            "task_id": "spawned-task",
            "role": role,
            "status": "failed",
            "summary": task,
            "error": "worker did not produce a result",
        }

    async def _run_batch(
        self,
        tasks: list[dict[str, object]],
        *,
        user_input: str,
        shared_context: str,
    ) -> list[dict[str, object]]:
        coroutines = [
            self._run_worker(task, user_input=user_input, shared_context=shared_context)
            for task in tasks
        ]
        return list(await asyncio.gather(*coroutines))

    async def _run_worker(
        self,
        task: dict[str, object],
        *,
        user_input: str,
        shared_context: str,
    ) -> dict[str, object]:
        if self._worker_runner is None:
            return await self._default_worker_runner(task, user_input=user_input, shared_context=shared_context)

        if self._worker_runner_accepts_context:
            return await self._worker_runner(task, user_input=user_input, shared_context=shared_context)
        return await self._worker_runner(task)

    async def _default_worker_runner(
        self,
        task: dict[str, object],
        *,
        user_input: str,
        shared_context: str,
    ) -> dict[str, object]:
        task_id = str(task.get("id", "")).strip() or "task"
        title = str(task.get("title", "")).strip() or task_id
        role = str(task.get("worker_role", "")).strip() or "worker"
        achat = getattr(self.provider, "achat", None)
        if not callable(achat):
            return self._failed_run(task, f"worker provider is not configured for role: {role}")

        messages = self._build_messages(task, user_input=user_input, shared_context=shared_context)
        model = self.settings.model if self.settings else None
        tools = self._build_worker_tools()

        try:
            for _ in range(_WORKER_MAX_TOOL_ROUNDS):
                response = await achat(messages, model=model, tools=tools or None)
                if not response.tool_calls:
                    break
                messages.append(ChatMessage(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                ))
                for raw_call in response.tool_calls:
                    func = raw_call.get("function", {})
                    tool_name = str(func.get("name", "")).strip()
                    try:
                        raw_args = func.get("arguments", "{}")
                        arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                    tool_result = self._execute_worker_tool(tool_name, arguments)
                    messages.append(ChatMessage(
                        role="tool",
                        name=tool_name,
                        tool_call_id=str(raw_call.get("id", "")),
                        content=tool_result,
                    ))
        except Exception as exc:
            return self._failed_run(task, str(exc))

        content = str(getattr(response, "content", "")).strip()
        summary = content.splitlines()[0].strip() if content else title
        if not summary:
            summary = title
        return {
            "id": f"worker-run-{task_id}",
            "task_id": task_id,
            "role": role,
            "status": "completed",
            "summary": summary,
            "result": content or summary,
        }

    def _build_worker_tools(self) -> list[dict[str, Any]]:
        """Build tool definitions visible to worker subagents."""
        if self.tool_registry is None:
            return []
        list_visible = getattr(self.tool_registry, "list_visible_tools", None)
        if not callable(list_visible):
            return []
        worker_excluded = {"spawn_worker", "send_message", "cron"}
        tools: list[dict[str, Any]] = []
        for spec in list_visible():
            if spec.name in worker_excluded:
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema or {"type": "object", "properties": {}},
                },
            })
        return tools

    def _execute_worker_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call from a worker subagent."""
        if self.tool_registry is None:
            return f"tool registry not available for tool: {name}"
        execute = getattr(self.tool_registry, "execute", None)
        if not callable(execute):
            return f"tool execution not available for tool: {name}"
        try:
            result = execute(ToolCall(name=name, arguments=arguments))
            return result.content
        except Exception as exc:
            return f"tool error: {exc}"

    def _build_messages(
        self,
        task: Mapping[str, object],
        *,
        user_input: str,
        shared_context: str,
    ) -> list[ChatMessage]:
        role = str(task.get("worker_role", "")).strip() or "researcher"
        title = str(task.get("title", "")).strip()
        expected_output = str(task.get("expected_output", "")).strip()

        system_parts = [_ROLE_PROMPTS.get(role, _ROLE_PROMPTS["executor"])]
        skills_block = self._build_active_skills_block()
        if skills_block:
            system_parts.append(skills_block)

        lines = [f"Task: {title or user_input or 'No task provided.'}"]
        if user_input.strip():
            lines.append(f"User request: {user_input.strip()}")
        if expected_output:
            lines.append(f"Expected output: {expected_output}")
        if shared_context.strip():
            lines.append(f"Prior worker context:\n{shared_context.strip()}")

        tool_hints = self._build_tool_hints()
        if tool_hints:
            lines.append(f"Available tools:\n{tool_hints}")

        return [
            ChatMessage(role="system", content="\n\n".join(system_parts)),
            ChatMessage(role="user", content="\n\n".join(lines)),
        ]

    def _build_active_skills_block(self) -> str:
        """Build active skills block for worker context."""
        if self.tool_registry is None:
            return ""
        skill_loader = getattr(self.tool_registry, "skill_loader", None)
        if skill_loader is None:
            return ""
        build_block = getattr(skill_loader, "build_active_skills_block", None)
        if not callable(build_block):
            return ""
        return build_block()

    def _build_tool_hints(self) -> str:
        """Generate a short list of available tools for worker context."""
        tools = self._build_worker_tools()
        if not tools:
            return ""
        lines: list[str] = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            if name:
                lines.append(f"- {name}: {desc[:80]}" if desc else f"- {name}")
        return "\n".join(lines)

    def _failed_run(self, task: Mapping[str, object], error: str) -> dict[str, object]:
        task_id = str(task.get("id", "")).strip() or "task"
        role = str(task.get("worker_role", "")).strip() or "worker"
        title = str(task.get("title", "")).strip() or task_id
        return {
            "id": f"worker-run-{task_id}",
            "task_id": task_id,
            "role": role,
            "status": "failed",
            "summary": title,
            "error": error,
        }

    def _format_completed_runs(self, runs: list[dict[str, object]]) -> str:
        if not runs:
            return ""
        lines = ["Completed worker results:"]
        for run in runs:
            role = str(run.get("role", "")).strip() or "worker"
            task_id = str(run.get("task_id", "")).strip() or "task"
            result = str(run.get("result", "")).strip() or str(run.get("error", "")).strip()
            lines.append(f"- [{role}] {task_id}: {result or 'no output'}")
        return "\n".join(lines)

    def _format_worker_context(
        self,
        batches: list[dict[str, object]],
        worker_runs: list[dict[str, object]],
    ) -> str:
        if not worker_runs:
            return ""

        task_lookup: dict[str, dict[str, object]] = {}
        for batch in batches:
            for task in batch.get("tasks", []):
                if isinstance(task, dict):
                    task_lookup[str(task.get("id", "")).strip()] = task

        lines = ["Worker orchestration context:"]
        for run in worker_runs:
            task_id = str(run.get("task_id", "")).strip()
            task = task_lookup.get(task_id, {})
            title = str(task.get("title", "")).strip() or task_id or "task"
            role = str(run.get("role", "")).strip() or str(task.get("worker_role", "")).strip() or "worker"
            status = str(run.get("status", "")).strip() or "unknown"
            detail = str(run.get("result", "")).strip() or str(run.get("error", "")).strip() or "no output"
            lines.append(f"- [{role}] {title} ({status}): {detail}")
        return "\n".join(lines)

    @staticmethod
    def _accepts_context(worker_runner: WorkerRunner | _WorkerRunnerWithContext | None) -> bool:
        if worker_runner is None:
            return False
        try:
            signature = inspect.signature(worker_runner)
        except (TypeError, ValueError):
            return False
        parameters = signature.parameters
        if "user_input" in parameters or "shared_context" in parameters:
            return True
        return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())



# _run_sync imported from miniclaw.utils.async_bridge


__all__ = ["WorkerManager"]
