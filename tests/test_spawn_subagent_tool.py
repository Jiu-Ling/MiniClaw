import asyncio
from collections.abc import Sequence
from unittest.mock import MagicMock

from miniclaw.observability.contracts import NoopTracer
from miniclaw.providers.contracts import ChatResponse
from miniclaw.tools.builtin.spawn_subagent import build_spawn_subagent_tool
from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool, ToolRegistry


class _ScriptedProvider:
    capabilities = MagicMock()

    def __init__(self, responses: Sequence[ChatResponse]) -> None:
        self.responses = list(responses)

    async def achat(self, messages, *, model=None, tools=None, **kwargs):
        if not self.responses:
            raise RuntimeError("provider exhausted")
        return self.responses.pop(0)

    async def astream_text(self, messages, *, model=None, tools=None, **kwargs):
        raise NotImplementedError


def _make_chat_response(content: str, tool_calls=None) -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        provider="test",
    )


def _settings():
    s = MagicMock()
    s.model = "fake"
    s.max_tool_rounds = 4
    s.max_consecutive_tool_errors = 4
    s.max_tool_result_chars = 16_000
    return s


def _registry():
    r = ToolRegistry(skill_loader=None, mcp_registry=None)
    r.register(RegisteredTool(
        spec=ToolSpec(name="read_file", description="r", source="test"),
        executor=lambda call: ToolResult(content="ok"),
    ))
    return r


def test_spawn_subagent_tool_runs_and_returns_result():
    provider = _ScriptedProvider([_make_chat_response("done")])
    tool = build_spawn_subagent_tool(
        provider=provider, settings=_settings(),
        tool_registry=_registry(), tracer=NoopTracer(),
    )
    assert tool.spec.name == "spawn_subagent"
    assert tool.spec.metadata.get("worker_visible") is False

    call = ToolCall(
        name="spawn_subagent",
        arguments={"role": "researcher", "task": "find X"},
        context={"current_fleet_id": "fleet-test", "thread_id": "t1", "channel": "cli"},
    )
    result = tool.executor(call)
    assert result.is_error is False
    assert "done" in result.content
    assert result.metadata.get("sub_id", "").startswith("fleet-test-")


def test_spawn_subagent_tool_failure_returns_error():
    provider = _ScriptedProvider([])
    tool = build_spawn_subagent_tool(
        provider=provider, settings=_settings(),
        tool_registry=_registry(), tracer=NoopTracer(),
    )
    call = ToolCall(
        name="spawn_subagent",
        arguments={"role": "researcher", "task": "x"},
        context={"current_fleet_id": "fleet-x"},
    )
    result = tool.executor(call)
    assert result.is_error is True
    # Error message should mention failure (provider exhausted)
    assert "subagent failed" in result.content.lower() or "exhausted" in result.content.lower()


def test_spawn_subagent_tool_validates_role_and_task():
    tool = build_spawn_subagent_tool(
        provider=_ScriptedProvider([]), settings=_settings(),
        tool_registry=_registry(), tracer=NoopTracer(),
    )
    r1 = tool.executor(ToolCall(name="spawn_subagent", arguments={"role": "", "task": "x"}, context={"current_fleet_id": "f"}))
    assert r1.is_error
    r2 = tool.executor(ToolCall(name="spawn_subagent", arguments={"role": "researcher", "task": ""}, context={"current_fleet_id": "f"}))
    assert r2.is_error


def test_spawn_subagent_tool_custom_role_requires_tools():
    tool = build_spawn_subagent_tool(
        provider=_ScriptedProvider([]), settings=_settings(),
        tool_registry=_registry(), tracer=NoopTracer(),
    )
    # Custom role without tools list → error
    result = tool.executor(ToolCall(
        name="spawn_subagent",
        arguments={"role": "data_scientist", "task": "analyze"},
        context={"current_fleet_id": "f"},
    ))
    assert result.is_error
    assert "tools" in result.content.lower()
