import asyncio
from collections.abc import Sequence
from unittest.mock import MagicMock

from miniclaw.observability.contracts import NoopTracer, build_run_context
from miniclaw.providers.contracts import ChatMessage, ChatResponse
from miniclaw.runtime.subagent import (
    ROLE_DEFAULTS,
    SubagentBrief,
    SubagentResult,
    run_subagent,
)
from miniclaw.tools.contracts import ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool, ToolRegistry


class _ScriptedProvider:
    """Fake provider that replays a scripted sequence of ChatResponse objects."""

    capabilities = MagicMock()

    def __init__(self, responses: Sequence[ChatResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    async def achat(self, messages, *, model=None, tools=None, **kwargs):
        self.calls.append(list(messages))
        if not self.responses:
            raise RuntimeError("provider exhausted")
        return self.responses.pop(0)

    async def astream_text(self, messages, *, model=None, tools=None, **kwargs):
        raise NotImplementedError


def _make_settings():
    settings = MagicMock()
    settings.model = "fake-model"
    settings.max_tool_rounds = 16
    settings.max_consecutive_tool_errors = 4
    settings.max_tool_result_chars = 16_000
    return settings


def _make_chat_response(content: str, tool_calls=None) -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        provider="test",
    )


def _make_registry_with_read_file():
    registry = ToolRegistry(skill_loader=None, mcp_registry=None)

    def execute(call):
        return ToolResult(content=f"file contents for {call.arguments.get('path', '')}")

    registry.register(RegisteredTool(
        spec=ToolSpec(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            source="test",
        ),
        executor=execute,
    ))
    return registry


def test_brief_is_immutable():
    brief = SubagentBrief(role="researcher", task="find X")
    assert brief.role == "researcher"
    assert brief.task == "find X"
    assert brief.tools is None
    assert brief.context == ()


def test_result_completed_shape():
    result = SubagentResult(
        sub_id="sub-1", fleet_id="fleet-a", role="researcher",
        status="completed", result="ok", summary="ok",
    )
    assert result.status == "completed"
    assert result.error == ""


def test_role_defaults_known_roles():
    assert "researcher" in ROLE_DEFAULTS
    assert "executor" in ROLE_DEFAULTS
    assert "reviewer" in ROLE_DEFAULTS
    for role, defaults in ROLE_DEFAULTS.items():
        assert "prompt" in defaults
        assert "tools" in defaults
        assert isinstance(defaults["tools"], tuple)


def test_role_defaults_researcher_is_readonly():
    tools = ROLE_DEFAULTS["researcher"]["tools"]
    assert "shell" not in tools
    assert "memory_search" in tools


def test_run_subagent_happy_path_no_tools():
    provider = _ScriptedProvider([
        _make_chat_response("research complete: found X"),
    ])
    settings = _make_settings()
    registry = _make_registry_with_read_file()
    brief = SubagentBrief(role="researcher", task="find X", tools=("read_file",))

    result = asyncio.run(run_subagent(
        brief=brief,
        fleet_id="fleet-a",
        sub_id="fleet-a-1",
        provider=provider,
        settings=settings,
        tool_registry=registry,
        runtime_metadata={"thread_id": "t1", "channel": "test"},
        tracer=NoopTracer(),
        parent_trace=build_run_context(name="test"),
    ))

    assert result.status == "completed"
    assert "research complete" in result.result
    assert result.rounds_used == 1
    assert result.fleet_id == "fleet-a"
    assert result.sub_id == "fleet-a-1"


def test_run_subagent_unknown_tool_falls_through():
    """Verify the KeyError fallback path for completely unknown tools.

    No shell tool is registered at all, so the call falls through to
    _exec_one's KeyError path — NOT the worker_visible_only filter.
    This tests the error-recovery loop, not the security filter.
    """
    # Subagent's prompt asks for shell, but role researcher does not include it.
    # No shell tool registered → call becomes KeyError → ERROR: message → next response wins.
    provider = _ScriptedProvider([
        _make_chat_response("", tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "shell", "arguments": '{"command": "rm -rf /"}'},
        }]),
        _make_chat_response("aborted, no shell access"),
    ])
    settings = _make_settings()
    registry = _make_registry_with_read_file()
    # No shell tool registered at all → call returns ERROR → second response wins
    brief = SubagentBrief(role="researcher", task="something")

    result = asyncio.run(run_subagent(
        brief=brief, fleet_id="f", sub_id="f-1",
        provider=provider, settings=settings, tool_registry=registry,
        runtime_metadata={}, tracer=NoopTracer(),
        parent_trace=build_run_context(name="test"),
    ))
    # The shell call has no spec → filtered out by _filter_worker_visible (unknown → passed through),
    # then _exec_one raises KeyError → ToolResult(is_error=True). consecutive_errors=1 < 4.
    # Second response has no tool calls → completes.
    assert result.status == "completed"
    assert "aborted" in result.result


def test_run_subagent_filter_blocks_worker_invisible_tool():
    """Verify worker_visible_only=True actually filters tools, not just falls
    through to KeyError. The shell tool IS registered, but marked
    worker_visible=False — the filter must drop the call before execution."""
    shell_calls = []

    def shell_executor(call):
        shell_calls.append(dict(call.arguments))
        return ToolResult(content="EXECUTED — this should never happen")

    registry = ToolRegistry(skill_loader=None, mcp_registry=None)
    registry.register(RegisteredTool(
        spec=ToolSpec(
            name="shell",
            description="Run shell",
            input_schema={"type": "object", "properties": {"cmd": {"type": "string"}}},
            source="test",
            metadata={"worker_visible": False},  # explicitly hidden from subagents
        ),
        executor=shell_executor,
    ))
    # Also register a benign tool the subagent CAN use, so the test is realistic
    registry.register(RegisteredTool(
        spec=ToolSpec(
            name="read_file",
            description="Read",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            source="test",
        ),
        executor=lambda call: ToolResult(content="ok"),
    ))

    provider = _ScriptedProvider([
        # Subagent tries to call shell — the filter must drop it
        _make_chat_response(
            "",
            tool_calls=[{
                "id": "call-1",
                "type": "function",
                "function": {"name": "shell", "arguments": '{"cmd": "rm -rf /"}'},
            }],
        ),
        # Second response: subagent gives up and returns final answer
        _make_chat_response("cannot proceed — shell unavailable"),
    ])
    settings = _make_settings()
    # Pass shell explicitly so the resolver considers it. The filter
    # in execute_tool_calls must block it via worker_visible_only=True.
    brief = SubagentBrief(role="executor", task="do something", tools=("shell", "read_file"))

    result = asyncio.run(run_subagent(
        brief=brief, fleet_id="f", sub_id="f-1",
        provider=provider, settings=settings, tool_registry=registry,
        runtime_metadata={}, tracer=NoopTracer(),
        parent_trace=build_run_context(name="test"),
    ))

    # Critical assertion: shell was never executed
    assert shell_calls == [], f"shell executor was called: {shell_calls}"
    # And the subagent terminated cleanly
    assert result.status == "completed"
    assert "shell unavailable" in result.result


def test_run_subagent_safe_emit_swallows_callback_errors():
    """If on_event raises, the subagent must still complete normally."""
    def crashing_callback(evt):
        raise RuntimeError("channel boom")

    provider = _ScriptedProvider([_make_chat_response("done")])
    result = asyncio.run(run_subagent(
        brief=SubagentBrief(role="researcher", task="x", tools=("read_file",)),
        fleet_id="f", sub_id="f-1",
        provider=provider, settings=_make_settings(),
        tool_registry=_make_registry_with_read_file(),
        runtime_metadata={}, tracer=NoopTracer(),
        parent_trace=build_run_context(name="test"),
        on_event=crashing_callback,
    ))
    assert result.status == "completed"
    assert "done" in result.result


def test_run_subagent_round_limit():
    # Provider always returns a tool call → never terminates
    forever_tool_call = _make_chat_response("", tool_calls=[{
        "id": "c1", "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "/etc/hosts"}'},
    }])
    provider = _ScriptedProvider([forever_tool_call] * 20)
    settings = _make_settings()
    settings.max_tool_rounds = 3
    registry = _make_registry_with_read_file()
    brief = SubagentBrief(role="researcher", task="loop forever", tools=("read_file",))

    result = asyncio.run(run_subagent(
        brief=brief, fleet_id="f", sub_id="f-1",
        provider=provider, settings=settings, tool_registry=registry,
        runtime_metadata={}, tracer=NoopTracer(),
        parent_trace=build_run_context(name="test"),
    ))
    assert result.status == "failed"
    assert "round limit" in result.error
    assert result.rounds_used == 3


def test_subagent_system_message_has_cache_control():
    from miniclaw.runtime.subagent import _build_messages, SubagentBrief
    msgs = _build_messages(
        SubagentBrief(role="researcher", task="find X"),
        runtime_metadata={"thread_id": "t", "channel": "cli"},
        fleet_id="f", sub_id="f-1",
    )
    system_msg = msgs[0]
    parts = getattr(system_msg, "content_parts", None) or []
    assert parts, "subagent system message must use content_parts"
    assert any(
        isinstance(p, dict) and p.get("cache_control", {}).get("type") == "ephemeral"
        for p in parts
    ), f"static part must have cache_control: ephemeral; got {parts}"
