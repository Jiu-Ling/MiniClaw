from unittest.mock import MagicMock

from miniclaw.runtime.nodes import make_agent
from miniclaw.runtime.state import ActiveCapabilities
from miniclaw.tools.contracts import ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool, ToolRegistry

from tests.test_subagent import _make_chat_response, _ScriptedProvider


class _CapturingTracer:
    """A test tracer that records every span/event call for assertion."""
    def __init__(self):
        self.spans = []  # list of (kind, name, ...) tuples
        self.contexts = []  # list of TraceContext objects

    def start_run(self, *, name, thread_id=None, channel=None, metadata=None, context=None):
        from miniclaw.observability.contracts import build_run_context
        ctx = context or build_run_context(name=name, thread_id=thread_id, channel=channel, metadata=metadata)
        self.spans.append(("run_start", name))
        self.contexts.append(ctx)
        return ctx

    def finish_run(self, ctx, *, status, output=None, metadata=None):
        self.spans.append(("run_finish", ctx.name, status))

    def start_span(self, parent, *, name, metadata=None, context=None, inputs=None, run_type=None):
        from miniclaw.observability.contracts import build_span_context
        ctx = context or build_span_context(parent, name=name, metadata=metadata)
        self.spans.append(("span_start", name))
        self.contexts.append(ctx)
        return ctx

    def finish_span(self, ctx, *, status, output=None, metadata=None, outputs=None):
        self.spans.append(("span_finish", ctx.name, status))

    def record_event(self, ctx, *, name, payload=None, metadata=None, status=None):
        self.spans.append(("event", name))


def _settings():
    s = MagicMock()
    s.model = "fake"
    s.max_tool_rounds = 4
    s.max_consecutive_tool_errors = 4
    s.max_tool_result_chars = 16_000
    s.system_prompt = "you are an agent"
    s.history_char_budget = 10000
    s.max_history_messages = 50
    return s


def test_make_agent_mints_fleet_id_when_spawn_present():
    """When the main agent emits >=1 spawn_subagent call in one round,
    make_agent must mint a fleet_id and write it into runtime_metadata
    so the spawn tool sees it via call.context."""
    captured_contexts = []

    def fake_executor(call):
        captured_contexts.append(dict(call.context))
        return ToolResult(content="sub done", metadata={"sub_id": "x"})

    registry = ToolRegistry(skill_loader=None, mcp_registry=None)
    registry.register(RegisteredTool(
        spec=ToolSpec(
            name="spawn_subagent",
            description="spawn",
            source="builtin",
            metadata={"worker_visible": False},
        ),
        executor=fake_executor,
    ))

    provider = _ScriptedProvider([
        _make_chat_response(
            "",
            tool_calls=[
                {"id": "c1", "type": "function",
                 "function": {"name": "spawn_subagent",
                              "arguments": '{"role":"researcher","task":"a"}'}},
                {"id": "c2", "type": "function",
                 "function": {"name": "spawn_subagent",
                              "arguments": '{"role":"researcher","task":"b"}'}},
            ],
        ),
        _make_chat_response("all done"),
    ])
    agent = make_agent(settings=_settings(), provider=provider, tool_registry=registry)
    state = {"messages": [], "user_input": "go", "active_capabilities": ActiveCapabilities()}
    result = agent(state)

    assert "all done" in result.get("response_text", ""), f"got {result}"
    assert len(captured_contexts) == 2, f"got {len(captured_contexts)} contexts"
    fleet_ids = {ctx.get("current_fleet_id") for ctx in captured_contexts}
    assert len(fleet_ids) == 1, f"all spawns in one round share one fleet_id; got {fleet_ids}"
    assert all(fid for fid in fleet_ids), f"fleet_id must be non-empty; got {fleet_ids}"


def test_make_agent_no_fleet_id_for_normal_tools():
    """For non-spawn tool calls, no fleet_id is minted."""
    captured_contexts = []

    def fake_executor(call):
        captured_contexts.append(dict(call.context))
        return ToolResult(content="ok")

    registry = ToolRegistry(skill_loader=None, mcp_registry=None)
    registry.register(RegisteredTool(
        spec=ToolSpec(name="read_file", description="r", source="test"),
        executor=fake_executor,
    ))

    provider = _ScriptedProvider([
        _make_chat_response("", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"x"}'},
        }]),
        _make_chat_response("finished"),
    ])
    agent = make_agent(settings=_settings(), provider=provider, tool_registry=registry)
    state = {"messages": [], "user_input": "go", "active_capabilities": ActiveCapabilities()}
    agent(state)

    # The read_file context should not have a fleet_id
    assert len(captured_contexts) == 1
    assert not captured_contexts[0].get("current_fleet_id"), \
        f"non-spawn rounds must not mint a fleet_id; got {captured_contexts[0]}"


def test_make_agent_forwards_subagent_events_to_on_event():
    """When make_agent's on_event is set and a spawn_subagent tool emits
    fleet events via its on_event arg, the events must reach the channel."""
    captured_events = []

    def on_event(evt):
        captured_events.append(dict(evt))

    def fake_spawn_executor(call):
        # Simulate dispatcher emitting events
        cb = call.context.get("_on_event")
        if cb:
            cb({"kind": "subagent_dispatched",
                "fleet_id": call.context.get("current_fleet_id"),
                "sub_id": "x-1", "role": "researcher", "task_summary": "..."})
            cb({"kind": "subagent_started", "sub_id": "x-1"})
            cb({"kind": "subagent_completed", "sub_id": "x-1", "status": "completed",
                "result_summary": "ok", "error": "", "rounds_used": 1})
        return ToolResult(content="done")

    registry = ToolRegistry(skill_loader=None, mcp_registry=None)
    registry.register(RegisteredTool(
        spec=ToolSpec(name="spawn_subagent", description="", source="builtin",
                      metadata={"worker_visible": False}),
        executor=fake_spawn_executor,
    ))
    from tests.test_subagent import _make_chat_response, _ScriptedProvider
    provider = _ScriptedProvider([
        _make_chat_response("", tool_calls=[
            {"id": "c1", "type": "function",
             "function": {"name": "spawn_subagent",
                          "arguments": '{"role":"researcher","task":"a"}'}},
        ]),
        _make_chat_response("finish"),
    ])
    agent = make_agent(settings=_settings(), provider=provider, tool_registry=registry, on_event=on_event)
    agent({"messages": [], "user_input": "go", "active_capabilities": ActiveCapabilities()})

    # Filter to only subagent fleet events (the agent loop emits other event kinds too)
    subagent_kinds = [e["kind"] for e in captured_events
                      if e.get("kind", "").startswith("subagent_")]
    assert "subagent_dispatched" in subagent_kinds, f"got kinds: {subagent_kinds}"
    assert "subagent_started" in subagent_kinds
    assert "subagent_completed" in subagent_kinds


def test_make_agent_emits_provider_and_round_spans():
    """make_agent must open agent.tool_loop.round_<n> and provider.achat
    spans for each tool-loop round."""
    tracer = _CapturingTracer()
    provider = _ScriptedProvider([
        _make_chat_response("ok"),
    ])
    registry = ToolRegistry(skill_loader=None, mcp_registry=None)
    agent = make_agent(settings=_settings(), provider=provider, tool_registry=registry, tracer=tracer)
    agent({"messages": [], "user_input": "hi", "active_capabilities": ActiveCapabilities()})

    span_names = [s[1] for s in tracer.spans if s[0] == "span_start"]
    assert "agent.tool_loop.round_0" in span_names, f"got: {span_names}"
    assert "provider.achat" in span_names, f"got: {span_names}"


def test_make_agent_closes_round_span_on_apply_tool_calls_exception(monkeypatch):
    """When apply_tool_calls raises, round_span must still be closed."""
    tracer = _CapturingTracer()

    from miniclaw.runtime import nodes
    def boom(*args, **kwargs):
        raise RuntimeError("apply boom")
    monkeypatch.setattr(nodes, "apply_tool_calls", boom)

    registry = ToolRegistry(skill_loader=None, mcp_registry=None)
    registry.register(RegisteredTool(
        spec=ToolSpec(name="read_file", description="", source="test"),
        executor=lambda call: ToolResult(content="ok"),
    ))
    from tests.test_subagent import _make_chat_response, _ScriptedProvider
    provider = _ScriptedProvider([
        _make_chat_response("", tool_calls=[
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"x"}'}},
        ]),
    ])
    agent = make_agent(settings=_settings(), provider=provider, tool_registry=registry, tracer=tracer)
    agent({"messages": [], "user_input": "go", "active_capabilities": ActiveCapabilities()})

    starts = [s for s in tracer.spans if s[0] == "span_start" and s[1].startswith("agent.tool_loop.round_")]
    finishes = [s for s in tracer.spans if s[0] == "span_finish" and s[1].startswith("agent.tool_loop.round_")]
    assert len(starts) == len(finishes) == 1, \
        f"round_span leaked: {len(starts)} starts, {len(finishes)} finishes"


def test_make_agent_propagates_parent_trace_to_spawn_calls():
    """When the agent spawns subagents, the round's span context must be
    written into runtime_metadata['_parent_trace'] so subagents hang under it."""
    tracer = _CapturingTracer()
    captured_parent_traces = []

    def fake_spawn_executor(call):
        captured_parent_traces.append(call.context.get("_parent_trace"))
        return ToolResult(content="ok")

    registry = ToolRegistry(skill_loader=None, mcp_registry=None)
    registry.register(RegisteredTool(
        spec=ToolSpec(name="spawn_subagent", description="", source="builtin",
                      metadata={"worker_visible": False}),
        executor=fake_spawn_executor,
    ))
    provider = _ScriptedProvider([
        _make_chat_response("", tool_calls=[
            {"id": "c1", "type": "function",
             "function": {"name": "spawn_subagent",
                          "arguments": '{"role":"researcher","task":"a"}'}},
        ]),
        _make_chat_response("done"),
    ])
    agent = make_agent(settings=_settings(), provider=provider, tool_registry=registry, tracer=tracer)
    agent({"messages": [], "user_input": "go", "active_capabilities": ActiveCapabilities()})

    assert len(captured_parent_traces) == 1
    assert captured_parent_traces[0] is not None, "_parent_trace must be set"
    # The captured parent must be a TraceContext with name agent.tool_loop.round_0
    assert getattr(captured_parent_traces[0], "name", "") == "agent.tool_loop.round_0"
