from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from miniclaw.persistence.memory_store import SQLiteMemoryStore
from miniclaw.prompting import ContextBuilder
from miniclaw.runtime.nodes import make_load_context, make_planner

# Reuse the test helper from test_subagent for ChatResponse construction:
from tests.test_subagent import _make_chat_response, _ScriptedProvider


# ---------------------------------------------------------------------------
# make_load_context tests (preserved — not about planner output shape)
# ---------------------------------------------------------------------------


def test_load_context_sets_planner_context_for_planned_route(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "runtime.sqlite3")
    store.initialize()
    load_context = make_load_context(store)

    result = load_context(
        {
            "thread_id": "thread-1",
            "route": "planned",
            "user_input": "Build a planner node",
        }
    )

    assert "Build a planner node" in result["planner_context"]


def test_load_context_skips_planner_context_for_simple_route(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "runtime.sqlite3")
    store.initialize()
    load_context = make_load_context(store)

    result = load_context(
        {
            "thread_id": "thread-2",
            "route": "simple",
            "user_input": "Read the README file",
        }
    )

    assert result["planner_context"] == ""


def test_context_builder_includes_planner_sections_when_present(tmp_path: Path) -> None:
    builder = ContextBuilder(workspace=tmp_path, system_prompt="You are MiniClaw.")

    prompt = builder.build_system_prompt(
        {
            "memory_context": "Relevant memory:\n- [fact] user prefers terse answers",
            "planner_context": "Planner context:\nUser request: Build a planner node",
            "plan_summary": "add request-aware context loading",
            "subagent_briefs": [
                {"role": "researcher", "task": "Load request-specific context", "expected_output": "context"},
                {"role": "executor", "task": "Inject planner fields into the prompt", "depends_on": [0]},
            ],
            "executor_notes": "Keep execution single-agent for now.",
        }
    )

    assert "## Planner Context" in prompt
    assert "Planner context:\nUser request: Build a planner node" in prompt
    assert "## Recommended Plan" in prompt
    assert "add request-aware context loading" in prompt
    assert "researcher" in prompt
    assert "Load request-specific context" in prompt
    assert "Inject planner fields into the prompt" in prompt
    assert "Keep execution single-agent for now." in prompt


# ---------------------------------------------------------------------------
# validate / error_handler tests (preserved — not about planner output shape)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# New planner tests — target the new subagent_briefs schema
# ---------------------------------------------------------------------------


def test_planner_returns_briefs():
    provider = _ScriptedProvider([_make_chat_response(
        content="",
        tool_calls=[{
            "id": "c1", "type": "function",
            "function": {
                "name": "submit_plan",
                "arguments": (
                    '{"summary":"investigate then patch",'
                    ' "subagent_briefs":['
                    '   {"role":"researcher","task":"find issue"},'
                    '   {"role":"executor","task":"apply fix","depends_on":[0]}'
                    ' ],'
                    ' "executor_notes":"start with researcher"}'
                ),
            },
        }],
    )])
    planner = make_planner(provider=provider, tool_registry=None)
    state = {"route": "planned", "user_input": "fix auth bug", "memory_context": ""}
    result = planner(state)
    assert result["plan_summary"] == "investigate then patch"
    assert len(result["subagent_briefs"]) == 2
    assert result["subagent_briefs"][0]["role"] == "researcher"
    assert result["executor_notes"] == "start with researcher"
    assert "last_error" not in result


def test_planner_empty_briefs_is_valid():
    provider = _ScriptedProvider([_make_chat_response(
        content="",
        tool_calls=[{
            "id": "c1", "type": "function",
            "function": {
                "name": "submit_plan",
                "arguments": '{"summary":"direct read","subagent_briefs":[]}',
            },
        }],
    )])
    planner = make_planner(provider=provider, tool_registry=None)
    result = planner({"route": "planned", "user_input": "read file", "memory_context": ""})
    assert result["subagent_briefs"] == []
    assert result["plan_summary"] == "direct read"


def test_planner_skips_when_route_not_planned():
    planner = make_planner(provider=MagicMock(), tool_registry=None)
    result = planner({"route": "simple"})
    assert result == {}


def test_planner_error_sets_last_error():
    class _Boom:
        async def achat(self, *a, **k):
            raise RuntimeError("nope")

    planner = make_planner(provider=_Boom(), tool_registry=None)
    result = planner({"route": "planned", "user_input": "x", "memory_context": ""})
    assert "last_error" in result
    assert "planner failed" in result["last_error"]


def test_planner_validates_custom_role_requires_tools():
    provider = _ScriptedProvider([_make_chat_response(
        content="",
        tool_calls=[{
            "id": "c1", "type": "function",
            "function": {
                "name": "submit_plan",
                "arguments": (
                    '{"summary":"x","subagent_briefs":['
                    '  {"role":"data_scientist","task":"analyze"}'
                    ']}'
                ),
            },
        }],
    )])
    planner = make_planner(provider=provider, tool_registry=None)
    result = planner({"route": "planned", "user_input": "x", "memory_context": ""})
    assert "last_error" in result
    assert "tools" in result["last_error"].lower()
