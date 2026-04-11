from __future__ import annotations
import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from miniclaw.config.settings import Settings
from miniclaw.persistence.memory_store import SQLiteMemoryStore
from miniclaw.providers.contracts import ChatMessage, ChatResponse, ChatUsage
from miniclaw.prompting import ContextBuilder
from miniclaw.runtime.service import RuntimeService


def _system_text(msg: ChatMessage) -> str:
    """Extract the full text of a system message, whether content or content_parts."""
    if msg.content is not None:
        return msg.content
    return "\n\n".join(
        part.get("text", "") for part in msg.content_parts if isinstance(part, dict)
    )

class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        self.calls.append(list(messages))
        # If the classify_intent tool is present, classify intent based on user message
        if tools and any(
            t.get("function", {}).get("name") == "classify_intent"
            for t in tools
            if isinstance(t, dict)
        ):
            # Check for planning markers in the user message to route planned requests
            user_text = ""
            for msg in messages:
                if msg.role == "user":
                    user_text = str(msg.content or "").lower()
                    break
            _planned_markers = ("implement", "build", "refactor", "fix", "plan", "design", "update tests", "migrate")
            intent = "planned" if any(m in user_text for m in _planned_markers) else "simple"
            return ChatResponse(
                content="",
                provider="fake",
                model=model,
                tool_calls=[
                    {
                        "id": "call_classify",
                        "function": {
                            "name": "classify_intent",
                            "arguments": json.dumps({"intent": intent}),
                        },
                    }
                ],
            )
        # If the submit_plan tool is present, return a plan tool call response
        if tools and any(
            t.get("function", {}).get("name") == "submit_plan"
            for t in tools
            if isinstance(t, dict)
        ):
            return ChatResponse(
                content="",
                provider="fake",
                model=model,
                tool_calls=[
                    {
                        "id": "call_plan",
                        "function": {
                            "name": "submit_plan",
                            "arguments": json.dumps(_FAKE_PLAN),
                        },
                    }
                ],
            )
        return ChatResponse(
            content="MiniClaw runtime reply",
            provider="fake",
            model=model,
            usage=ChatUsage(prompt_tokens=7, completion_tokens=5, total_tokens=12),
        )


def test_runtime_service_runs_minimal_graph_with_memory_context(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite3"
    settings = Settings(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
        sqlite_path=sqlite_path,
        system_prompt="You are MiniClaw.",
    )
    memory_store = SQLiteMemoryStore(sqlite_path)
    memory_store.initialize()
    memory_store.append_fact(
        "thread-1",
        "user prefers terse answers",
        "fact",
        {"source": "test"},
    )
    provider = FakeProvider()
    service = RuntimeService(
        settings=settings,
        provider=provider,
        memory_store=memory_store,
    )

    result = service.run_turn(thread_id="thread-1", user_input="Say hello")

    assert result.thread_id == "thread-1"
    assert result.response_text == "MiniClaw runtime reply"
    assert result.last_error == ""
    assert result.usage == {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    assert result.checkpoint_id is not None

    # classify node + agent node = 2 calls
    assert len(provider.calls) == 2
    messages = provider.calls[1]
    assert len(messages) >= 2
    assert messages[0].role == "system"
    assert "You are MiniClaw." in _system_text(messages[0])
    user_msg = messages[1]
    assert user_msg.role == "user"
    assert "Say hello" in user_msg.content
    assert "thread_id: thread-1" in user_msg.content
    assert "clock:" in user_msg.content

    with sqlite3.connect(sqlite_path) as conn:
        checkpoint_count = conn.execute(
            "SELECT COUNT(*) FROM langgraph_checkpoints WHERE thread_id = ?",
            ("thread-1",),
        ).fetchone()[0]

    assert checkpoint_count >= 1


class FailingProvider:
    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        raise RuntimeError("provider failed")


def test_runtime_service_captures_provider_error_in_state(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite3"
    settings = Settings(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
        sqlite_path=sqlite_path,
    )
    memory_store = SQLiteMemoryStore(sqlite_path)
    memory_store.initialize()
    service = RuntimeService(
        settings=settings,
        provider=FailingProvider(),
        memory_store=memory_store,
    )

    result = service.run_turn(thread_id="thread-err", user_input="Say hello")

    assert result.thread_id == "thread-err"
    # T12: errors now routed through error_handler which fills response_text with a user-facing message
    assert "provider failed" in result.response_text
    assert result.last_error == "provider failed"
    assert result.usage == {}
    assert result.checkpoint_id is not None


def test_runtime_service_reuses_checkpointed_message_history(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite3"
    settings = Settings(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
        sqlite_path=sqlite_path,
        system_prompt="You are MiniClaw.",
    )
    memory_store = SQLiteMemoryStore(sqlite_path)
    memory_store.initialize()
    provider = FakeProvider()
    service = RuntimeService(
        settings=settings,
        provider=provider,
        memory_store=memory_store,
    )

    first = service.run_turn(thread_id="thread-2", user_input="hello")
    second = service.run_turn(thread_id="thread-2", user_input="again")

    assert first.response_text == "MiniClaw runtime reply"
    assert second.response_text == "MiniClaw runtime reply"
    # Each turn: classify + agent = 2 calls; 2 turns = 4 total
    assert len(provider.calls) == 4
    second_messages = provider.calls[3]
    assert [message.role for message in second_messages] == ["system", "user", "assistant", "user"]
    assert second_messages[1].content == "hello"
    assert second_messages[2].content == "MiniClaw runtime reply"
    assert "again" in second_messages[3].content
    assert "thread_id: thread-2" in second_messages[3].content
    assert "clock:" in second_messages[3].content


def test_runtime_service_runs_inside_async_context(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite3"
    settings = Settings(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
        sqlite_path=sqlite_path,
    )
    memory_store = SQLiteMemoryStore(sqlite_path)
    memory_store.initialize()
    service = RuntimeService(
        settings=settings,
        provider=FakeProvider(),
        memory_store=memory_store,
    )

    async def run() -> object:
        return service.run_turn(thread_id="thread-async", user_input="hello")

    result = asyncio.run(run())

    assert result.response_text == "MiniClaw runtime reply"
    assert result.last_error == ""


def test_runtime_service_does_not_replay_failed_turn_message(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite3"
    settings = Settings(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
        sqlite_path=sqlite_path,
    )
    memory_store = SQLiteMemoryStore(sqlite_path)
    memory_store.initialize()
    failing_service = RuntimeService(
        settings=settings,
        provider=FailingProvider(),
        memory_store=memory_store,
    )
    failing_result = failing_service.run_turn(thread_id="thread-fail", user_input="first try")

    assert failing_result.last_error == "provider failed"

    provider = FakeProvider()
    succeeding_service = RuntimeService(
        settings=settings,
        provider=provider,
        memory_store=memory_store,
    )
    succeeding_result = succeeding_service.run_turn(thread_id="thread-fail", user_input="second try")

    assert succeeding_result.response_text == "MiniClaw runtime reply"
    # classify + agent = 2 calls
    assert len(provider.calls) == 2
    sent_messages = provider.calls[1]
    assert [message.role for message in sent_messages] == ["system", "user"]
    assert "second try" in sent_messages[1].content
    assert "thread_id: thread-fail" in sent_messages[1].content


def test_runtime_service_builds_safe_defaults_for_planner_executor_state(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite3"
    settings = Settings(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
        sqlite_path=sqlite_path,
    )
    memory_store = SQLiteMemoryStore(sqlite_path)
    memory_store.initialize()
    service = RuntimeService(
        settings=settings,
        provider=FakeProvider(),
        memory_store=memory_store,
    )

    initial_state = service._build_initial_state(
        thread_id="thread-safe-defaults",
        user_input="hello",
        runtime_metadata=None,
        user_content_parts=None,
        snapshot_values={},
    )

    assert initial_state["planner_context"] == ""
    assert initial_state["plan_summary"] == ""
    assert initial_state["subagent_briefs"] == []
    assert initial_state["executor_notes"] == ""
    assert initial_state["fleet_runs"] == []


def test_route_on_error_sends_to_error_handler():
    from miniclaw.runtime.nodes import route_on_error
    fn = route_on_error("agent")
    assert fn({"last_error": "boom"}) == "error_handler"
    assert fn({"last_error": ""}) == "agent"
    assert fn({}) == "agent"


def test_graph_has_unified_agent_node_no_executor_validate():
    from miniclaw.runtime.graph import build_graph
    from unittest.mock import MagicMock

    # Build a minimal graph with stub providers
    settings = MagicMock()
    settings.system_prompt = "you are an agent"
    settings.history_char_budget = 10000
    settings.max_history_messages = 50
    settings.max_tool_rounds = 4
    settings.max_consecutive_tool_errors = 4
    settings.max_tool_result_chars = 16000
    settings.model = "fake"

    provider = MagicMock()
    memory_store = MagicMock()
    memory_store.recent_messages.return_value = []

    graph = build_graph(
        settings=settings,
        provider=provider,
        memory_store=memory_store,
        tool_registry=None,
    )

    compiled = graph.compile()
    node_names = set(compiled.get_graph().nodes.keys())
    assert "agent" in node_names, f"missing 'agent' node, got: {node_names}"
    assert "executor" not in node_names, f"'executor' node should be removed, got: {node_names}"
    assert "validate" not in node_names, f"'validate' node should be removed, got: {node_names}"
    assert "planner" in node_names
    assert "error_handler" in node_names
    assert "complete" in node_names
