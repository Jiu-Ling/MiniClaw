from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from miniclaw.bootstrap import build_runtime_service
from miniclaw.config.settings import Settings
from miniclaw.memory.files import MemoryFileStore
from miniclaw.persistence.memory_store import SQLiteMemoryStore
from miniclaw.providers.contracts import ChatMessage, ChatResponse, ChatUsage
from miniclaw.runtime.service import RuntimeService


class EchoProvider:
    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        import json as _json

        # Handle submit_plan tool call from planner node
        if tools and any(
            t.get("function", {}).get("name") == "submit_plan" for t in tools
        ):
            return ChatResponse(
                content="",
                provider="fake",
                model=model,
                usage=ChatUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                tool_calls=[
                    {
                        "id": "plan_call_1",
                        "function": {
                            "name": "submit_plan",
                            "arguments": _json.dumps({
                                "summary": "Execute the planned request",
                                "tasks": [
                                    {"title": "Execute", "kind": "execution", "worker_role": "executor", "parallel_group": "execute"},
                                ],
                                "executor_notes": "",
                            }),
                        },
                    }
                ],
            )

        latest_user_content = ""
        for message in reversed(messages):
            if getattr(message, "role", "") != "user":
                continue
            latest_user_content = _extract_user_text(message)
            break
        return ChatResponse(
            content=f"echo:{latest_user_content}",
            provider="fake",
            model=model,
            usage=ChatUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )


class FailingMemoryFileStore(MemoryFileStore):
    def update(
        self,
        *,
        long_term_facts: list[str],
        recent_work: dict[str, list[str]],
    ) -> None:
        del long_term_facts, recent_work
        raise OSError("disk full")


def _build_runtime(
    tmp_path: Path,
    *,
    memory_file_store: MemoryFileStore | None = None,
) -> tuple[RuntimeService, SQLiteMemoryStore]:
    sqlite_path = tmp_path / "runtime.sqlite3"
    settings = Settings(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
        sqlite_path=sqlite_path,
    )
    memory_store = SQLiteMemoryStore(sqlite_path)
    memory_store.initialize()
    runtime = RuntimeService(
        settings=settings,
        provider=EchoProvider(),
        memory_store=memory_store,
        memory_file_store=memory_file_store,
        clock=lambda: datetime(2026, 3, 28, 20, 10, tzinfo=timezone.utc),
    )
    return runtime, memory_store


def test_runtime_service_records_thread_digest_after_turn(tmp_path: Path) -> None:
    runtime, memory_store = _build_runtime(tmp_path)

    result = runtime.run_turn(thread_id="thread-1", user_input="Plan the memory layer")

    assert result.response_text == "echo:Plan the memory layer"
    thread_items = memory_store.list_recent("thread-1", limit=10)
    summaries = [item for item in thread_items if item.kind == "thread_summary"]
    assert len(summaries) == 1
    assert summaries[0].content == (
        "2026-03-28 20:10: User asked to plan the memory layer; outcome: echo:Plan the memory layer"
    )


def test_runtime_service_consolidates_recent_work_after_three_summaries(tmp_path: Path) -> None:
    memory_file_store = MemoryFileStore(tmp_path / "MEMORY.md")
    runtime, _ = _build_runtime(tmp_path, memory_file_store=memory_file_store)

    runtime.run_turn(thread_id="thread-1", user_input="First task")
    runtime.run_turn(thread_id="thread-1", user_input="Second task")
    runtime.run_turn(thread_id="thread-1", user_input="Third task")

    document = memory_file_store.read()

    assert document.recent_work["thread:thread-1"] == [
        "2026-03-28 20:10: User asked to first task; outcome: echo:First task",
        "2026-03-28 20:10: User asked to second task; outcome: echo:Second task",
        "2026-03-28 20:10: User asked to third task; outcome: echo:Third task",
    ]


def test_runtime_service_promotes_high_importance_fact_into_memory_file(tmp_path: Path) -> None:
    memory_file_store = MemoryFileStore(tmp_path / "MEMORY.md")
    runtime, memory_store = _build_runtime(tmp_path, memory_file_store=memory_file_store)

    runtime.run_turn(
        thread_id="thread-1",
        user_input="Please remember that we use uv for Python environments.",
    )

    document = memory_file_store.read()
    facts = [item for item in memory_store.list_recent("thread-1", limit=10) if item.kind == "project"]

    assert "Use uv for Python environments." in document.long_term_facts
    assert len(facts) == 1
    assert facts[0].metadata["importance"] == "high"


def test_runtime_service_ignores_memory_file_write_failures(tmp_path: Path) -> None:
    runtime, memory_store = _build_runtime(
        tmp_path,
        memory_file_store=FailingMemoryFileStore(tmp_path / "MEMORY.md"),
    )

    result = runtime.run_turn(
        thread_id="thread-1",
        user_input="Please remember that we only support OpenAI-compatible APIs.",
    )

    assert result.response_text == "echo:Please remember that we only support OpenAI-compatible APIs."
    items = memory_store.list_recent("thread-1", limit=10)
    assert any(item.kind == "thread_summary" for item in items)
    assert any(item.kind == "project" for item in items)


def test_build_runtime_service_wires_memory_file_store(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "runtime.sqlite3"
    settings = Settings(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
        sqlite_path=sqlite_path,
    )
    memory_store = SQLiteMemoryStore(sqlite_path)
    memory_store.initialize()

    runtime = build_runtime_service(
        settings,
        provider=EchoProvider(),
        memory_store=memory_store,
    )

    assert runtime.memory_file_store is not None
    assert runtime.memory_file_store.path == tmp_path / "MEMORY.md"


def _extract_user_text(message: ChatMessage) -> str:
    if message.content:
        return _strip_runtime_metadata(str(message.content))
    for part in message.content_parts:
        if part.get("type") == "text":
            return _strip_runtime_metadata(str(part.get("text", "")))
    return ""


def _strip_runtime_metadata(value: str) -> str:
    metadata_marker = "\n\n## Runtime Metadata"
    if metadata_marker in value:
        return value.split(metadata_marker, 1)[0]
    return value
