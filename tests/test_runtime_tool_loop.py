from __future__ import annotations

from pathlib import Path

from miniclaw.bootstrap import build_tool_registry
from miniclaw.config.settings import Settings
from miniclaw.mcp.contracts import MCPServerConfig, MCPToolSpec
from miniclaw.mcp.registry import MCPRegistry
from miniclaw.persistence.memory_store import SQLiteMemoryStore
from miniclaw.providers.contracts import ChatMessage
from miniclaw.runtime.service import RuntimeService
from miniclaw.skills.loader import SkillLoader


def _system_text(msg: ChatMessage) -> str:
    """Extract the full text of a system message, whether content or content_parts."""
    if msg.content is not None:
        return msg.content
    return "\n\n".join(
        part.get("text", "") for part in msg.content_parts if isinstance(part, dict)
    )


class FakeToolCallingProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[list[ChatMessage], list[dict[str, object]]]] = []

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict[str, object]] | None = None,
    ):
        self.calls.append((list(messages), list(tools or [])))
        # Call 1: classify node — return simple intent
        if tools and any(
            t.get("function", {}).get("name") == "classify_intent"
            for t in tools
            if isinstance(t, dict)
        ):
            return type(
                "Response",
                (),
                {
                    "content": "",
                    "content_parts": [],
                    "tool_calls": [
                        {
                            "id": "call_classify",
                            "type": "function",
                            "function": {"name": "classify_intent", "arguments": '{"intent":"simple"}'},
                        }
                    ],
                    "provider": "fake",
                    "model": model,
                    "usage": None,
                    "raw": {},
                },
            )()
        # Call 2 (first executor call): return a read_file tool call
        if len(self.calls) == 2:
            return type(
                "Response",
                (),
                {
                    "content": "",
                    "content_parts": [],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"note.txt"}'},
                        }
                    ],
                    "provider": "fake",
                    "model": model,
                    "usage": None,
                    "raw": {},
                },
            )()
        return type(
            "Response",
            (),
            {
                "content": "final answer",
                "content_parts": [],
                "tool_calls": [],
                "provider": "fake",
                "model": model,
                "usage": None,
                "raw": {},
            },
        )()


class LoopingToolProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict[str, object]] | None = None,
    ):
        del messages
        # Handle classify_intent call without counting it toward the loop cap
        if tools and any(
            t.get("function", {}).get("name") == "classify_intent"
            for t in (tools or [])
            if isinstance(t, dict)
        ):
            return type(
                "Response",
                (),
                {
                    "content": "",
                    "content_parts": [],
                    "tool_calls": [
                        {
                            "id": "call_classify",
                            "type": "function",
                            "function": {"name": "classify_intent", "arguments": '{"intent":"simple"}'},
                        }
                    ],
                    "provider": "fake",
                    "model": model,
                    "usage": None,
                    "raw": {},
                },
            )()
        self.calls += 1
        return type(
            "Response",
            (),
            {
                "content": "",
                "content_parts": [],
                "tool_calls": [
                    {
                        "id": f"call_{self.calls}",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"note.txt"}'},
                    }
                ],
                "provider": "fake",
                "model": model,
                "usage": None,
                "raw": {},
            },
        )()


class FakeActivationProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[list[ChatMessage], list[dict[str, object]]]] = []

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict[str, object]] | None = None,
    ):
        self.calls.append((list(messages), list(tools or [])))
        # Handle classify_intent call
        if tools and any(
            t.get("function", {}).get("name") == "classify_intent"
            for t in tools
            if isinstance(t, dict)
        ):
            return type(
                "Response",
                (),
                {
                    "content": "",
                    "content_parts": [],
                    "tool_calls": [
                        {
                            "id": "call_classify",
                            "type": "function",
                            "function": {"name": "classify_intent", "arguments": '{"intent":"simple"}'},
                        }
                    ],
                    "provider": "fake",
                    "model": model,
                    "usage": None,
                    "raw": {},
                },
            )()
        # First executor call: load_skill_tools
        if len(self.calls) == 2:
            return type(
                "Response",
                (),
                {
                    "content": "",
                    "content_parts": [],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "load_skill_tools", "arguments": '{"skills":["editor"]}'},
                        }
                    ],
                    "provider": "fake",
                    "model": model,
                    "usage": None,
                    "raw": {},
                },
            )()
        return type(
            "Response",
            (),
            {
                "content": "skill loaded",
                "content_parts": [],
                "tool_calls": [],
                "provider": "fake",
                "model": model,
                "usage": None,
                "raw": {},
            },
        )()


class FakeMCPClient:
    def connect(self) -> None:
        return None

    def list_tools(self) -> list[MCPToolSpec]:
        return [
            MCPToolSpec(
                name="search_docs",
                description="Search docs",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                metadata={},
            )
        ]

    def call_tool(self, name: str, arguments: dict[str, object]) -> str:
        return f"{name}:{arguments.get('query', '')}"


class FakeMCPActivationProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[list[ChatMessage], list[dict[str, object]]]] = []

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict[str, object]] | None = None,
    ):
        self.calls.append((list(messages), list(tools or [])))
        # Handle classify_intent call
        if tools and any(
            t.get("function", {}).get("name") == "classify_intent"
            for t in tools
            if isinstance(t, dict)
        ):
            return type(
                "Response",
                (),
                {
                    "content": "",
                    "content_parts": [],
                    "tool_calls": [
                        {
                            "id": "call_classify",
                            "type": "function",
                            "function": {"name": "classify_intent", "arguments": '{"intent":"simple"}'},
                        }
                    ],
                    "provider": "fake",
                    "model": model,
                    "usage": None,
                    "raw": {},
                },
            )()
        # First executor call: load_mcp_tools
        if len(self.calls) == 2:
            return type(
                "Response",
                (),
                {
                    "content": "",
                    "content_parts": [],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "load_mcp_tools", "arguments": '{"server":"docs"}'},
                        }
                    ],
                    "provider": "fake",
                    "model": model,
                    "usage": None,
                    "raw": {},
                },
            )()
        # Second executor call: search_docs
        if len(self.calls) == 3:
            return type(
                "Response",
                (),
                {
                    "content": "",
                    "content_parts": [],
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "search_docs", "arguments": '{"query":"runtime"}'},
                        }
                    ],
                    "provider": "fake",
                    "model": model,
                    "usage": None,
                    "raw": {},
                },
            )()
        return type(
            "Response",
            (),
            {
                "content": "docs loaded",
                "content_parts": [],
                "tool_calls": [],
                "provider": "fake",
                "model": model,
                "usage": None,
                "raw": {},
            },
        )()


def _build_service(
    tmp_path: Path,
    provider: object,
    *,
    mcp_registry: MCPRegistry | None = None,
) -> RuntimeService:
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
    return RuntimeService(
        settings=settings,
        provider=provider,
        memory_store=memory_store,
        tool_registry=build_tool_registry(
            workspace=tmp_path,
            skill_loader=SkillLoader(workspace=tmp_path),
            mcp_registry=mcp_registry,
        ),
    )


def test_runtime_service_completes_one_tool_call_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello from file", encoding="utf-8")
    provider = FakeToolCallingProvider()
    service = _build_service(tmp_path, provider)

    result = service.run_turn(thread_id="tool-thread", user_input="read the note")

    assert result.last_error == ""
    assert result.response_text == "final answer"
    # classify + executor call 1 (read_file tool call) + executor call 2 (final answer) = 3 calls
    assert len(provider.calls) == 3
    first_executor_tools = provider.calls[1][1]
    first_call_tool_names = [t["function"]["name"] for t in first_executor_tools]
    assert "read_file" in first_call_tool_names
    assert "load_mcp_tools" in first_call_tool_names
    assert "load_skill_tools" in first_call_tool_names
    assert "cron" in first_call_tool_names
    second_messages = provider.calls[2][0]
    assert [message.role for message in second_messages] == ["system", "user", "assistant", "tool"]
    assert second_messages[2].tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"note.txt"}'},
        }
    ]
    assert second_messages[3].tool_call_id == "call_1"
    assert second_messages[3].content == "hello from file"

    resumed = service.resume_thread(thread_id="tool-thread")
    assert resumed is not None
    assert resumed.message_count == 4
    assert resumed.active_capabilities.skills == []
    assert resumed.active_capabilities.mcp_servers == []


def test_runtime_service_stops_at_tool_iteration_cap(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello from file", encoding="utf-8")
    provider = LoopingToolProvider()
    service = _build_service(tmp_path, provider)

    result = service.run_turn(thread_id="loop-thread", user_input="keep reading")

    # T12: errors now routed through error_handler which fills response_text with a user-facing message
    assert "tool loop round limit reached after 16 rounds" in result.response_text
    assert result.last_error == "tool loop round limit reached after 16 rounds"
    assert provider.calls == 16


def test_runtime_service_persists_skill_activation_and_updates_prompt(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "editor"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "description: Editor skill\n"
        "---\n\n"
        "Edit safely.",
        encoding="utf-8",
    )
    provider = FakeActivationProvider()
    service = _build_service(tmp_path, provider)

    result = service.run_turn(thread_id="skill-thread", user_input="activate editor")

    assert result.last_error == ""
    assert result.response_text == "skill loaded"
    # classify + executor call 1 (load_skill_tools) + executor call 2 (final) = 3 calls
    assert len(provider.calls) == 3
    assert "### editor" in _system_text(provider.calls[2][0][0])
    assert "Edit safely." in _system_text(provider.calls[2][0][0])

    resumed = service.resume_thread(thread_id="skill-thread")
    assert resumed is not None
    assert resumed.active_capabilities.skills == ["editor"]
    assert resumed.message_count == 4


def test_runtime_service_activates_mcp_tools_for_followup_round(tmp_path: Path) -> None:
    mcp_registry = MCPRegistry()
    mcp_registry.register_server(
        MCPServerConfig(name="docs", transport="memory", settings={}),
        client_factory=FakeMCPClient,
    )
    provider = FakeMCPActivationProvider()
    service = _build_service(tmp_path, provider, mcp_registry=mcp_registry)

    result = service.run_turn(thread_id="mcp-thread", user_input="find docs")

    assert result.last_error == ""
    assert result.response_text == "docs loaded"
    # classify + executor call 1 (load_mcp_tools) + executor call 2 (search_docs) + executor call 3 (final) = 4 calls
    assert len(provider.calls) == 4
    second_round_tools = provider.calls[2][1]
    assert any(tool["function"]["name"] == "search_docs" for tool in second_round_tools)
    assert provider.calls[3][0][-1].role == "tool"
    assert provider.calls[3][0][-1].content == "search_docs:runtime"

    resumed = service.resume_thread(thread_id="mcp-thread")
    assert resumed is not None
    assert resumed.active_capabilities.mcp_servers == ["docs"]
    assert resumed.active_capabilities.mcp_tools == ["search_docs"]
