from pathlib import Path

from miniclaw.mcp.contracts import MCPServerConfig, MCPToolSpec
from miniclaw.mcp.registry import MCPRegistry
from miniclaw.bootstrap import build_skill_loader, build_tool_registry
from miniclaw.providers.contracts import ChatMessage
from miniclaw.prompting import BootstrapLoader, ContextBuilder


def _write_bootstrap_file(root: Path, name: str, content: str) -> None:
    (root / name).write_text(content, encoding="utf-8")


def _write_skill(root: Path, name: str, content: str) -> None:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


class FakeMCPClient:
    def connect(self) -> None:
        return None

    def list_tools(self) -> list[MCPToolSpec]:
        return [
            MCPToolSpec(
                name="search_docs",
                description="Search docs",
                input_schema={},
                metadata={},
            )
        ]

    def call_tool(self, name: str, arguments: dict[str, object]) -> str:
        return f"{name}:{arguments}"


def test_bootstrap_loader_preserves_contract_order_and_skips_missing(tmp_path: Path) -> None:
    _write_bootstrap_file(tmp_path, "SOUL.md", "Soul guidance.")
    _write_bootstrap_file(tmp_path, "USER.md", "User guidance.")
    _write_bootstrap_file(tmp_path, "USER.md", "User guidance.")

    loader = BootstrapLoader(workspace=tmp_path)

    files = loader.load()

    assert [item.name for item in files] == ["SOUL.md", "USER.md"]
    assert [item.content for item in files] == [
        "Soul guidance.",
        "User guidance.",
    ]


def test_context_builder_builds_stable_system_prompt_from_bootstrap_files(tmp_path: Path) -> None:
    _write_bootstrap_file(tmp_path, "SOUL.md", "Soul guidance.")
    _write_bootstrap_file(tmp_path, "SOUL.md", "Soul guidance.")
    _write_bootstrap_file(tmp_path, "USER.md", "User guidance.")

    builder = ContextBuilder(workspace=tmp_path)

    prompt = builder.build_system_prompt()

    assert prompt == builder.build_system_prompt()
    assert prompt.startswith("# MiniClaw\n\nYou are MiniClaw, a recoverable agent runtime for real task execution.")
    assert "## Runtime" in prompt
    assert "## Workspace" in prompt
    assert f"- Workspace root: {tmp_path.resolve()}" in prompt
    assert "## Operating Guidelines" in prompt
    assert "## Memory" in prompt
    assert "## SOUL.md\nSoul guidance." in prompt
    assert "## SOUL.md\nSoul guidance." in prompt
    assert "## USER.md\nUser guidance." in prompt
    assert "## Capability Guidance" in prompt


def _system_text(msg: ChatMessage) -> str:
    """Extract the full text of a system message, whether content or content_parts."""
    if msg.content is not None:
        return msg.content
    return "\n\n".join(
        part.get("text", "") for part in msg.content_parts if isinstance(part, dict)
    )


def test_context_builder_builds_provider_messages_from_runtime_state(tmp_path: Path) -> None:
    builder = ContextBuilder(workspace=tmp_path, system_prompt="You are MiniClaw.")

    messages = builder.build_provider_messages(
        {
            "thread_id": "thread-1",
            "user_input": "Say hello",
            "messages": [{"role": "user", "content": "Say hello"}],
            "memory_context": (
                "Relevant memory:\n"
                '- [fact] user prefers terse answers {"source": "test"}'
            ),
        }
    )

    assert [message.role for message in messages] == ["system", "user"]
    # Static sections first (for prefix caching), then dynamic (memory)
    system_text = _system_text(messages[0])
    assert system_text == (
        "You are MiniClaw.\n\n"
        "## Capability Guidance\n"
        "Visible tools can be used directly.\n"
        "Use load_skill_tools to activate skill-backed capabilities.\n"
        "Use load_mcp_tools to activate MCP server tools.\n\n"
        "Activation tools:\n"
        "- load_skill_tools\n"
        "- load_mcp_tools\n\n"
        "## Memory\n"
        "Relevant memory:\n"
        '- [fact] user prefers terse answers {"source": "test"}'
    )
    assert "Say hello" in messages[1].content
    assert "thread_id: thread-1" in messages[1].content
    assert messages[1].content_parts == []
    # System message uses content_parts for cache-control split
    assert messages[0].content_parts, "system message must use content_parts"
    assert any(
        p.get("cache_control", {}).get("type") == "ephemeral"
        for p in messages[0].content_parts
    )


def test_context_builder_includes_bootstrap_content_in_provider_messages(tmp_path: Path) -> None:
    _write_bootstrap_file(tmp_path, "SOUL.md", "Soul guidance.")

    builder = ContextBuilder(workspace=tmp_path, system_prompt="You are MiniClaw.")

    messages = builder.build_provider_messages(
        {
            "messages": [{"role": "user", "content": "Say hello"}],
        }
    )

    assert messages[0].role == "system"
    system_text = _system_text(messages[0])
    assert system_text == (
        "You are MiniClaw.\n\n## SOUL.md\nSoul guidance.\n\n"
        "## Capability Guidance\n"
        "Visible tools can be used directly.\n"
        "Use load_skill_tools to activate skill-backed capabilities.\n"
        "Use load_mcp_tools to activate MCP server tools.\n\n"
        "Activation tools:\n"
        "- load_skill_tools\n"
        "- load_mcp_tools"
    )
    assert messages[1].role == "user"
    assert messages[1].content == "Say hello"


def test_context_builder_keeps_system_prompt_stable_when_runtime_metadata_changes(
    tmp_path: Path,
) -> None:
    builder = ContextBuilder(workspace=tmp_path, system_prompt="You are MiniClaw.")

    first_messages = builder.build_provider_messages(
        {
            "user_input": "Say hello",
            "thread_id": "thread-1",
            "runtime_metadata": {"thread_id": "thread-1", "clock": "2026-03-27T10:00:00Z"},
            "messages": [{"role": "user", "content": "Say hello"}],
        }
    )
    second_messages = builder.build_provider_messages(
        {
            "user_input": "Say hello",
            "thread_id": "thread-2",
            "runtime_metadata": {"thread_id": "thread-2", "clock": "2026-03-27T10:05:00Z"},
            "messages": [{"role": "user", "content": "Say hello"}],
        }
    )

    system_prompt = builder.build_system_prompt()
    assert "## Runtime Metadata (metadata-only)" not in system_prompt
    assert "thread_id:" not in system_prompt
    assert "clock:" not in system_prompt
    first_system_text = _system_text(first_messages[0])
    second_system_text = _system_text(second_messages[0])
    assert first_system_text == second_system_text == system_prompt
    assert first_messages[1].content != second_messages[1].content
    assert "thread_id: thread-1" in first_messages[1].content
    assert "clock: 2026-03-27T10:00:00Z" in first_messages[1].content
    assert "thread_id: thread-2" in second_messages[1].content
    assert "clock: 2026-03-27T10:05:00Z" in second_messages[1].content


def test_context_builder_keeps_plain_text_provider_messages_compatible(tmp_path: Path) -> None:
    builder = ContextBuilder(workspace=tmp_path, system_prompt="You are MiniClaw.")

    messages = builder.build_provider_messages(
        {
            "user_input": "Say hello",
            "thread_id": "thread-plain",
            "runtime_metadata": {"thread_id": "thread-plain"},
            "messages": [ChatMessage(role="user", content="Say hello").model_dump()],
        }
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert "You are MiniClaw." in _system_text(messages[0])
    assert messages[1].role == "user"
    assert "Say hello" in messages[1].content
    assert "thread_id: thread-plain" in messages[1].content


def test_context_builder_preserves_multimodal_user_message_parts(tmp_path: Path) -> None:
    builder = ContextBuilder(workspace=tmp_path, system_prompt="You are MiniClaw.")

    messages = builder.build_provider_messages(
        {
            "user_input": "describe this",
            "thread_id": "thread-vision",
            "runtime_metadata": {"thread_id": "thread-vision"},
            "messages": [
                {
                    "role": "user",
                    "content": "describe this",
                    "content_parts": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image_url", "image_url": {"url": "https://files.example/photo.jpg"}},
                    ],
                }
            ],
        }
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    parts = messages[1].content_parts
    assert parts[0] == {"type": "text", "text": "describe this"}
    assert parts[1] == {"type": "image_url", "image_url": {"url": "https://files.example/photo.jpg"}}
    metadata_part = parts[2]["text"]
    assert "thread_id: thread-vision" in metadata_part


def test_context_builder_does_not_append_runtime_metadata_to_history_only_messages(
    tmp_path: Path,
) -> None:
    builder = ContextBuilder(workspace=tmp_path, system_prompt="You are MiniClaw.")

    messages = builder.build_provider_messages(
        {
            "thread_id": "thread-history",
            "runtime_metadata": {"thread_id": "thread-history", "clock": "2026-03-27T11:00:00Z"},
            "messages": [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": "Historical follow-up"},
            ],
        }
    )

    expected_system_text = (
        "You are MiniClaw.\n\n"
        "## Capability Guidance\n"
        "Visible tools can be used directly.\n"
        "Use load_skill_tools to activate skill-backed capabilities.\n"
        "Use load_mcp_tools to activate MCP server tools.\n\n"
        "Activation tools:\n"
        "- load_skill_tools\n"
        "- load_mcp_tools"
    )
    assert len(messages) == 4
    assert messages[0].role == "system"
    assert _system_text(messages[0]) == expected_system_text
    assert messages[1] == ChatMessage(role="user", content="Previous question")
    assert messages[2] == ChatMessage(role="assistant", content="Previous answer")
    assert messages[3] == ChatMessage(role="user", content="Historical follow-up")


def test_context_builder_orders_memory_and_skill_sections_in_system_prompt(tmp_path: Path) -> None:
    _write_bootstrap_file(tmp_path, "SOUL.md", "Soul guidance.")
    _write_skill(
        tmp_path,
        "editor",
        "---\n"
        "description: Editor skill\n"
        "always_active: true\n"
        "---\n\n"
        "Edit files carefully.",
    )
    _write_skill(
        tmp_path,
        "notes",
        "---\n"
        "description: Notes skill\n"
        "---\n\n"
        "Take concise notes.",
    )

    builder = ContextBuilder(workspace=tmp_path, system_prompt="You are MiniClaw.")

    prompt = builder.build_system_prompt(
        {
            "memory_context": "Relevant memory:\n- [fact] user prefers terse answers",
        }
    )

    assert "You are MiniClaw." in prompt
    assert "## SOUL.md" in prompt
    assert "## Memory" in prompt
    assert "## Active Skills" in prompt
    assert "### editor" in prompt
    assert "## Capability Guidance" in prompt
    assert "Visible tools can be used directly." in prompt
    assert "Activation tools:" in prompt
    assert "## Skills Summary" in prompt
    assert "- editor: Editor skill" in prompt
    assert "[active]" in prompt
    assert "- notes: Notes skill" in prompt
    assert "[discoverable]" in prompt
    assert "## Tools Summary" not in prompt
    assert "## MCP Summary" not in prompt


def test_context_builder_adds_tool_and_mcp_summaries_without_exposing_visible_tools(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "notes",
        "---\n"
        "description: Notes skill\n"
        "---\n\n"
        "Take concise notes.",
    )
    mcp_registry = MCPRegistry()
    mcp_registry.register_server(
        MCPServerConfig(name="docs", transport="memory", settings={}),
        client_factory=FakeMCPClient,
    )
    loader = build_skill_loader(workspace=tmp_path)
    tool_registry = build_tool_registry(
        workspace=tmp_path,
        skill_loader=loader,
        mcp_registry=mcp_registry,
    )
    builder = ContextBuilder(
        workspace=tmp_path,
        system_prompt="You are MiniClaw.",
        skills_loader=loader,
        tool_registry=tool_registry,
        mcp_registry=mcp_registry,
    )

    prompt = builder.build_system_prompt({})

    assert "## Capability Guidance" in prompt
    assert "Visible tools can be used directly." in prompt
    assert "## Skills Summary" in prompt
    # Non-core tools appear in Tools Summary as discoverable hints
    assert "## Tools Summary" in prompt
    assert "## MCP Summary" in prompt
    assert "Use load_mcp_tools to activate MCP server tools." in prompt
    assert "- docs: 1 tool: search_docs" in prompt
