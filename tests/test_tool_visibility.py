from pathlib import Path

from miniclaw.bootstrap import build_skill_loader, build_tool_registry
from miniclaw.prompting.context import ContextBuilder


def _write_skill(root: Path, name: str, content: str) -> None:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def test_default_visible_tools_only_include_always_active_tools(tmp_path: Path) -> None:
    loader = build_skill_loader(workspace=tmp_path)
    registry = build_tool_registry(workspace=tmp_path, skill_loader=loader)

    all_tool_names = [tool.name for tool in registry.list_tools()]
    tool_names = [tool.name for tool in registry.list_visible_tools()]

    assert all_tool_names == [
        "add_mcp_server",
        "cron",
        "list_skills",
        "load_mcp_tools",
        "load_skill",
        "load_skill_tools",
        "read_file",
        "reload_mcp_servers",
        "send",
        "shell",
        "spawn_subagent",
        "web_search",
    ]
    # Core tools are always visible; others are discoverable but not visible
    assert tool_names == [
        "cron",
        "load_mcp_tools",
        "load_skill_tools",
        "read_file",
        "send",
        "shell",
        "spawn_subagent",
        "web_search",
    ]
    # Non-core tools are still registered (executable) but not in the visible set
    assert len(all_tool_names) > len(tool_names)


def test_send_tool_not_worker_visible():
    from miniclaw.tools.builtin.send import build_send_tool
    tool = build_send_tool(messaging_bridge=None)
    assert tool.spec.metadata.get("worker_visible") is False


def test_default_tool_is_worker_visible():
    from miniclaw.tools.contracts import ToolSpec
    spec = ToolSpec(name="x", description="", source="builtin")
    assert spec.metadata.get("worker_visible") is True


def test_non_active_skills_remain_visible_in_prompt_summary(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "editor",
        "---\n"
        "description: Editor skill\n"
        "---\n\n"
        "Edit safely.",
    )

    builder = ContextBuilder(workspace=tmp_path, system_prompt="You are MiniClaw.")

    prompt = builder.build_system_prompt({})

    assert "## Skills Summary" in prompt
    assert "- editor: Editor skill" in prompt
    assert "## Active Skills" not in prompt
