from pathlib import Path

from miniclaw.bootstrap import build_mcp_registry, build_skill_loader, build_tool_registry
from miniclaw.config.settings import Settings


def test_build_skill_loader_uses_workspace_and_builtin_skill_root(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".miniclaw"
    builtin_root = tmp_path / "builtin"
    builtin_skill = builtin_root / "shared"
    builtin_skill.mkdir(parents=True)
    (builtin_skill / "SKILL.md").write_text(
        "---\ndescription: Builtin skill\n---\n\nBuiltin content.",
        encoding="utf-8",
    )

    user_skill = runtime_dir / "skills" / "shared"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text(
        "---\ndescription: Workspace skill\n---\n\nWorkspace content.",
        encoding="utf-8",
    )

    settings = Settings(
        api_key="test-api-key",
        base_url="https://api.example.com/v1",
        model="gpt-4o-mini",
        sqlite_path=runtime_dir / "miniclaw.sqlite3",
    )
    loader = build_skill_loader(settings=settings, builtin_skills_dir=builtin_root)

    loaded = loader.load_skill("shared")

    assert loaded is not None
    assert loaded.path == user_skill / "SKILL.md"
    assert "Workspace content." in loaded.content


def test_build_tool_registry_registers_capability_tools(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "writer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: Writing help\n---\n\nWrite clearly.",
        encoding="utf-8",
    )

    loader = build_skill_loader(workspace=tmp_path)
    registry = build_tool_registry(workspace=tmp_path, skill_loader=loader)

    tool_names = [tool.name for tool in registry.list_tools()]
    visible_tool_names = [tool.name for tool in registry.list_visible_tools()]

    assert tool_names == [
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
    # Core tools are visible; others are registered but discoverable-only
    assert visible_tool_names == [
        "cron",
        "load_mcp_tools",
        "load_skill_tools",
        "read_file",
        "send",
        "shell",
        "spawn_subagent",
        "web_search",
    ]


def test_build_mcp_registry_starts_empty() -> None:
    registry = build_mcp_registry()

    assert registry.list_tools() == []


def test_bootstrap_registers_spawn_subagent_tool():
    from miniclaw.bootstrap import build_tool_registry

    registry = build_tool_registry()
    tool = registry.get("spawn_subagent")
    spec = tool.spec if tool is not None else None
    assert spec is not None, "spawn_subagent must be registered"
    assert spec.metadata.get("worker_visible") is False
