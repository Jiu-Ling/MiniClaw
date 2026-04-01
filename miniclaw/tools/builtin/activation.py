from __future__ import annotations

import json
from pathlib import Path

from miniclaw.mcp.registry import MCPRegistry
from miniclaw.skills.loader import SkillLoader
from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool


def build_load_skill_tools_tool(loader: SkillLoader) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        raw_skills = call.arguments.get("skills", [])
        if not isinstance(raw_skills, list) or not raw_skills:
            return ToolResult(content="skills must be a non-empty list", is_error=True)

        found: list[str] = []
        missing: list[str] = []
        for value in raw_skills:
            skill_name = str(value).strip()
            if not skill_name:
                continue

            if loader.load_skill(skill_name) is None:
                missing.append(skill_name)
            else:
                found.append(skill_name)

        if not found:
            return ToolResult(content=f"skills not found: {', '.join(missing)}", is_error=True)
        if missing:
            return ToolResult(
                content=f"skills not found: {', '.join(missing)}",
                is_error=True,
                metadata={"skills": found},
            )

        return ToolResult(
            content=json.dumps({"activated_skills": found}, ensure_ascii=True),
            metadata={"activation_type": "skills", "skills": found},
        )

    return RegisteredTool(
        spec=ToolSpec(
            name="load_skill_tools",
            description=(
                "Activate one or more skills for the current thread so their guidance "
                "and tool visibility become available to the runtime."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["skills"],
                "additionalProperties": False,
            },
            source="builtin",
            metadata={"always_active": True, "discoverable": True},
        ),
        executor=execute,
    )


def build_add_mcp_server_tool(mcp_registry: MCPRegistry, mcp_config_path: Path) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        raw_config = call.arguments.get("server_config")
        if not isinstance(raw_config, dict):
            return ToolResult(content="server_config must be a JSON object", is_error=True)

        from miniclaw.mcp.config import append_server_config, _build_client_factory

        try:
            config = append_server_config(mcp_config_path, raw_config)
        except ValueError as exc:
            return ToolResult(content=str(exc), is_error=True)

        try:
            factory = _build_client_factory(config)
            mcp_registry.register_server(config, client_factory=factory)
        except ValueError as exc:
            return ToolResult(content=str(exc), is_error=True)

        tool_names = mcp_registry.list_tool_names(config.name)
        return ToolResult(
            content=json.dumps(
                {"added_server": config.name, "transport": config.transport, "tool_names": tool_names},
                ensure_ascii=False,
            ),
            metadata={"activation_type": "mcp", "server": config.name, "tool_names": tool_names},
        )

    return RegisteredTool(
        spec=ToolSpec(
            name="add_mcp_server",
            description=(
                "Register a new MCP server by appending its full JSON config to mcp.json "
                "and connecting it immediately. The user must provide the complete server_config object."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "server_config": {
                        "type": "object",
                        "description": (
                            'Full MCP server config, e.g. '
                            '{"name": "my-server", "transport": "stdio", "settings": {"command": "npx", "args": ["-y", "some-mcp-server"]}}'
                        ),
                    }
                },
                "required": ["server_config"],
                "additionalProperties": False,
            },
            source="builtin",
            metadata={"always_active": True, "discoverable": True},
        ),
        executor=execute,
    )


def build_reload_mcp_servers_tool(mcp_registry: MCPRegistry, mcp_config_path: Path) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        from miniclaw.mcp.config import register_servers_from_config

        try:
            registered = register_servers_from_config(mcp_registry, mcp_config_path)
        except Exception as exc:
            return ToolResult(content=f"reload failed: {exc}", is_error=True)

        all_servers = mcp_registry.list_server_names()
        return ToolResult(
            content=json.dumps(
                {"newly_registered": registered, "all_servers": all_servers},
                ensure_ascii=False,
            ),
        )

    return RegisteredTool(
        spec=ToolSpec(
            name="reload_mcp_servers",
            description=(
                "Re-read mcp.json and register any new MCP servers that were added "
                "since the last load. Already registered servers are kept."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            source="builtin",
            metadata={"always_active": True, "discoverable": True},
        ),
        executor=execute,
    )

def build_load_mcp_tools_tool(mcp_registry: MCPRegistry) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        server_name = str(call.arguments.get("server", "")).strip()
        if not server_name:
            return ToolResult(content="server is required", is_error=True)

        try:
            tool_names = mcp_registry.list_tool_names(server_name)
        except KeyError as error:
            return ToolResult(content=str(error), is_error=True)

        return ToolResult(
            content=json.dumps({"activated_server": server_name, "tool_names": tool_names}, ensure_ascii=True),
            metadata={"activation_type": "mcp", "server": server_name, "tool_names": tool_names},
        )

    return RegisteredTool(
        spec=ToolSpec(
            name="load_mcp_tools",
            description=(
                "Activate every tool exposed by one MCP server for the current thread "
                "before you call that server's tools."
            ),
            input_schema={
                "type": "object",
                "properties": {"server": {"type": "string"}},
                "required": ["server"],
                "additionalProperties": False,
            },
            source="builtin",
            metadata={"always_active": True, "discoverable": True},
        ),
        executor=execute,
    )
