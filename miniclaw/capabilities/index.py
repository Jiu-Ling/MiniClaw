from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from miniclaw.capabilities.contracts import CapabilityEntry, CapabilityIndex
from miniclaw.mcp.registry import MCPRegistry
from miniclaw.skills.loader import SkillLoader
from miniclaw.tools.contracts import ToolSpec
from miniclaw.tools.registry import ToolRegistry


class CapabilityIndexBuilder:
    def __init__(
        self,
        *,
        skill_loader: SkillLoader,
        tool_registry: ToolRegistry,
        mcp_registry: MCPRegistry | None = None,
    ) -> None:
        self.skill_loader = skill_loader
        self.tool_registry = tool_registry
        self.mcp_registry = mcp_registry

    def build(self, active_capabilities: Any | None = None) -> CapabilityIndex:
        visible_tools = self._build_visible_tools(active_capabilities)
        visible_names = {entry.name for entry in visible_tools}

        return CapabilityIndex(
            visible_tools=visible_tools,
            discoverable_tools=self._build_discoverable_tools(active_capabilities, visible_names),
            skills=self._build_skills(active_capabilities),
            mcp_servers=self._build_mcp_servers(active_capabilities),
            selection_hints=[
                "Visible tools can be used directly.",
                "Use load_skill_tools to activate skill-backed capabilities.",
                "Use load_mcp_tools to activate MCP server tools.",
            ],
            activation_tools=["load_skill_tools", "load_mcp_tools"],
        )

    def _build_visible_tools(self, active_capabilities: Any | None) -> list[CapabilityEntry]:
        return [
            self._tool_entry(tool, active_capabilities)
            for tool in self.tool_registry.list_visible_tools(active_capabilities)
        ]

    def _build_discoverable_tools(
        self,
        active_capabilities: Any | None,
        visible_names: set[str],
    ) -> list[CapabilityEntry]:
        entries: list[CapabilityEntry] = []
        for tool in self.tool_registry.list_tools():
            if tool.name in visible_names:
                continue
            if not self._is_discoverable(tool):
                continue
            entries.append(self._tool_entry(tool, active_capabilities))
        return entries

    def _build_skills(self, active_capabilities: Any | None) -> list[CapabilityEntry]:
        active_names = set(getattr(active_capabilities, "skills", []) or [])
        return [
            self._skill_entry(skill, active_names)
            for skill in self.skill_loader.list_skills()
        ]

    def _build_mcp_servers(self, active_capabilities: Any | None) -> list[CapabilityEntry]:
        if self.mcp_registry is None:
            return []

        active_servers = set(getattr(active_capabilities, "mcp_servers", []) or [])
        entries: list[CapabilityEntry] = []

        for server_name in self.mcp_registry.list_server_names():
            tool_names = sorted(self.mcp_registry.list_tool_names(server_name))
            server_tools = set(tool_names)
            entries.append(
                CapabilityEntry(
                    kind="mcp_server",
                    name=server_name,
                    source="mcp",
                    description=self._describe_mcp_server(tool_names),
                    always_active=False,
                    discoverable=True,
                    active=server_name in active_servers,
                    children=tool_names,
                )
            )
        return entries

    def _tool_entry(self, tool: ToolSpec, active_capabilities: Any | None) -> CapabilityEntry:
        active = self._is_visible(tool) or tool.name in (getattr(active_capabilities, "tools", []) or [])
        metadata = tool.metadata
        return CapabilityEntry(
            kind="tool",
            name=tool.name,
            source=tool.source,
            description=tool.description,
            always_active=self._is_truthy(metadata.get("always_active")),
            discoverable=self._is_discoverable(tool),
            active=active,
        )

    def _skill_entry(self, skill, active_names: set[str]) -> CapabilityEntry:
        always_active = self._is_truthy(skill.metadata.get("always_active"))
        return CapabilityEntry(
            kind="skill",
            name=skill.name,
            source=skill.source,
            description=skill.description,
            always_active=always_active,
            discoverable=self._is_truthy(skill.metadata.get("discoverable", True)),
            active=always_active or skill.name in active_names,
        )

    @staticmethod
    def _is_discoverable(tool: ToolSpec) -> bool:
        return CapabilityIndexBuilder._is_truthy(tool.metadata.get("discoverable", True))

    @staticmethod
    def _is_visible(tool: ToolSpec) -> bool:
        return CapabilityIndexBuilder._is_truthy(tool.metadata.get("always_active")) or CapabilityIndexBuilder._is_truthy(
            tool.metadata.get("default_visible")
        )

    @staticmethod
    def _is_truthy(value: object) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _describe_mcp_server(tool_names: Iterable[str]) -> str:
        names = [name.strip() for name in tool_names if str(name).strip()]
        if not names:
            return "No tools available."
        if len(names) == 1:
            return f"1 tool: {names[0]}"
        return f"{len(names)} tools: {', '.join(names)}"
