from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from miniclaw.tools.contracts import ToolCall, ToolExecutor, ToolResult, ToolSpec

if TYPE_CHECKING:
    from miniclaw.mcp.registry import MCPRegistry
    from miniclaw.runtime.state import ActiveCapabilities
    from miniclaw.skills.loader import SkillLoader


@dataclass(slots=True)
class RegisteredTool:
    spec: ToolSpec
    executor: ToolExecutor


class ToolRegistry:
    def __init__(
        self,
        *,
        skill_loader: SkillLoader | None = None,
        mcp_registry: MCPRegistry | None = None,
    ) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self.skill_loader = skill_loader
        self.mcp_registry = mcp_registry

    def register(self, tool: RegisteredTool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"tool already registered: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def register_many(self, tools: list[RegisteredTool]) -> None:
        for tool in tools:
            self.register(tool)

    def clone(self) -> ToolRegistry:
        cloned = ToolRegistry(
            skill_loader=self.skill_loader,
            mcp_registry=self.mcp_registry,
        )
        cloned._tools = dict(self._tools)
        return cloned

    def replace(self, tool: RegisteredTool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return [self._tools[name].spec for name in sorted(self._tools)]

    def list_visible_tools(self, active_capabilities: ActiveCapabilities | None = None) -> list[ToolSpec]:
        return [tool.spec for tool in self._visible_tools(active_capabilities)]

    def visible_tool_names(self, active_capabilities: ActiveCapabilities | None = None) -> list[str]:
        return [tool.spec.name for tool in self._visible_tools(active_capabilities)]

    def execute(self, call: ToolCall, active_capabilities: ActiveCapabilities | None = None) -> ToolResult:
        tool = self._resolve_tool(call.name, active_capabilities)
        if tool is None:
            raise KeyError(f"unknown tool: {call.name}")
        return tool.executor(call)

    def _visible_tools(self, active_capabilities: ActiveCapabilities | None) -> list[RegisteredTool]:
        resolved: dict[str, RegisteredTool] = {
            name: tool
            for name, tool in sorted(self._tools.items())
            if self._is_visible(tool.spec) or self._is_activated(tool.spec, active_capabilities)
        }
        for tool in self._dynamic_mcp_tools(active_capabilities):
            resolved.setdefault(tool.spec.name, tool)
        return [resolved[name] for name in sorted(resolved)]

    def _resolve_tool(self, name: str, active_capabilities: ActiveCapabilities | None) -> RegisteredTool | None:
        tool = self.get(name)
        if tool is not None:
            return tool

        dynamic_tools = [tool for tool in self._dynamic_mcp_tools(active_capabilities) if tool.spec.name == name]
        if not dynamic_tools:
            return None
        if len(dynamic_tools) > 1:
            raise KeyError(f"multiple active tools named: {name}")
        return dynamic_tools[0]

    def _dynamic_mcp_tools(self, active_capabilities: ActiveCapabilities | None) -> list[RegisteredTool]:
        if active_capabilities is None or self.mcp_registry is None:
            return []

        from miniclaw.mcp.adapters import adapt_mcp_tool

        tools: list[RegisteredTool] = []
        for server_name in active_capabilities.mcp_servers:
            for tool in self.mcp_registry.list_tools(server_name):
                tools.append(adapt_mcp_tool(server_name=server_name, tool=tool, registry=self.mcp_registry))
        return tools

    @staticmethod
    def _is_visible(spec: ToolSpec) -> bool:
        return ToolRegistry._is_truthy(spec.metadata.get("always_active")) or ToolRegistry._is_truthy(
            spec.metadata.get("default_visible")
        )

    @staticmethod
    def _is_activated(spec: ToolSpec, active_capabilities: ActiveCapabilities | None) -> bool:
        if active_capabilities is None:
            return False
        return spec.name in active_capabilities.tools

    @staticmethod
    def _is_truthy(value: object) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
