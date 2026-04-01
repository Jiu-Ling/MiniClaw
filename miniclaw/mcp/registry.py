from __future__ import annotations

from collections.abc import Callable
from typing import Any

from miniclaw.mcp.contracts import MCPClient, MCPServerConfig, MCPToolSpec


class MCPRegistry:
    def __init__(self) -> None:
        self._servers: dict[str, tuple[MCPServerConfig, Callable[[], MCPClient]]] = {}
        self._clients: dict[str, MCPClient] = {}

    def register_server(self, config: MCPServerConfig, *, client_factory: Callable[[], MCPClient]) -> None:
        if config.name in self._servers:
            raise ValueError(f"mcp server already registered: {config.name}")
        self._servers[config.name] = (config, client_factory)

    def _get_client(self, server_name: str) -> MCPClient:
        client = self._clients.get(server_name)
        if client is not None:
            return client

        try:
            _, factory = self._servers[server_name]
        except KeyError as error:
            raise KeyError(f"unknown mcp server: {server_name}") from error

        client = factory()
        client.connect()
        self._clients[server_name] = client
        return client

    def list_tools(self, server_name: str | None = None) -> list[MCPToolSpec]:
        if server_name is not None:
            return [self._attach_server_context(server_name, tool) for tool in self._get_client(server_name).list_tools()]

        tools: list[MCPToolSpec] = []
        for name in sorted(self._servers):
            tools.extend(self._attach_server_context(name, tool) for tool in self._get_client(name).list_tools())
        return tools

    def list_server_names(self) -> list[str]:
        return sorted(self._servers)

    def list_tool_names(self, server_name: str) -> list[str]:
        return [tool.name for tool in self.list_tools(server_name)]

    def call_tool(self, server_name: str, name: str, arguments: dict[str, Any]) -> str:
        return self._get_client(server_name).call_tool(name, arguments)

    @staticmethod
    def _attach_server_context(server_name: str, tool: MCPToolSpec) -> MCPToolSpec:
        return tool.model_copy(update={"metadata": {**tool.metadata, "server": server_name}})
