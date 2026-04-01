from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from miniclaw.mcp.clients import SseMCPClient, StdioMCPClient
from miniclaw.mcp.contracts import MCPServerConfig
from miniclaw.mcp.registry import MCPRegistry

_SUPPORTED_TRANSPORTS = {"stdio", "sse"}


def load_mcp_config(config_path: Path) -> list[MCPServerConfig]:
    """Parse mcp.json and return a list of MCPServerConfig objects."""
    if not config_path.is_file():
        return []

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    servers = raw.get("servers", [])
    if not isinstance(servers, list):
        return []

    configs: list[MCPServerConfig] = []
    for entry in servers:
        if not isinstance(entry, dict):
            continue
        transport = str(entry.get("transport", "")).strip().lower()
        if transport not in _SUPPORTED_TRANSPORTS:
            continue
        configs.append(
            MCPServerConfig(
                name=str(entry["name"]).strip(),
                transport=transport,
                settings=entry.get("settings", {}),
            )
        )
    return configs


def register_servers_from_config(registry: MCPRegistry, config_path: Path) -> list[str]:
    """Load mcp.json and register all servers into the given registry.

    Returns the list of registered server names.
    """
    configs = load_mcp_config(config_path)
    registered: list[str] = []
    for config in configs:
        if config.name in {name for name in registry.list_server_names()}:
            continue
        factory = _build_client_factory(config)
        registry.register_server(config, client_factory=factory)
        registered.append(config.name)
    return registered


def append_server_config(config_path: Path, server_json: dict[str, Any]) -> MCPServerConfig:
    """Append a server entry to mcp.json and return the parsed config.

    Raises ValueError if the server name already exists or transport is unsupported.
    """
    name = str(server_json.get("name", "")).strip()
    transport = str(server_json.get("transport", "")).strip().lower()
    if not name:
        raise ValueError("server config must include a non-empty 'name'")
    if transport not in _SUPPORTED_TRANSPORTS:
        raise ValueError(f"unsupported transport: {transport!r} (must be one of {_SUPPORTED_TRANSPORTS})")

    existing: dict[str, Any] = {"servers": []}
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))

    servers = existing.get("servers", [])
    if not isinstance(servers, list):
        servers = []

    for entry in servers:
        if isinstance(entry, dict) and str(entry.get("name", "")).strip() == name:
            raise ValueError(f"server already exists in config: {name}")

    servers.append(server_json)
    existing["servers"] = servers
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return MCPServerConfig(
        name=name,
        transport=transport,
        settings=server_json.get("settings", {}),
    )


def _build_client_factory(config: MCPServerConfig):
    """Return a zero-arg callable that creates the appropriate MCPClient."""
    if config.transport == "stdio":
        return lambda: StdioMCPClient(config)
    if config.transport == "sse":
        return lambda: SseMCPClient(config)
    raise ValueError(f"unsupported transport: {config.transport}")
