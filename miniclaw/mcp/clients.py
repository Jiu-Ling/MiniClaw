from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any

import requests

from miniclaw.mcp.contracts import MCPClient, MCPServerConfig, MCPToolSpec


class StdioMCPClient:
    """MCP client that communicates with a server via stdin/stdout using JSON-RPC."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        settings = self._config.settings
        command = settings.get("command", "")
        args: list[str] = list(settings.get("args", []))
        env: dict[str, str] | None = settings.get("env") or None

        import os

        merged_env: dict[str, str] | None = None
        if env:
            merged_env = {**os.environ, **env}

        self._process = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=merged_env,
        )

    def list_tools(self) -> list[MCPToolSpec]:
        response = self._send_request("tools/list", {})
        raw_tools = response.get("tools", [])
        tools: list[MCPToolSpec] = []
        for raw in raw_tools:
            tools.append(
                MCPToolSpec(
                    name=raw.get("name", ""),
                    description=raw.get("description", ""),
                    input_schema=raw.get("inputSchema", {}),
                )
            )
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        response = self._send_request("tools/call", {"name": name, "arguments": arguments})
        content_list = response.get("content", [])
        parts: list[str] = []
        for item in content_list:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
        return "\n".join(parts) if parts else json.dumps(response, ensure_ascii=False)

    def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError(f"stdio mcp client not connected: {self._config.name}")

        with self._lock:
            self._request_id += 1
            request_id = self._request_id

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        raw = json.dumps(payload, ensure_ascii=False) + "\n"

        with self._lock:
            self._process.stdin.write(raw)
            self._process.stdin.flush()
            line = self._process.stdout.readline()

        if not line:
            raise RuntimeError(f"stdio mcp server closed unexpectedly: {self._config.name}")

        response = json.loads(line)
        if "error" in response:
            error = response["error"]
            raise RuntimeError(f"mcp error ({error.get('code', '?')}): {error.get('message', 'unknown')}")

        return response.get("result", {})

    def __del__(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()


class SseMCPClient:
    """MCP client that communicates with a server via HTTP SSE transport."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._base_url: str = ""
        self._session: requests.Session | None = None
        self._request_id = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        settings = self._config.settings
        self._base_url = settings.get("url", "").rstrip("/")
        if not self._base_url:
            raise ValueError(f"sse mcp server requires 'url' in settings: {self._config.name}")

        self._session = requests.Session()
        headers = settings.get("headers") or {}
        if headers:
            self._session.headers.update(headers)

    def list_tools(self) -> list[MCPToolSpec]:
        response = self._send_request("tools/list", {})
        raw_tools = response.get("tools", [])
        tools: list[MCPToolSpec] = []
        for raw in raw_tools:
            tools.append(
                MCPToolSpec(
                    name=raw.get("name", ""),
                    description=raw.get("description", ""),
                    input_schema=raw.get("inputSchema", {}),
                )
            )
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        response = self._send_request("tools/call", {"name": name, "arguments": arguments})
        content_list = response.get("content", [])
        parts: list[str] = []
        for item in content_list:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
        return "\n".join(parts) if parts else json.dumps(response, ensure_ascii=False)

    def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError(f"sse mcp client not connected: {self._config.name}")

        with self._lock:
            self._request_id += 1
            request_id = self._request_id

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        response = self._session.post(
            self._base_url,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            error = data["error"]
            raise RuntimeError(f"mcp error ({error.get('code', '?')}): {error.get('message', 'unknown')}")

        return data.get("result", {})
