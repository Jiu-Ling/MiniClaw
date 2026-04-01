from __future__ import annotations

from miniclaw.mcp.contracts import MCPToolSpec
from miniclaw.mcp.registry import MCPRegistry
from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool


def adapt_mcp_tool(*, server_name: str, tool: MCPToolSpec, registry: MCPRegistry) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        content = registry.call_tool(server_name, call.name, call.arguments)
        return ToolResult(content=content, metadata={"server": server_name})

    return RegisteredTool(
        spec=ToolSpec(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            source="mcp",
            metadata={**tool.metadata, "server": server_name},
        ),
        executor=execute,
    )
