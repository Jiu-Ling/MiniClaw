from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    transport: str
    settings: dict[str, Any] = Field(default_factory=dict)


class MCPToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _apply_metadata_defaults(self) -> "MCPToolSpec":
        metadata = dict(self.metadata)
        metadata.setdefault("discoverable", True)
        self.metadata = metadata
        return self


@runtime_checkable
class MCPClient(Protocol):
    def connect(self) -> None: ...

    def list_tools(self) -> list[MCPToolSpec]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...
