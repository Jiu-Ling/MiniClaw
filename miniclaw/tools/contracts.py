from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _apply_metadata_defaults(self) -> "ToolSpec":
        metadata = dict(self.metadata)
        metadata.setdefault("discoverable", True)
        metadata.setdefault("worker_visible", True)
        self.metadata = metadata
        return self


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.is_error


ToolExecutor = Callable[[ToolCall], ToolResult]
