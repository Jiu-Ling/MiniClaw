from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CapabilityEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str
    source: str
    description: str = ""
    always_active: bool = False
    discoverable: bool = True
    active: bool = False
    children: list[str] = Field(default_factory=list)


class CapabilityIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visible_tools: list[CapabilityEntry] = Field(default_factory=list)
    discoverable_tools: list[CapabilityEntry] = Field(default_factory=list)
    skills: list[CapabilityEntry] = Field(default_factory=list)
    mcp_servers: list[CapabilityEntry] = Field(default_factory=list)
    selection_hints: list[str] = Field(default_factory=list)
    activation_tools: list[str] = Field(default_factory=list)
