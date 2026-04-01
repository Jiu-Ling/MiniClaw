from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SkillSummary(BaseModel):
    """Lightweight index entry for a discovered skill."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    path: Path
    source: str = "workspace"
    available: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _apply_metadata_defaults(self) -> "SkillSummary":
        metadata = dict(self.metadata)
        metadata.setdefault("discoverable", True)
        self.metadata = metadata
        return self


class LoadedSkill(BaseModel):
    """Full skill payload loaded from disk."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: Path
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _apply_metadata_defaults(self) -> "LoadedSkill":
        metadata = dict(self.metadata)
        metadata.setdefault("discoverable", True)
        self.metadata = metadata
        return self
