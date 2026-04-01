from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict


RUNTIME_METADATA_BLOCK_TITLE = "Runtime Metadata (metadata-only)"


class RuntimeMetadata(BaseModel):
    """Runtime-only metadata that should stay out of the system prompt."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    channel: str | None = None
    chat_id: str | None = None
    clock: str | None = None

    def render_block(self) -> str:
        return render_runtime_metadata_block(self)


def render_runtime_metadata_block(runtime_metadata: Mapping[str, Any] | RuntimeMetadata | None) -> str:
    """Render runtime metadata as plain text for the current user message."""

    if runtime_metadata is None:
        return ""

    if isinstance(runtime_metadata, RuntimeMetadata):
        values = runtime_metadata.model_dump(exclude_none=True)
    else:
        values = {key: value for key, value in runtime_metadata.items() if value is not None}

    if not values:
        return ""

    ordered_keys = ("thread_id", "channel", "chat_id", "clock")
    lines = [f"## {RUNTIME_METADATA_BLOCK_TITLE}"]
    for key in ordered_keys:
        if key in values:
            lines.append(f"{key}: {values[key]}")
    for key in sorted(key for key in values if key not in ordered_keys):
        lines.append(f"{key}: {values[key]}")
    return "\n".join(lines)
