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


# Keys visible to the LLM in the user message. Everything else is internal.
# Keep this list minimal — every extra key dilutes the user's actual message.
_VISIBLE_KEYS = ("thread_id", "channel", "clock")


def render_runtime_metadata_block(runtime_metadata: Mapping[str, Any] | RuntimeMetadata | None) -> str:
    """Render runtime metadata as a compact annotation on the current user message.

    Only includes keys in _VISIBLE_KEYS. Internal fields (_turn_run_id,
    _turn_trace_id), user-sandbox paths, sender_id, user_id, and any key
    starting with '_' are excluded to avoid confusing the LLM.
    """
    if runtime_metadata is None:
        return ""

    if isinstance(runtime_metadata, RuntimeMetadata):
        values = runtime_metadata.model_dump(exclude_none=True)
    else:
        values = {key: value for key, value in runtime_metadata.items() if value is not None}

    if not values:
        return ""

    lines = [f"## {RUNTIME_METADATA_BLOCK_TITLE}"]
    for key in _VISIBLE_KEYS:
        if key in values:
            lines.append(f"{key}: {values[key]}")

    # If no visible keys matched, don't emit an empty block
    if len(lines) == 1:
        return ""

    return "\n".join(lines)
