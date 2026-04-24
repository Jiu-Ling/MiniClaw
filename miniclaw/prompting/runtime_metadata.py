from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict


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
    """Render runtime metadata as a <system-reminder>-wrapped block.

    Used by ContextBuilder to inject metadata into the current-turn user message
    without polluting the cache-eligible system prompt. The <system-reminder>
    convention tells Claude-family models to treat content as system-level
    rather than user input. Other providers see plain text and degrade
    gracefully.

    Only keys in _VISIBLE_KEYS are emitted. Internal fields (_turn_run_id,
    _turn_trace_id), user-sandbox paths, sender_id, user_id, and any key
    starting with '_' are excluded.
    """
    if runtime_metadata is None:
        return ""

    if isinstance(runtime_metadata, RuntimeMetadata):
        values = runtime_metadata.model_dump(exclude_none=True)
    else:
        values = {key: value for key, value in runtime_metadata.items() if value is not None}

    if not values:
        return ""

    inner_lines = []
    for key in _VISIBLE_KEYS:
        if key in values:
            inner_lines.append(f"{key}: {values[key]}")

    if not inner_lines:
        return ""

    inner = "\n".join(inner_lines)
    return f"<system-reminder>\nruntime_metadata:\n{inner}\n</system-reminder>"
