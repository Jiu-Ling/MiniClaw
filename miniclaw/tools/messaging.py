from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class OutboundMessage:
    text: str
    thread_id: str | None = None
    channel: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MessagingBridge(Protocol):
    def send_message(self, message: OutboundMessage) -> dict[str, object]: ...
    def send_file(self, file_path: str, *, caption: str = "") -> dict[str, object]: ...


__all__ = ["MessagingBridge", "OutboundMessage"]
