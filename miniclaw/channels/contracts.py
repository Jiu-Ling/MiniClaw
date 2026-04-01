from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class StreamCapability(Enum):
    NATIVE = "native"
    EDIT = "edit"
    BUFFER = "buffer"


@dataclass
class InboundMessage:
    channel: str
    channel_id: str
    thread_id: str
    message_id: str
    text: str
    sender_id: str = ""
    attachments: list[Attachment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutboundMessage:
    channel_id: str
    text: str
    thread_id: str = ""
    reply_to: str = ""
    parse_mode: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Attachment:
    kind: str
    file_id: str
    mime_type: str = ""
    file_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SentMessage:
    message_id: str
    channel_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamHandle:
    channel_id: str
    handle_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Channel(Protocol):
    """Unified channel interface. Implementations declare capabilities via inner Meta class."""

    class Meta:
        name: str = ""
        stream_capability: StreamCapability = StreamCapability.BUFFER
        id_prefix: str = ""
        max_message_length: int = 0
        supports_edit: bool = False
        supports_delete: bool = False
        supports_react: bool = False
        supports_pin: bool = False
        supports_file: bool = False
        markdown_mode: str = ""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def receive(self) -> AsyncIterator[InboundMessage]: ...
    async def send_message(self, msg: OutboundMessage) -> SentMessage: ...
    async def edit_message(self, channel_id: str, message_id: str, text: str) -> None: ...
    async def delete_message(self, channel_id: str, message_id: str) -> None: ...
    async def send_typing(self, channel_id: str) -> None: ...
    async def start_stream(self, channel_id: str, *, thread_id: str = "", reply_to: str = "") -> StreamHandle: ...
    async def append_stream(self, handle: StreamHandle, text: str) -> None: ...
    async def stop_stream(self, handle: StreamHandle, final_text: str) -> SentMessage: ...
    async def send_file(self, channel_id: str, file_path: str, *, caption: str = "") -> SentMessage: ...
    async def get_file(self, file_id: str) -> Path: ...
    async def react(self, channel_id: str, message_id: str, emoji: str) -> None: ...
    async def pin(self, channel_id: str, message_id: str) -> None: ...
