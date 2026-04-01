from __future__ import annotations

from typing import Any

from miniclaw.channels.contracts import OutboundMessage as ChannelOutboundMessage, SentMessage
from miniclaw.tools.messaging import MessagingBridge, OutboundMessage as ToolOutboundMessage
from miniclaw.utils.async_bridge import run_sync


class ChannelMessagingBridge:
    """Adapts a Channel to the MessagingBridge protocol for tool use.

    channel_id can be set dynamically per-turn via set_channel_id(),
    or passed at construction time for fixed-channel scenarios.
    """

    def __init__(self, channel: Any, channel_id: str = "") -> None:
        self._channel = channel
        self._channel_id = channel_id

    def set_channel_id(self, channel_id: str) -> None:
        self._channel_id = channel_id

    def send_message(self, message: ToolOutboundMessage) -> dict[str, object]:
        channel_id = self._resolve_channel_id()
        outbound = ChannelOutboundMessage(channel_id=channel_id, text=message.text)
        result: SentMessage = run_sync(self._channel.send_message(outbound))
        return {"ok": True, "message_id": result.message_id}

    def send_file(self, file_path: str, *, caption: str = "") -> dict[str, object]:
        channel_id = self._resolve_channel_id()
        result: SentMessage = run_sync(self._channel.send_file(channel_id, file_path, caption=caption))
        return {"ok": True, "message_id": result.message_id}

    def _resolve_channel_id(self) -> str:
        if not self._channel_id:
            raise RuntimeError("messaging bridge channel_id not set — call set_channel_id() first")
        return self._channel_id
