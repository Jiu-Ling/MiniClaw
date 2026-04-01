from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from miniclaw.channels.contracts import (
    InboundMessage,
    OutboundMessage,
    SentMessage,
    StreamCapability,
    StreamHandle,
)


class CLIChannel:
    class Meta:
        name = "cli"
        stream_capability = StreamCapability.BUFFER
        id_prefix = "cli"
        max_message_length = 0
        supports_edit = False
        supports_delete = False
        supports_react = False
        supports_pin = False
        supports_file = True
        markdown_mode = ""

    def __init__(self, thread_id: str = "default") -> None:
        self._console = Console()
        self._thread_id = thread_id
        self._live: Live | None = None
        self._msg_counter = 0

    async def start(self) -> None:
        self._console.print("[bold blue]MiniClaw REPL[/bold blue]")

    async def stop(self) -> None:
        self._console.print("[dim]Session ended.[/dim]")

    async def receive(self) -> AsyncIterator[InboundMessage]:
        while True:
            try:
                text = await asyncio.to_thread(self._console.input, "[bold green]> [/]")
            except (EOFError, KeyboardInterrupt):
                return
            text = text.strip()
            if not text:
                continue
            if text.lower() in {"/exit", "/quit"}:
                return
            self._msg_counter += 1
            yield InboundMessage(
                channel="cli",
                channel_id=f"cli:{self._thread_id}",
                thread_id=f"cli:{self._thread_id}",
                message_id=f"cli-{self._msg_counter}",
                text=text,
            )

    async def send_message(self, msg: OutboundMessage) -> SentMessage:
        self._console.print(Markdown(msg.text))
        self._msg_counter += 1
        return SentMessage(message_id=f"cli-{self._msg_counter}", channel_id=msg.channel_id)

    async def start_stream(self, channel_id: str, *, thread_id: str = "", reply_to: str = "") -> StreamHandle:
        self._live = Live(console=self._console, refresh_per_second=10)
        self._live.start()
        return StreamHandle(channel_id=channel_id, handle_id="cli-stream")

    async def append_stream(self, handle: StreamHandle, text: str) -> None:
        if self._live:
            self._live.update(Markdown(text))

    async def stop_stream(self, handle: StreamHandle, final_text: str) -> SentMessage:
        if self._live:
            self._live.update(Markdown(final_text))
            self._live.stop()
            self._live = None
        self._msg_counter += 1
        return SentMessage(message_id=f"cli-{self._msg_counter}", channel_id=handle.channel_id)

    async def send_file(self, channel_id: str, file_path: str, *, caption: str = "") -> SentMessage:
        self._console.print(f"📎 [bold]{file_path}[/bold]")
        if caption:
            self._console.print(f"   {caption}")
        self._msg_counter += 1
        return SentMessage(message_id=f"cli-{self._msg_counter}", channel_id=channel_id)

    async def get_file(self, file_id: str) -> Path:
        return Path(file_id)

    async def edit_message(self, channel_id: str, message_id: str, text: str) -> None:
        pass

    async def delete_message(self, channel_id: str, message_id: str) -> None:
        pass

    async def send_typing(self, channel_id: str) -> None:
        pass

    async def react(self, channel_id: str, message_id: str, emoji: str) -> None:
        pass

    async def pin(self, channel_id: str, message_id: str) -> None:
        pass
