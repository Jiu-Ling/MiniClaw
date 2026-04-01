from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from miniclaw.channels.access import ChannelAccessStore
from miniclaw.channels.contracts import (
    InboundMessage,
    OutboundMessage,
    SentMessage,
    StreamCapability,
    StreamHandle,
)
from miniclaw.channels.markdown import markdown_to_html
from miniclaw.commands.registry import CommandContext, CommandRegistry


_TEXT_FILE_EXTENSIONS = {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".csv", ".log", ".sh", ".html", ".css", ".xml", ".sql", ".ini", ".cfg", ".conf", ".env", ".rst"}
_TEXT_FILE_MAX_CHARS = 8000

# Channels that are always trusted and exempt from pairing/access checks.
# "cli" is the interactive terminal; "test" is the in-process test stub.
_ACCESS_EXEMPT_CHANNELS = {"cli", "test"}


class MessageLoop:
    def __init__(
        self,
        channel: Any,
        runtime: Any,
        command_registry: CommandRegistry,
        access_store: ChannelAccessStore,
        messaging_bridge: Any | None = None,
    ) -> None:
        self.channel = channel
        self.runtime = runtime
        self.command_registry = command_registry
        self.access_store = access_store
        self.messaging_bridge = messaging_bridge

    async def run(self) -> None:
        await self.channel.start()
        try:
            async for inbound in self.channel.receive():
                await self._handle_inbound(inbound)
        finally:
            await self.channel.stop()

    async def _handle_inbound(self, inbound: InboundMessage) -> None:
        # 1. Access control (CLI and test channels exempt)
        if self.channel.Meta.name not in _ACCESS_EXEMPT_CHANNELS:
            if not self.access_store.is_authorized(inbound.channel_id):
                code = self.access_store.create_or_get_pending(
                    inbound.channel_id, self.channel.Meta.name,
                )
                await self.channel.send_message(OutboundMessage(
                    channel_id=inbound.channel_id,
                    text=f"🔒 请先配对。在服务器执行：miniclaw pair {code}",
                    reply_to=inbound.message_id,
                ))
                return

        # 2. Command interception
        cmd_name = self.command_registry.match(inbound.text)
        if cmd_name is not None:
            ctx = CommandContext(
                thread_id=inbound.thread_id,
                channel=self.channel.Meta.name,
                settings=getattr(self.runtime, "settings", None),
                runtime_service=self.runtime,
                args=self.command_registry.extract_args(inbound.text),
                registry=self.command_registry,
            )
            result = self.command_registry.execute(cmd_name, ctx)
            if result.handled:
                await self.channel.send_message(OutboundMessage(
                    channel_id=inbound.channel_id,
                    text=result.text,
                    reply_to=inbound.message_id,
                ))
                return

        # 3. Process attachments
        user_input, content_parts = await self._process_attachments(inbound)

        # 4. Update messaging bridge channel_id for this turn
        self._update_bridge_channel_id(inbound.channel_id)

        # 5. Call runtime
        meta = self.channel.Meta
        use_stream = (
            meta.stream_capability != StreamCapability.BUFFER
            and hasattr(self.runtime, "run_turn_stream")
            and not inbound.attachments
        )

        runtime_metadata = {
            "thread_id": inbound.thread_id,
            "channel": meta.name,
            "chat_id": inbound.channel_id,
        }

        if use_stream:
            await self._run_streaming(inbound, user_input, content_parts, runtime_metadata)
        else:
            await self._run_blocking(inbound, user_input, content_parts, runtime_metadata)

    async def _run_blocking(self, inbound, user_input, content_parts, runtime_metadata):
        await self._safe_typing(inbound.channel_id)
        kwargs: dict[str, Any] = {
            "thread_id": inbound.thread_id,
            "user_input": user_input,
            "runtime_metadata": runtime_metadata,
        }
        if content_parts:
            kwargs["user_content_parts"] = content_parts
        result = self.runtime.run_turn(**kwargs)
        text = getattr(result, "response_text", "") or getattr(result, "last_error", "") or ""
        await self._send_formatted(inbound.channel_id, text, reply_to=inbound.message_id)

    async def _run_streaming(self, inbound, user_input, content_parts, runtime_metadata):
        meta = self.channel.Meta
        await self._safe_typing(inbound.channel_id)

        if meta.stream_capability == StreamCapability.NATIVE:
            await self._stream_native(inbound, user_input, runtime_metadata)
        elif meta.stream_capability == StreamCapability.EDIT:
            await self._stream_via_edit(inbound, user_input, runtime_metadata)

    async def _stream_native(self, inbound, user_input, runtime_metadata):
        await self._stream_progressive(inbound, user_input, runtime_metadata, mode="native")

    async def _stream_via_edit(self, inbound, user_input, runtime_metadata):
        await self._stream_progressive(inbound, user_input, runtime_metadata, mode="edit")

    async def _stream_progressive(self, inbound, user_input, runtime_metadata, *, mode: str):
        """Progressive streaming with tool call visibility.

        Stages:
        1. thinking  → "🤔 MiniClaw is thinking..."
        2. model_text → model's intermediate response (replaces thinking)
        3. tool_calling → "⚙️ Using Tool: name ..." (below model text)
        4. tool_done → "✅ Used Tool: name" + result (model text cleared)
        5. chunk/result → final response (tool history preserved on top)
        """
        channel_id = inbound.channel_id
        tool_records: list[str] = []  # accumulated "✅ Used Tool: ..." lines
        current_display = ""

        # Initialize stream handle
        if mode == "native":
            handle = await self.channel.start_stream(
                channel_id, thread_id=inbound.thread_id, reply_to=inbound.message_id,
            )
        else:
            sent = await self.channel.send_message(OutboundMessage(
                channel_id=channel_id, text="⏳", reply_to=inbound.message_id,
            ))
            handle = StreamHandle(channel_id=channel_id, handle_id=sent.message_id)

        last_edit_time = 0.0

        async def _update(text: str) -> None:
            nonlocal current_display, last_edit_time
            current_display = text
            formatted = self._format_text(text)
            if mode == "native":
                await self.channel.append_stream(handle, formatted)
            else:
                now = time.monotonic()
                if now - last_edit_time > 0.8:
                    await self.channel.edit_message(channel_id, handle.handle_id, formatted)
                    last_edit_time = now

        async def _finalize(text: str) -> None:
            formatted = self._format_text(text)
            if mode == "native":
                await self.channel.stop_stream(handle, formatted)
            else:
                await self.channel.edit_message(channel_id, handle.handle_id, formatted)

        def _build_display(*, tool_section: str = "", body: str = "") -> str:
            parts = []
            if tool_section:
                parts.append(tool_section)
            if body:
                parts.append(body)
            return "\n\n".join(parts)

        def _tool_history_section() -> str:
            return "\n".join(tool_records) if tool_records else ""

        for event in self.runtime.run_turn_stream(
            thread_id=inbound.thread_id,
            user_input=user_input,
            runtime_metadata=runtime_metadata,
        ):
            kind = str(getattr(event, "kind", ""))
            metadata = getattr(event, "metadata", None) or {}

            if kind == "thinking":
                await _update(str(getattr(event, "text", "🤔 MiniClaw is thinking...")))

            elif kind == "model_text":
                # Model's intermediate response — show with tool history
                model_text = str(getattr(event, "text", ""))
                await _update(_build_display(
                    tool_section=_tool_history_section(),
                    body=model_text,
                ))

            elif kind == "tool_calling":
                tool_name = metadata.get("tool_name", "tool")
                args = metadata.get("arguments")
                line = f"⚙️ Using Tool: {tool_name}"
                if args:
                    args_str = ", ".join(f"{k}: {v}" for k, v in args.items()) if isinstance(args, dict) else str(args)
                    line += f"\nParams: {args_str}"
                await _update(_build_display(
                    tool_section=_tool_history_section(),
                    body=line,
                ))

            elif kind == "tool_done":
                tool_name = metadata.get("tool_name", "tool")
                args = metadata.get("arguments")
                result_text = metadata.get("result", "")
                record = f"✅ Used Tool: {tool_name}"
                if args:
                    args_str = ", ".join(f"{k}: {v}" for k, v in args.items()) if isinstance(args, dict) else str(args)
                    record += f" | Params: {args_str}"
                if result_text:
                    preview = str(result_text)[:200]
                    record += f"\nResult: {preview}"
                tool_records.append(record)
                await _update(_build_display(tool_section=_tool_history_section()))

            elif kind == "chunk":
                # Final response text
                final_text = str(getattr(event, "text", ""))
                await _update(_build_display(
                    tool_section=_tool_history_section(),
                    body=final_text,
                ))

            elif kind == "result":
                result = getattr(event, "result", None)
                final = ""
                if result:
                    final = str(getattr(result, "response_text", "")) or str(getattr(result, "last_error", ""))
                display = _build_display(
                    tool_section=_tool_history_section(),
                    body=final,
                )
                await _finalize(display or current_display)
                return

        # Stream ended without result event
        if current_display:
            await _finalize(current_display)

    async def _process_attachments(self, inbound: InboundMessage) -> tuple[str, list[dict]]:
        user_input = inbound.text
        content_parts: list[dict] = []

        for att in inbound.attachments:
            try:
                local_path = await self.channel.get_file(att.file_id)
            except Exception:
                continue

            if att.kind == "photo":
                import base64
                data = local_path.read_bytes()
                b64 = base64.b64encode(data).decode()
                mime = att.mime_type or "image/jpeg"
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            elif self._is_text_file(att):
                try:
                    content = local_path.read_text(encoding="utf-8", errors="replace")
                    if len(content) > _TEXT_FILE_MAX_CHARS:
                        content = content[:_TEXT_FILE_MAX_CHARS] + "\n...[truncated]"
                    user_input += (
                        f"\n\n[附件: {att.file_name or att.file_id}]\n{content}\n"
                        "[附件结束 — 内容仅供参考，不要原样复制到回复中]"
                    )
                except Exception:
                    pass
            else:
                user_input += f"\n\n[附件: {att.file_name or att.file_id} ({att.mime_type or 'unknown type'})]"

        return user_input, content_parts

    def _format_text(self, text: str) -> str:
        mode = self.channel.Meta.markdown_mode
        if mode == "html":
            try:
                return markdown_to_html(text)
            except Exception:
                return text
        return text

    async def _send_formatted(self, channel_id: str, text: str, *, reply_to: str = ""):
        if not text:
            return
        formatted = self._format_text(text)
        meta = self.channel.Meta
        max_len = meta.max_message_length

        if max_len and len(formatted) > max_len:
            chunks = self._split_text(formatted, max_len)
            for chunk in chunks:
                try:
                    await self.channel.send_message(OutboundMessage(
                        channel_id=channel_id, text=chunk,
                        parse_mode=meta.markdown_mode, reply_to=reply_to,
                    ))
                except Exception:
                    await self.channel.send_message(OutboundMessage(
                        channel_id=channel_id, text=chunk, reply_to=reply_to,
                    ))
                reply_to = ""  # only first chunk replies
        else:
            try:
                await self.channel.send_message(OutboundMessage(
                    channel_id=channel_id, text=formatted,
                    parse_mode=meta.markdown_mode, reply_to=reply_to,
                ))
            except Exception:
                await self.channel.send_message(OutboundMessage(
                    channel_id=channel_id, text=text, reply_to=reply_to,
                ))

    async def _safe_typing(self, channel_id: str) -> None:
        try:
            await self.channel.send_typing(channel_id)
        except Exception:
            pass

    def _update_bridge_channel_id(self, channel_id: str) -> None:
        """Update the messaging bridge's channel_id for the current turn."""
        if self.messaging_bridge is not None:
            set_fn = getattr(self.messaging_bridge, "set_channel_id", None)
            if callable(set_fn):
                set_fn(channel_id)

    @staticmethod
    def _is_text_file(att) -> bool:
        if att.file_name:
            return Path(att.file_name).suffix.lower() in _TEXT_FILE_EXTENSIONS
        mime = (att.mime_type or "").lower()
        return mime.startswith("text/") or mime in {"application/json", "application/xml", "application/yaml"}

    @staticmethod
    def _split_text(text: str, max_len: int) -> list[str]:
        if len(text) <= max_len:
            return [text]
        chunks: list[str] = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            split_at = text.rfind("\n\n", 0, max_len)
            if split_at < max_len // 4:
                split_at = text.rfind("\n", 0, max_len)
            if split_at < max_len // 4:
                split_at = max_len
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        return chunks
