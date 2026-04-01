from __future__ import annotations

import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import telegram
import telegram.error

from miniclaw.channels.contracts import (
    Attachment,
    InboundMessage,
    OutboundMessage,
    SentMessage,
    StreamCapability,
    StreamHandle,
)


class TelegramChannel:
    class Meta:
        name = "telegram"
        stream_capability = StreamCapability.NATIVE
        id_prefix = "tg"
        max_message_length = 4096
        supports_edit = True
        supports_delete = True
        supports_react = True
        supports_pin = True
        supports_file = True
        markdown_mode = "html"

    def __init__(self, bot_token: str, *, poll_timeout: int = 30) -> None:
        self._bot_token = bot_token
        self._poll_timeout = poll_timeout
        self._bot: telegram.Bot | None = None
        self._offset: int | None = None
        self._draft_counter = 0

    async def start(self) -> None:
        self._bot = telegram.Bot(token=self._bot_token)
        await self._bot.initialize()

    async def stop(self) -> None:
        if self._bot:
            await self._bot.shutdown()
            self._bot = None

    async def receive(self) -> AsyncIterator[InboundMessage]:
        if self._bot is None:
            return
        while True:
            try:
                updates = await self._bot.get_updates(
                    offset=self._offset,
                    timeout=self._poll_timeout,
                )
            except telegram.error.TimedOut:
                continue
            except Exception:
                continue
            for update in updates:
                self._offset = update.update_id + 1
                inbound = self._resolve_update(update)
                if inbound is not None:
                    yield inbound

    async def send_message(self, msg: OutboundMessage) -> SentMessage:
        chat_id = self._to_chat_id(msg.channel_id)
        thread_id = self._extract_message_thread_id(msg.channel_id)
        reply_to = int(msg.reply_to) if msg.reply_to and msg.reply_to.isdigit() else None

        kwargs: dict = {"chat_id": chat_id, "text": msg.text}
        if thread_id:
            kwargs["message_thread_id"] = thread_id
        if reply_to:
            kwargs["reply_to_message_id"] = reply_to
        if msg.parse_mode:
            kwargs["parse_mode"] = msg.parse_mode

        try:
            sent = await self._bot.send_message(**kwargs)
        except telegram.error.BadRequest:
            kwargs.pop("parse_mode", None)
            sent = await self._bot.send_message(**kwargs)

        return SentMessage(message_id=str(sent.message_id), channel_id=msg.channel_id)

    async def edit_message(self, channel_id: str, message_id: str, text: str) -> None:
        chat_id = self._to_chat_id(channel_id)
        try:
            await self._bot.edit_message_text(
                chat_id=chat_id, message_id=int(message_id),
                text=text, parse_mode="HTML",
            )
        except telegram.error.BadRequest:
            try:
                await self._bot.edit_message_text(
                    chat_id=chat_id, message_id=int(message_id), text=text,
                )
            except Exception:
                pass

    async def delete_message(self, channel_id: str, message_id: str) -> None:
        try:
            await self._bot.delete_message(
                chat_id=self._to_chat_id(channel_id), message_id=int(message_id),
            )
        except Exception:
            pass

    async def send_typing(self, channel_id: str) -> None:
        try:
            await self._bot.send_chat_action(
                chat_id=self._to_chat_id(channel_id), action="typing",
            )
        except Exception:
            pass

    async def start_stream(self, channel_id: str, *, thread_id: str = "", reply_to: str = "") -> StreamHandle:
        self._draft_counter += 1
        draft_id = self._draft_counter
        chat_id = self._to_chat_id(channel_id)
        msg_thread_id = self._extract_message_thread_id(channel_id)

        kwargs: dict = {"chat_id": chat_id, "draft_id": draft_id, "text": "..."}
        if msg_thread_id:
            kwargs["message_thread_id"] = msg_thread_id
        if reply_to and reply_to.isdigit():
            kwargs["reply_to_message_id"] = int(reply_to)

        try:
            await self._bot.send_message_draft(**kwargs)
        except Exception:
            pass

        return StreamHandle(
            channel_id=channel_id, handle_id=str(draft_id),
            metadata={"chat_id": chat_id, "draft_id": draft_id},
        )

    async def append_stream(self, handle: StreamHandle, text: str) -> None:
        chat_id = handle.metadata.get("chat_id")
        draft_id = handle.metadata.get("draft_id")
        if not chat_id or not draft_id:
            return
        try:
            await self._bot.send_message_draft(
                chat_id=chat_id, draft_id=draft_id, text=text, parse_mode="HTML",
            )
        except Exception:
            try:
                await self._bot.send_message_draft(
                    chat_id=chat_id, draft_id=draft_id, text=text,
                )
            except Exception:
                pass

    async def stop_stream(self, handle: StreamHandle, final_text: str) -> SentMessage:
        sent = await self.send_message(OutboundMessage(
            channel_id=handle.channel_id, text=final_text, parse_mode="HTML",
        ))
        chat_id = handle.metadata.get("chat_id")
        draft_id = handle.metadata.get("draft_id")
        if chat_id and draft_id:
            try:
                await self._bot.delete_message_draft(chat_id=chat_id, draft_id=draft_id)
            except Exception:
                pass
        return sent

    async def send_file(self, channel_id: str, file_path: str, *, caption: str = "") -> SentMessage:
        chat_id = self._to_chat_id(channel_id)
        path = Path(file_path)
        suffix = path.suffix.lower()
        with open(path, "rb") as f:
            if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                sent = await self._bot.send_photo(chat_id=chat_id, photo=f, caption=caption or None)
            else:
                sent = await self._bot.send_document(chat_id=chat_id, document=f, caption=caption or None)
        return SentMessage(message_id=str(sent.message_id), channel_id=channel_id)

    async def get_file(self, file_id: str) -> Path:
        tg_file = await self._bot.get_file(file_id)
        tmp_dir = Path(tempfile.mkdtemp(prefix="miniclaw_"))
        local_path = tmp_dir / (tg_file.file_unique_id or str(uuid.uuid4()))
        await tg_file.download_to_drive(local_path)
        return local_path

    async def react(self, channel_id: str, message_id: str, emoji: str) -> None:
        try:
            await self._bot.set_message_reaction(
                chat_id=self._to_chat_id(channel_id), message_id=int(message_id),
                reaction=[telegram.ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception:
            pass

    async def pin(self, channel_id: str, message_id: str) -> None:
        try:
            await self._bot.pin_chat_message(
                chat_id=self._to_chat_id(channel_id), message_id=int(message_id),
            )
        except Exception:
            pass

    def _resolve_update(self, update: telegram.Update) -> InboundMessage | None:
        message = update.message or update.edited_message
        if message is None:
            return None
        chat = message.chat
        if chat is None:
            return None
        text = message.text or message.caption or ""
        attachments = self._extract_attachments(message)
        if not text and not attachments:
            return None
        chat_id = chat.id
        msg_thread_id = message.message_thread_id
        channel_id = self._to_channel_id(chat_id, thread_id=msg_thread_id)
        return InboundMessage(
            channel="telegram",
            channel_id=channel_id,
            thread_id=channel_id,
            message_id=str(message.message_id),
            text=text,
            sender_id=str(message.from_user.id) if message.from_user else "",
            attachments=attachments,
            metadata={"update_id": update.update_id, "chat_type": chat.type},
        )

    def _extract_attachments(self, message: telegram.Message) -> list[Attachment]:
        attachments: list[Attachment] = []
        if message.photo:
            largest = max(message.photo, key=lambda p: p.width * p.height)
            attachments.append(Attachment(
                kind="photo", file_id=largest.file_id,
                metadata={"width": largest.width, "height": largest.height},
            ))
        if message.document:
            doc = message.document
            attachments.append(Attachment(
                kind="document", file_id=doc.file_id,
                mime_type=doc.mime_type or "", file_name=doc.file_name or "",
            ))
        return attachments

    @staticmethod
    def _to_chat_id(channel_id: str) -> int:
        return int(channel_id.removeprefix("tg:").split(":")[0])

    @staticmethod
    def _to_channel_id(chat_id: int, thread_id: int | None = None) -> str:
        if thread_id:
            return f"tg:{chat_id}:{thread_id}"
        return f"tg:{chat_id}"

    @staticmethod
    def _extract_message_thread_id(channel_id: str) -> int | None:
        parts = channel_id.removeprefix("tg:").split(":")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        return None
