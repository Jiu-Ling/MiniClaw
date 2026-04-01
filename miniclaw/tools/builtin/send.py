from __future__ import annotations

from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.messaging import MessagingBridge, OutboundMessage
from miniclaw.tools.registry import RegisteredTool


def build_send_tool(*, messaging_bridge: MessagingBridge | None) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        text = str(call.arguments.get("text", "")).strip()
        file_path = str(call.arguments.get("file_path", "")).strip()
        caption = str(call.arguments.get("caption", "")).strip()

        if not text and not file_path:
            return ToolResult(content="text or file_path is required", is_error=True)

        if messaging_bridge is None:
            return ToolResult(content="messaging bridge is not configured", is_error=True)

        try:
            if file_path:
                result = messaging_bridge.send_file(file_path, caption=caption or text)
            else:
                result = messaging_bridge.send_message(OutboundMessage(text=text))
        except Exception as exc:
            return ToolResult(content=f"send failed: {exc}", is_error=True)

        if not isinstance(result, dict):
            return ToolResult(content="bridge returned invalid result", is_error=True)
        if not result.get("ok"):
            return ToolResult(content=str(result.get("error", "send failed")), is_error=True)

        content = str(result.get("message_id", "sent"))
        return ToolResult(content=content, metadata=dict(result))

    return RegisteredTool(
        spec=ToolSpec(
            name="send",
            description="Send a message or file to the current channel.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Message text"},
                    "file_path": {"type": "string", "description": "Local file path to send (optional)"},
                    "caption": {"type": "string", "description": "Caption for the file (optional)"},
                },
                "additionalProperties": False,
            },
            source="builtin",
        ),
        executor=execute,
    )


__all__ = ["build_send_tool"]
