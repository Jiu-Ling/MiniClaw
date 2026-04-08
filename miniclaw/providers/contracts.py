from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vision: bool = False


class ChatMessage(BaseModel):
    """Normalized message contract for chat providers."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    content_parts: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    name: str | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_message_shape(self) -> "ChatMessage":
        if self.content is not None and self.content_parts:
            raise ValueError("ChatMessage cannot set both content and content_parts")
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("tool messages require tool_call_id")
            has_content = bool(self.content and self.content.strip())
            if not has_content and not self.content_parts:
                raise ValueError("tool messages require content or content_parts")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid on tool messages")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages may carry tool_calls")
        return self


class ChatUsage(BaseModel):
    """Normalized token accounting for chat responses."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None


class ChatResponse(BaseModel):
    """Normalized response contract for runtime consumers."""

    model_config = ConfigDict(extra="forbid")

    content: str
    content_parts: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    provider: str
    usage: ChatUsage | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ChatProvider(Protocol):
    """Minimal shared chat provider contract for runtime integration."""

    capabilities: ProviderCapabilities

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ChatResponse: ...

    async def astream_text(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]: ...


def provider_supports_vision(provider: object) -> bool:
    capabilities = getattr(provider, "capabilities", None)
    if isinstance(capabilities, ProviderCapabilities):
        return capabilities.vision
    return bool(getattr(provider, "supports_vision", False))
