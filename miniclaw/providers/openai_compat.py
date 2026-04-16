from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from openai import AsyncOpenAI

from miniclaw.providers.contracts import ChatMessage, ChatProvider, ChatResponse, ChatUsage, ProviderCapabilities


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _dump(source: Any) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    dump = getattr(source, "model_dump", None)
    if callable(dump):
        return dict(dump())
    if hasattr(source, "__dict__"):
        return dict(vars(source))
    return {}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_value(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _normalize_value(dump())
    if hasattr(value, "__dict__"):
        return {
            key: _normalize_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _first_choice(response: Any) -> Any:
    choices = _value(response, "choices", None)
    if not choices:
        return None
    return choices[0]


def _normalize_content_parts(content: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content, []
    if content is None:
        return "", []
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        normalized_parts = [_normalize_value(part) for part in content]
        text_chunks: list[str] = []
        for part in normalized_parts:
            if isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    text_chunks.append(text)
        return "".join(text_chunks), [part for part in normalized_parts if isinstance(part, dict)]
    normalized = _normalize_value(content)
    if isinstance(normalized, Mapping):
        text = normalized.get("text")
        return (text if isinstance(text, str) else ""), [dict(normalized)]
    return "", [normalized] if isinstance(normalized, dict) else []


def _serialize_message(message: ChatMessage, *, cache_control: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role}
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.content_parts:
        parts = [dict(part) for part in message.content_parts]
        if cache_control and parts:
            parts[-1] = {**parts[-1], "cache_control": {"type": "ephemeral"}}
        payload["content"] = parts
    elif message.content is not None or message.tool_calls:
        if cache_control and message.content:
            payload["content"] = [
                {"type": "text", "text": message.content, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            payload["content"] = message.content
    if message.tool_calls:
        payload["tool_calls"] = [_normalize_value(call) for call in message.tool_calls]
    return payload


def _parse_usage(usage_source: Any) -> ChatUsage:
    details = _value(usage_source, "prompt_tokens_details", None)
    cached_tokens = None
    cache_creation_tokens = None
    if details is not None:
        cached_tokens = _value(details, "cached_tokens", None)
        cache_creation_tokens = _value(details, "cache_creation_input_tokens", None)
    return ChatUsage(
        prompt_tokens=_value(usage_source, "prompt_tokens", None),
        completion_tokens=_value(usage_source, "completion_tokens", None),
        total_tokens=_value(usage_source, "total_tokens", None),
        cached_tokens=cached_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )


class OpenAICompatibleProvider(ChatProvider):
    """Thin async adapter for OpenAI-compatible chat completions."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        supports_vision: bool = False,
        enable_prompt_cache: bool = False,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.enable_prompt_cache = enable_prompt_cache
        self.capabilities = ProviderCapabilities(vision=supports_vision)
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        first_system_seen = False
        request_messages = []
        for msg in messages:
            apply_cache = False
            if self.enable_prompt_cache and msg.role == "system" and not first_system_seen:
                apply_cache = True
                first_system_seen = True
            request_messages.append(_serialize_message(msg, cache_control=apply_cache))
        kwargs: dict[str, Any] = {"model": model or self.model, "messages": request_messages}
        if tools:
            kwargs["tools"] = list(tools)
        response = await self._client.chat.completions.create(**kwargs)

        response_data = _dump(response)
        choice = _first_choice(response)
        if choice is None:
            raise ValueError("OpenAI-compatible provider returned no usable assistant message: no choices")
        message = _value(choice, "message", None)
        if message is None:
            raise ValueError("OpenAI-compatible provider returned no usable assistant message: missing message")
        if _value(message, "role", None) != "assistant":
            raise ValueError("OpenAI-compatible provider response role must be assistant")
        content, content_parts = _normalize_content_parts(_value(message, "content", ""))
        tool_calls = _value(message, "tool_calls", None) or []
        if not content and not content_parts and not tool_calls:
            raise ValueError("OpenAI-compatible provider returned no usable assistant message: empty message")
        normalized_tool_calls = [
            _normalize_value(call) for call in tool_calls
        ]

        usage_source = _value(response, "usage", None)
        usage = _parse_usage(usage_source) if usage_source is not None else None

        return ChatResponse(
            content=content,
            content_parts=[part for part in content_parts if isinstance(part, dict)],
            tool_calls=[call for call in normalized_tool_calls if isinstance(call, dict)],
            model=_value(response, "model", model or self.model),
            provider="openai-compatible",
            usage=usage,
            raw=response_data,
        )

    async def achat_structured(
        self,
        messages: Sequence[Any],
        *,
        schema: type,
        model: str | None = None,
    ) -> Any:
        """Return a Pydantic model instance using langchain-openai's with_structured_output.

        Constructs a ChatOpenAI with this provider's credentials, applies
        with_structured_output(schema), and invokes with the given messages.
        The schema should be a Pydantic BaseModel subclass.
        """
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=model or self.model,
        )
        structured = llm.with_structured_output(schema)
        # Normalize messages to dicts if they're ChatMessage objects
        normalized = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized.append(msg)
            else:
                normalized.append({"role": getattr(msg, "role", "user"), "content": getattr(msg, "content", "")})
        return await structured.ainvoke(normalized)

    async def astream_text(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        request_messages = [_serialize_message(message) for message in messages]
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": request_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = list(tools)
        stream = await self._client.chat.completions.create(**kwargs)

        async for chunk in stream:
            choices = _value(chunk, "choices", None) or []
            if not choices:
                continue
            delta = _value(choices[0], "delta", None)
            if delta is None:
                continue
            if _value(delta, "tool_calls", None):
                raise RuntimeError("streaming tool calls are not supported")
            text = _value(delta, "content", None)
            if isinstance(text, str) and text:
                yield text
