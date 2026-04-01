from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@runtime_checkable
class SearchBackend(Protocol):
    def search(self, query: str, limit: int = 5) -> list[SearchResult]: ...


@dataclass(slots=True)
class SearchBackendConfig:
    provider: str = "none"
    api_key: str | None = None
    base_url: str | None = None
    proxy: str | None = None
    max_results: int = 5


class RemoteSearchBackend:
    def __init__(self, config: SearchBackendConfig) -> None:
        self.config = config

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        provider = self.config.provider.strip().lower()
        resolved_limit = max(1, min(limit or self.config.max_results, 10))
        if provider == "brave":
            return self._search_brave(query, resolved_limit)
        if provider == "tavily":
            return self._search_tavily(query, resolved_limit)
        if provider == "searxng":
            return self._search_searxng(query, resolved_limit)
        if provider == "jina":
            return self._search_jina(query, resolved_limit)
        raise RuntimeError(f"unsupported search provider: {provider or '<empty>'}")

    def _search_brave(self, query: str, limit: int) -> list[SearchResult]:
        api_key = self.config.api_key or os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            raise RuntimeError("BRAVE_API_KEY is not configured")
        data = self._request_json(
            "GET",
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            query={"q": query, "count": limit},
        )
        items = data.get("web", {}).get("results", [])
        return [
            SearchResult(
                title=str(item.get("title", "")).strip(),
                url=str(item.get("url", "")).strip(),
                snippet=str(item.get("description", "")).strip(),
            )
            for item in items
        ]

    def _search_tavily(self, query: str, limit: int) -> list[SearchResult]:
        api_key = self.config.api_key or os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured")
        data = self._request_json(
            "POST",
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            body={"query": query, "max_results": limit},
        )
        return [
            SearchResult(
                title=str(item.get("title", "")).strip(),
                url=str(item.get("url", "")).strip(),
                snippet=str(item.get("content", "")).strip(),
            )
            for item in data.get("results", [])
        ]

    def _search_searxng(self, query: str, limit: int) -> list[SearchResult]:
        base_url = (self.config.base_url or os.environ.get("SEARXNG_BASE_URL", "")).strip()
        if not base_url:
            raise RuntimeError("SEARXNG_BASE_URL is not configured")
        data = self._request_json(
            "GET",
            f"{base_url.rstrip('/')}/search",
            headers={"Accept": "application/json", "User-Agent": "MiniClaw/1.0"},
            query={"q": query, "format": "json"},
        )
        return [
            SearchResult(
                title=str(item.get("title", "")).strip(),
                url=str(item.get("url", "")).strip(),
                snippet=str(item.get("content", "")).strip(),
            )
            for item in data.get("results", [])[:limit]
        ]

    def _search_jina(self, query: str, limit: int) -> list[SearchResult]:
        api_key = self.config.api_key or os.environ.get("JINA_API_KEY", "")
        if not api_key:
            raise RuntimeError("JINA_API_KEY is not configured")
        data = self._request_json(
            "GET",
            "https://s.jina.ai/",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            query={"q": query},
        )
        items = data.get("data", [])[:limit]
        return [
            SearchResult(
                title=str(item.get("title", "")).strip(),
                url=str(item.get("url", "")).strip(),
                snippet=str(item.get("content", "")).strip(),
            )
            for item in items
        ]

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        query: dict[str, object] | None = None,
        body: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        resolved_url = url
        if query:
            resolved_url = f"{url}?{urlencode({key: value for key, value in query.items() if value is not None})}"
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
        request = Request(
            resolved_url,
            data=payload,
            headers=headers or {},
            method=method,
        )
        with urlopen(request, timeout=15) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("search backend returned a non-object JSON payload")
        return parsed


def build_search_backend(settings: object | None) -> SearchBackend | None:
    if settings is None:
        return None
    provider = str(getattr(settings, "search_provider", "none")).strip().lower()
    if not provider or provider == "none":
        return None
    return RemoteSearchBackend(
        SearchBackendConfig(
            provider=provider,
            api_key=_optional_str(getattr(settings, "search_api_key", None)),
            base_url=_optional_str(getattr(settings, "search_base_url", None)),
            proxy=_optional_str(getattr(settings, "search_proxy", None)),
            max_results=int(getattr(settings, "search_max_results", 5) or 5),
        )
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "RemoteSearchBackend",
    "SearchBackend",
    "SearchBackendConfig",
    "SearchResult",
    "build_search_backend",
]
