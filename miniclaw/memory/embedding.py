from __future__ import annotations

import httpx


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, dims: int, *, _transport: object | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dims = dims
        self._timeout = httpx.Timeout(connect=5.0, read=120.0, write=5.0, pool=5.0)
        self._transport = _transport  # For testing only

    async def embed(self, texts: list[str], *, batch_size: int = 16) -> list[list[float]]:
        """Embed texts in batches to avoid Ollama timeout on large inputs."""
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        kwargs: dict = {"base_url": self.base_url, "timeout": self._timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        async with httpx.AsyncClient(**kwargs) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await client.post(
                    "/api/embed",
                    json={"model": self.model, "input": batch},
                )
                response.raise_for_status()
                data = response.json()
                all_embeddings.extend(data["embeddings"])
        return all_embeddings

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]

    async def health_check(self) -> None:
        vectors = await self.embed(["ping"])
        actual_dims = len(vectors[0])
        if actual_dims != self.dims:
            raise RuntimeError(
                f"embedding dimension mismatch: expected {self.dims}, got {actual_dims}"
            )
