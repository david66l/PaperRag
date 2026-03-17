"""API-based embedding provider (OpenAI-compatible)."""

import numpy as np
import httpx

from app.core.logging import get_logger

from .base import BaseEmbeddingProvider

logger = get_logger(__name__)


class APIEmbeddingProvider(BaseEmbeddingProvider):
    """Calls an OpenAI-compatible /embeddings endpoint."""

    def __init__(self, api_url: str, api_key: str, model: str, dimension: int = 1536):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._dim = dimension

    @property
    def dim(self) -> int:
        return self._dim

    def _call_api(self, texts: list[str]) -> np.ndarray:
        url = f"{self._api_url}/embeddings"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self._model, "input": texts}

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        return np.array(embeddings, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        logger.info("API embedding %d documents", len(texts))
        # Batch in groups of 100
        all_embs = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embs.append(self._call_api(batch))
        return np.concatenate(all_embs, axis=0)

    def embed_query(self, query: str) -> np.ndarray:
        result = self._call_api([query])
        return result[0]
