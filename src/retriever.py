from __future__ import annotations
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from src.config import settings
from src.embedder import EmbeddingService


class RetrieverService:
    """Queries Qdrant for semantically similar document chunks."""

    def __init__(self, embedder: EmbeddingService) -> None:
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._embedder = embedder

    def ping(self) -> bool:
        """Return True if Qdrant is reachable and the collection exists."""
        try:
            return self._client.collection_exists(settings.collection)
        except Exception:
            return False

    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredPoint]:
        """Return the top-k most relevant chunks for a given question."""
        vector = self._embedder.embed(question)
        response = self._client.query_points(
            collection_name=settings.collection,
            query=vector,
            limit=top_k or settings.top_k,
            with_payload=True,
        )
        return response.points
