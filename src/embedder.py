from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


class EmbeddingService:
    """Wraps a SentenceTransformer model for text embedding."""

    def __init__(self) -> None:
        self._model = SentenceTransformer(settings.embedding_model)

    def embed(self, text: str) -> list[float]:
        """Embed a single string and return a normalized vector."""
        vector: np.ndarray = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings and return normalized vectors."""
        vectors: np.ndarray = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
