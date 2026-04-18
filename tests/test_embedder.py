from __future__ import annotations
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from src.embedder import EmbeddingService


@pytest.fixture()
def embedder() -> EmbeddingService:
    return EmbeddingService()


def test_embed_returns_list_of_floats(embedder: EmbeddingService) -> None:
    mock_vector = np.array([0.1, 0.2, 0.3])
    embedder._model.encode = MagicMock(return_value=mock_vector)

    result = embedder.embed("test text")

    assert isinstance(result, list)
    assert result == pytest.approx([0.1, 0.2, 0.3])


def test_embed_batch_returns_list_of_vectors(embedder: EmbeddingService) -> None:
    mock_vectors = np.array([[0.1, 0.2], [0.3, 0.4]])
    embedder._model.encode = MagicMock(return_value=mock_vectors)

    result = embedder.embed_batch(["text1", "text2"])

    assert len(result) == 2
    assert result[0] == pytest.approx([0.1, 0.2])
    assert result[1] == pytest.approx([0.3, 0.4])


def test_embed_calls_model_with_normalization(embedder: EmbeddingService) -> None:
    embedder._model.encode = MagicMock(return_value=np.array([0.5]))

    embedder.embed("hello")

    embedder._model.encode.assert_called_once_with("hello", normalize_embeddings=True)


def test_embed_batch_calls_model_with_correct_args(embedder: EmbeddingService) -> None:
    embedder._model.encode = MagicMock(return_value=np.array([[0.1], [0.2]]))

    embedder.embed_batch(["a", "b"])

    call_kwargs = embedder._model.encode.call_args[1]
    assert call_kwargs["normalize_embeddings"] is True
    assert call_kwargs["show_progress_bar"] is False
