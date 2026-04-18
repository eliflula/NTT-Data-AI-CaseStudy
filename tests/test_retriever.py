from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from src.retriever import RetrieverService


@pytest.fixture()
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 1024
    return embedder


@pytest.fixture()
def retriever(mock_embedder: MagicMock) -> RetrieverService:
    return RetrieverService(mock_embedder)


def test_ping_returns_true_when_collection_exists(retriever: RetrieverService) -> None:
    retriever._client.collection_exists.return_value = True
    assert retriever.ping() is True


def test_ping_returns_false_when_collection_missing(retriever: RetrieverService) -> None:
    retriever._client.collection_exists.return_value = False
    assert retriever.ping() is False


def test_ping_returns_false_on_exception(retriever: RetrieverService) -> None:
    retriever._client.collection_exists.side_effect = Exception("connection refused")
    assert retriever.ping() is False


def test_retrieve_calls_embedder_and_qdrant(retriever: RetrieverService) -> None:
    mock_point = MagicMock()
    mock_response = MagicMock()
    mock_response.points = [mock_point]
    retriever._client.query_points.return_value = mock_response

    result = retriever.retrieve("carbon emissions?", top_k=3)

    retriever._embedder.embed.assert_called_once_with("carbon emissions?")
    retriever._client.query_points.assert_called_once()
    assert result == [mock_point]


def test_retrieve_uses_default_top_k(retriever: RetrieverService) -> None:
    mock_response = MagicMock()
    mock_response.points = []
    retriever._client.query_points.return_value = mock_response

    retriever.retrieve("question")

    call_kwargs = retriever._client.query_points.call_args[1]
    assert "limit" in call_kwargs
