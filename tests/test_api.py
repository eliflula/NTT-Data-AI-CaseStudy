from __future__ import annotations
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], float)


@patch("src.api._pipeline")
def test_ask_rag_response(mock_pipeline: MagicMock) -> None:
    mock_chunk = MagicMock()
    mock_chunk.payload = {
        "source": "NTT-DATA_Sustainability-Report_2024_CaseBook",
        "year": 2024,
        "page": 5,
        "chunk_id": "chunk_0001",
        "content": "NTT DATA targets carbon neutrality by 2040.",
    }
    mock_chunk.score = 0.92

    mock_pipeline.ask.return_value = (
        "NTT DATA targets carbon neutrality by 2040.",
        "rag",
        [mock_chunk],
        [],
    )

    response = client.post("/ask", json={"question": "What is the carbon target?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "NTT DATA targets carbon neutrality by 2040."
    assert data["source_type"] == "rag"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["year"] == 2024
    assert data["sources"][0]["score"] == 0.92


@patch("src.api._pipeline")
def test_ask_web_response(mock_pipeline: MagicMock) -> None:
    mock_pipeline.ask.return_value = (
        "The president is determined by election.",
        "web",
        [],
        [{"title": "Example", "url": "https://example.com"}],
    )

    response = client.post("/ask", json={"question": "Who is the president?"})

    assert response.status_code == 200
    data = response.json()
    assert data["source_type"] == "web"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["url"] == "https://example.com"


def test_ask_empty_question_returns_400() -> None:
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_ask_missing_field_returns_422() -> None:
    response = client.post("/ask", json={})
    assert response.status_code == 422