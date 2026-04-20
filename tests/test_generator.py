from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from src.generator import GenerateResult, GeneratorService


@pytest.fixture()
def generator() -> GeneratorService:
    return GeneratorService()


def _make_chunk(content: str, source: str = "report_2024", year: int = 2024, page: int = 1) -> MagicMock:
    chunk = MagicMock()
    chunk.payload = {"content": content, "source": source, "year": year, "page": page}
    chunk.score = 0.85
    return chunk


def test_build_context_formats_chunks_correctly(generator: GeneratorService) -> None:
    chunks = [_make_chunk("NTT DATA targets carbon neutrality.")]
    context = generator.build_context(chunks)

    assert "NTT DATA targets carbon neutrality." in context


def test_build_context_separates_multiple_chunks(generator: GeneratorService) -> None:
    chunks = [_make_chunk("First chunk"), _make_chunk("Second chunk")]
    context = generator.build_context(chunks)

    assert "First chunk" in context
    assert "Second chunk" in context
    assert "---" in context


def test_build_context_empty_list_returns_empty_string(generator: GeneratorService) -> None:
    assert generator.build_context([]) == ""


def test_generate_returns_generate_result(generator: GeneratorService) -> None:
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 50
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 70

    mock_message = MagicMock()
    mock_message.content = "Carbon neutrality by 2030."

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    generator._client.chat.completions.create = MagicMock(return_value=mock_response)

    result = generator.generate("What is the carbon target?", "some context")

    assert isinstance(result, GenerateResult)
    assert result.answer == "Carbon neutrality by 2030."
    assert result.prompt_tokens == 50
    assert result.completion_tokens == 20
    assert result.total_tokens == 70


def test_generate_passes_question_and_context_to_llm(generator: GeneratorService) -> None:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "answer"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    generator._client.chat.completions.create = MagicMock(return_value=mock_response)

    generator.generate("my question", "my context")

    call_kwargs = generator._client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    user_message = next(m for m in messages if m["role"] == "user")
    assert "my question" in user_message["content"]
    assert "my context" in user_message["content"]
