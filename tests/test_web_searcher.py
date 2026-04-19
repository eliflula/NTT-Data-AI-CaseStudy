from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.web_searcher import WebSearchResult, WebSearchService


@pytest.fixture()
def searcher() -> WebSearchService:
    with patch("src.web_searcher.TavilyClient"):
        return WebSearchService()


def _make_hit(title: str = "Test Title", url: str = "https://example.com", content: str = "Test content") -> dict:
    return {"title": title, "url": url, "content": content}


def test_search_returns_results(searcher: WebSearchService) -> None:
    searcher._client.search = MagicMock(return_value={"results": [
        _make_hit("Article 1", "https://a.com", "Content A"),
        _make_hit("Article 2", "https://b.com", "Content B"),
    ]})

    results = searcher.search("NTT DATA sustainability")

    assert len(results) == 2
    assert all(isinstance(r, WebSearchResult) for r in results)


def test_search_maps_fields_correctly(searcher: WebSearchService) -> None:
    searcher._client.search = MagicMock(return_value={"results": [
        _make_hit("My Title", "https://example.com", "My Content"),
    ]})

    results = searcher.search("query")

    assert results[0].title == "My Title"
    assert results[0].url == "https://example.com"
    assert results[0].content == "My Content"


def test_search_returns_empty_on_no_results(searcher: WebSearchService) -> None:
    searcher._client.search = MagicMock(return_value={"results": []})

    results = searcher.search("unknown query")

    assert results == []


def test_search_excludes_domains(searcher: WebSearchService) -> None:
    searcher._client.search = MagicMock(return_value={"results": []})

    searcher.search("query")

    call_kwargs = searcher._client.search.call_args[1]
    excluded = call_kwargs["exclude_domains"]
    assert "instagram.com" in excluded
    assert "twitter.com" in excluded
    assert "eksisozluk.com" in excluded


def test_build_context_formats_results(searcher: WebSearchService) -> None:
    results = [
        WebSearchResult("Title A", "https://a.com", "Content A"),
        WebSearchResult("Title B", "https://b.com", "Content B"),
    ]

    context = searcher.build_context(results)

    assert "[Web Search Results]" in context
    assert "Title A" in context
    assert "Content A" in context
    assert "---" in context


def test_build_context_empty_returns_header(searcher: WebSearchService) -> None:
    context = searcher.build_context([])

    assert "[Web Search Results]" in context
