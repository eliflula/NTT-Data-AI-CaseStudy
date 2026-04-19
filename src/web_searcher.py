from __future__ import annotations

import logging

from tavily import TavilyClient

from src.config import settings

logger = logging.getLogger(__name__)

_EXCLUDE_DOMAINS = [
    "instagram.com", "twitter.com", "x.com",
    "eksisozluk.com", "reddit.com", "facebook.com",
    "tiktok.com", "youtube.com",
]


class WebSearchResult:
    def __init__(self, title: str, url: str, content: str) -> None:
        self.title = title
        self.url = url
        self.content = content


class WebSearchService:
    def __init__(self) -> None:
        self._client = TavilyClient(api_key=settings.tavily_api_key)

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        response = self._client.search(
            query,
            max_results=max_results,
            include_raw_content=False,
            #search_depth="advanced",
            exclude_domains=_EXCLUDE_DOMAINS,
        )
        hits = response.get("results", [])
        logger.info("[WEB] search: %d results for query=%s", len(hits), query)
        for i, h in enumerate(hits, 1):
            logger.info("[WEB]   [%d] %s", i, h.get("url", ""))
        return [
            WebSearchResult(
                title=h.get("title", ""),
                url=h.get("url", ""),
                content=h.get("content", ""),
            )
            for h in hits
        ]

    def build_context(self, results: list[WebSearchResult]) -> str:
        parts = [
            f"[Result {i}] {r.title}\n{r.content}"
            for i, r in enumerate(results, 1)
        ]
        return "[Web Search Results]\n\n" + "\n\n---\n\n".join(parts)
