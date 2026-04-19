from __future__ import annotations
from unittest.mock import MagicMock, patch
_patches: list = []


def pytest_configure(config: object) -> None:  # noqa: ARG001
    """Start patches before test collection so module-level code is mocked."""
    targets = [
        "sentence_transformers.SentenceTransformer",
        "sentence_transformers.CrossEncoder",
        "qdrant_client.QdrantClient",
        "pymongo.MongoClient",
        "tavily.TavilyClient",
    ]
    for target in targets:
        p = patch(target, return_value=MagicMock())
        p.start()
        _patches.append(p)


def pytest_unconfigure(config: object) -> None:  # noqa: ARG001
    for p in _patches:
        p.stop()
