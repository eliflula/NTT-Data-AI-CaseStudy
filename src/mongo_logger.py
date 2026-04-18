from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Protocol

from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError

from src.config import settings


class SourceLike(Protocol):
    """Structural type for any object with source/score fields."""

    source: str
    score: float


logger = logging.getLogger(__name__)


class QueryLogger:
    """Persists RAG query logs to MongoDB. Failures are logged but never raised."""

    def __init__(self) -> None:
        self._client: MongoClient | None = None
        self._collection = None

        if not settings.mongo_url:
            logger.warning("MONGO_URL not set — MongoDB logging disabled.")
            return

        try:
            self._client = MongoClient(settings.mongo_url, serverSelectionTimeoutMS=3000)
            db = self._client[settings.mongo_db]
            self._collection = db[settings.mongo_collection]
            logger.info(
                "MongoDB logger connected: db=%s collection=%s",
                settings.mongo_db,
                settings.mongo_collection,
            )
        except PyMongoError as exc:
            logger.error("MongoDB logger init failed: %s", exc)
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._collection is not None

    def log(
        self,
        *,
        question: str,
        answer: str,
        sources: list[SourceLike],
        source_type: str,
        latency: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        """Insert one query log document. Silently skips if MongoDB is unavailable."""
        if not self.enabled:
            return

        if source_type == "web":
            doc = {
                "question": question,
                "answer": answer,
                "source_type": "web",
                "web_sources": [
                    {"title": s.source, "url": getattr(s, "url", "")}
                    for s in sources
                ],
                "llm_model": settings.llm_model,
                "embedding_model": settings.embedding_model,
                "latency_seconds": latency,
                "timestamp": datetime.now(timezone.utc),
            }
        else:
            top_score = getattr(sources[0], "score", 0.0) if sources else 0.0
            top_source = sources[0].source if sources else ""
            doc = {
                "question": question,
                "answer": answer,
                "source_type": "rag",
                "source_doc": top_source,
                "top_score": top_score,
                "chunk_count": len(sources),
                "chunks": [
                    {
                        "chunk_id": getattr(s, "chunk_id", None),
                        "content": getattr(s, "chunk_text", None),
                        "score": getattr(s, "score", 0.0),
                    }
                    for s in sources
                ],
                "llm_model": settings.llm_model,
                "embedding_model": settings.embedding_model,
                "latency_seconds": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "timestamp": datetime.now(timezone.utc),
            }

        try:
            self._collection.insert_one(doc)
        except PyMongoError as exc:
            logger.error("MongoDB insert failed: %s", exc)


class SessionStore:
    """
    Stores conversation sessions and per-turn messages in MongoDB.

    Collections
    -----------
    sessions  : one document per conversation session
    messages  : one document per message (user or assistant)
    """

    def __init__(self) -> None:
        self._sessions = None
        self._messages = None

        if not settings.mongo_url:
            logger.warning("MONGO_URL not set — SessionStore disabled.")
            return

        try:
            client = MongoClient(settings.mongo_url, serverSelectionTimeoutMS=3000)
            db = client[settings.mongo_db]
            self._sessions = db["sessions"]
            self._messages = db["messages"]
            self._messages.create_index(
                [("session_id", ASCENDING), ("timestamp", ASCENDING)]
            )
            logger.info("SessionStore connected: db=%s", settings.mongo_db)
        except PyMongoError as exc:
            logger.error("SessionStore init failed: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._sessions is not None

    def get_or_create(self, session_id: str | None) -> str:
        """Return an existing session_id or create a new one."""
        if not self.enabled:
            return session_id or str(uuid.uuid4())

        if session_id:
            if self._sessions.find_one({"_id": session_id}):
                return session_id

        new_id = str(uuid.uuid4())
        try:
            self._sessions.insert_one(
                {"_id": new_id, "created_at": datetime.now(timezone.utc)}
            )
        except PyMongoError as exc:
            logger.error("SessionStore create failed: %s", exc)
        return new_id

    def get_history(self, session_id: str, limit: int = 6) -> list[dict]:
        """
        Return the last *limit* messages for a session as
        [{"role": "user"|"assistant", "content": "..."}].
        """
        if not self.enabled:
            return []
        try:
            docs = list(
                self._messages.find(
                    {"session_id": session_id},
                    {"_id": 0, "role": 1, "content": 1},
                )
                .sort("timestamp", ASCENDING)
                .skip(
                    max(
                        0,
                        self._messages.count_documents({"session_id": session_id})
                        - limit,
                    )
                )
            )
            return docs
        except PyMongoError as exc:
            logger.error("SessionStore get_history failed: %s", exc)
            return []

    def save_turn(self, session_id: str, question: str, answer: str) -> None:
        """Persist one user+assistant exchange."""
        if not self.enabled:
            return
        now = datetime.now(timezone.utc)
        try:
            self._messages.insert_many([
                {"session_id": session_id, "role": "user",      "content": question, "timestamp": now},
                {"session_id": session_id, "role": "assistant", "content": answer,   "timestamp": now},
            ])
        except PyMongoError as exc:
            logger.error("SessionStore save_turn failed: %s", exc)
