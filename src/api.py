from __future__ import annotations
import asyncio
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.mongo_logger import QueryLogger
from src.rag_graph import RAGGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.utils.import_utils").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NTT DATA Sustainability RAG API",
    description="RAG pipeline over NTT DATA sustainability reports (2020–2025)",
    version="1.0.0",
)

_pipeline = RAGGraph()
_mongo = QueryLogger()
_start_time = time.time()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str


class RAGSourceItem(BaseModel):
    source: str
    score: float
    year: str | int
    page: str | int
    chunk_id: str | None = None
    chunk_text: str | None = None


class WebSourceItem(BaseModel):
    source: str
    url: str


class AskResponse(BaseModel):
    answer: str
    source_type: str  # "rag" | "web"
    sources: list[RAGSourceItem] | list[WebSourceItem]


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    embedding_model: str
    llm_model: str
    collection: str
    vector_db: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    t0 = time.time()
    try:
        answer, source_type, chunks, web_urls = await asyncio.to_thread(_pipeline.ask, question)
    except Exception:
        logger.exception("Pipeline error for question: %s", question)
        raise HTTPException(status_code=500, detail="Internal pipeline error")
    latency = round(time.time() - t0, 3)

    if source_type == "web":
        sources = [
            WebSourceItem(
                source=w.get("title", "Web"),
                url=w.get("url", ""),
            )
            for w in web_urls
        ]
        top_score = 0.0
    else:
        sources = [
            RAGSourceItem(
                source=point.payload.get("source", ""),
                year=point.payload.get("year", ""),
                page=point.payload.get("page", ""),
                score=round(point.score, 4),
                chunk_id=point.payload.get("chunk_id"),
                chunk_text=point.payload.get("content"),
            )
            for point in chunks
        ]
        top_score = sources[0].score if sources else 0.0
    logger.info(
        'question="%s" source=%s chunks=%d top_score=%.4f latency=%.3fs',
        question,
        source_type,
        len(sources),
        top_score,
        latency,
    )

    _mongo.log(
        question=question,
        answer=answer,
        sources=sources,
        source_type=source_type,
        latency=latency,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )

    return AskResponse(answer=answer, sources=sources, source_type=source_type)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    qdrant_ok = await asyncio.to_thread(_pipeline.ping)
    overall = "ok" if qdrant_ok else "degraded"

    if not qdrant_ok:
        logger.warning("Health check: Qdrant unreachable or collection missing.")

    return HealthResponse(
        status=overall,
        uptime_seconds=round(time.time() - _start_time, 2),
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        collection=settings.collection,
        vector_db="ok" if qdrant_ok else "unreachable",
    )
