from __future__ import annotations
import json
import logging
from groq import Groq
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from qdrant_client.models import ScoredPoint
from src.config import settings
from src.embedder import EmbeddingService
from src.generator import GeneratorService
from src.retriever import RetrieverService
from src.web_searcher import WebSearchService

logger = logging.getLogger(__name__)

_GRADER_PROMPT = """You are a relevance grader. Given a question and retrieved document chunks,
decide if the chunks contain information useful to answer the question.

Respond with ONLY one word: "relevant" or "irrelevant".

Question: {question}
Retrieved context: {context}

Are these chunks relevant to the question?"""

_RERANK_PROMPT = """You are a document relevance scorer.
Given a question and document chunks, score each chunk from 1 to 10
based on how useful it is to answer the question.
1 = completely irrelevant, 10 = directly answers the question.

Return ONLY a valid JSON array, no explanation:
[{{"id": 1, "score": 8}}, {{"id": 2, "score": 3}}, ...]

Question: {question}

Chunks:
{chunks}"""

_RERANK_MIN_SCORE = 2
_RERANK_TOP_K = 5

_REWRITE_PROMPT = """Convert the following question into a precise search query.
The input may be in any language or shorthand.
Return ONLY the query, nothing else.

Question: {question}"""

_AGENT_SYSTEM_PROMPT = """You are an AI research assistant for NTT DATA sustainability topics.

Use search_documents first. If it returns "IRRELEVANT", then use search_web.
Base your answer solely on tool results — never use prior knowledge.
If no tool returns useful information, say: "The information is not available in the provided documents."

Answer rules: be clear and professional, preserve exact numbers, no source citations, no hallucination."""


class RAGGraph:
    """
    Agentic RAG — LLM decides at runtime which tool to call and how many times.

    START → agent (ReAct loop) → END
              ├─ tool: search_documents  (Qdrant + LLM rerank)
              └─ tool: search_web        (Tavily)
    """

    def __init__(self) -> None:
        embedder = EmbeddingService()
        self._retriever = RetrieverService(embedder)
        self._generator = GeneratorService()
        self._web_searcher = WebSearchService()
        self._groq = Groq(api_key=settings.groq_api_key)

        # Per-request state — reset in ask()
        self._last_chunks: list[ScoredPoint] = []
        self._last_web_urls: list[dict] = []
        self._last_source_type: str = "rag"
        self._last_context: str = ""

        self._graph = self._build_agent()

    def ping(self) -> bool:
        return self._retriever.ping()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rewrite_query(self, question: str) -> str:
        response = self._groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": _REWRITE_PROMPT.format(question=question)}],
            temperature=0,
            max_tokens=60,
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info('[AGENT] rewrite: "%s" → "%s"', question, rewritten)
        return rewritten

    def _rerank(self, question: str, chunks: list[ScoredPoint]) -> list[ScoredPoint]:
        """Score chunks with Groq, filter low-scorers, return top-k."""
        formatted = "\n\n".join(
            f"[{i + 1}] {(chunks[i].payload or {}).get('content', '')[:400]}"
            for i in range(len(chunks))
        )
        response = self._groq.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": _RERANK_PROMPT.format(
                question=question, chunks=formatted
            )}],
            temperature=0,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()

        try:
            scores = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[AGENT] rerank: JSON parse failed, keeping all chunks")
            scores = [{"id": i + 1, "score": 5} for i in range(len(chunks))]

        filtered = [s for s in scores if s.get("score", 0) >= _RERANK_MIN_SCORE]
        ranked = sorted(filtered, key=lambda x: x["score"], reverse=True)

        if not ranked:
            logger.warning("[AGENT] rerank: all below min_score=%d, falling back to top-3", _RERANK_MIN_SCORE)
            ranked = sorted(scores, key=lambda x: x.get("score", 0), reverse=True)[:3]

        top_ids = [r["id"] - 1 for r in ranked[:_RERANK_TOP_K]]
        reranked = [chunks[i] for i in top_ids if i < len(chunks)]

        logger.info(
            "[AGENT] rerank: %d → %d chunks (min_score=%d, top_k=%d)",
            len(chunks), len(reranked), _RERANK_MIN_SCORE, _RERANK_TOP_K,
        )
        return reranked

    # ------------------------------------------------------------------
    # Agent builder
    # ------------------------------------------------------------------

    def _build_agent(self):
        # llama-3.1-8b-instant: Groq'ta tool calling için güvenilir
        # Grader/reranker zaten settings.llm_model (70b) kullanıyor
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=settings.groq_api_key,
            temperature=0.2,
            model_kwargs={"parallel_tool_calls": False},
        )

        @tool
        def search_documents(query: str) -> str:
            """Search NTT DATA sustainability PDF documents stored in Qdrant vector database.
            Returns document context if relevant, or 'IRRELEVANT' if documents do not contain useful information."""
            logger.info('[AGENT] tool=search_documents query="%s"', query)
            rewritten = self._rewrite_query(query)
            results = self._retriever.retrieve(rewritten, top_k=settings.retrieval_top_k)

            if not results:
                logger.info("[AGENT] search_documents: no chunks found → IRRELEVANT")
                return "IRRELEVANT"

            # Grade: check if retrieved chunks are useful for the question
            sample = self._generator.build_context(results[:3])
            grade_prompt = _GRADER_PROMPT.format(question=query, context=sample[:2000])
            grade_resp = self._groq.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": grade_prompt}],
                temperature=0,
                max_tokens=5,
            )
            verdict = grade_resp.choices[0].message.content.strip().lower()
            relevant = verdict == "relevant"
            logger.info("[AGENT] search_documents: grade=%s", verdict)

            if not relevant:
                logger.info("[AGENT] search_documents: irrelevant → IRRELEVANT")
                return "IRRELEVANT"

            # Rerank
            reranked = self._rerank(query, results)
            logger.info("[AGENT] search_documents: %d → %d chunks after rerank", len(results), len(reranked))

            if not reranked:
                logger.info("[AGENT] search_documents: rerank empty → IRRELEVANT")
                return "IRRELEVANT"

            self._last_chunks = reranked
            self._last_source_type = "rag"
            self._last_context = self._generator.build_context(reranked)
            return self._last_context

        @tool
        def search_web(query: str) -> str:
            """Search the internet using Tavily for up-to-date information."""
            logger.info('[AGENT] tool=search_web query="%s"', query)

            if self._last_chunks:
                logger.info("[AGENT] search_web: RAG context already found, ignoring web results")
                return "Documents already provided sufficient context."

            results = self._web_searcher.search(query)
            self._last_web_urls = [{"title": r.title, "url": r.url} for r in results]
            self._last_source_type = "web"
            self._last_chunks = []
            self._last_context = self._web_searcher.build_context(results)
            return self._last_context

             #create_react_agent = hidden graph (nodes + edges + loop)
        return create_react_agent(
            model=llm,
            tools=[search_documents, search_web],
            prompt=_AGENT_SYSTEM_PROMPT,
        )


    def ask(self, question: str) -> tuple[str, str, list[ScoredPoint], list[dict]]:
        """Returns (answer, source_type, rag_chunks, web_urls)."""
        self._last_chunks = []
        self._last_web_urls = []
        self._last_source_type = "rag"
        self._last_context = ""

        self._graph.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": 6},
        )

        # GeneratorService: _SYSTEM_PROMPT + settings.llm_model (70b) ile cevap üret
        if self._last_context:
            gen = self._generator.generate(question, self._last_context)
            answer = gen.answer
        else:
            answer = "The information is not available in the provided documents."

        logger.info(
            "[AGENT] done | source=%s chunks=%d web_urls=%d answer_len=%d",
            self._last_source_type,
            len(self._last_chunks),
            len(self._last_web_urls),
            len(answer),
        )
        return answer, self._last_source_type, self._last_chunks, self._last_web_urls
