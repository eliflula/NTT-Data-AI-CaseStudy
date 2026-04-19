# NTT DATA Sustainability RAG

An Agentic RAG system that answers questions over NTT DATA's publicly available sustainability reports (2020–2025). A LangGraph ReAct agent decides at runtime whether to search the vector database or fall back to web search — with LLM-based relevance grading, query rewriting, and reranking at each step.

## 🗺️ Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Features

- 🧩 **Agentic RAG**: LangGraph ReAct agent decides which tool to call and how many times
- ✏️ **Query rewriting**: converts shorthand or non-English questions into precise English search queries
- 🎯 **LLM grading + reranking**: Groq 70b scores retrieved chunks for relevance before generation
- 🌍 **Web fallback**: automatically searches Tavily when documents don't contain the answer
- 📑 **Advanced PDF processing**: semantic chunking with Docling extraction
- 💡 **Multi-document reasoning**: compares data across report years
- ⚡ **Async API**: non-blocking FastAPI endpoints with `asyncio.to_thread`
- 🔎 **Observability**: structured logging and MongoDB query logs
- 🐳 **Containerized**: single `docker-compose up` starts both API and Streamlit UI

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Preprocessing                            │
│      Documents → Docling Parsing → Structured Markdown          │
│                      → Semantic Chunking                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌──────────┐   ┌────────────────▼──────────┐
│   User   │   │     Vector Embeddings     │
│ Question │──►│       (BAAI/bge-m3)       │
└──────────┘   └────────────────┬──────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    Search Document    │
                    │       (Qdrant)        │
                    └───────────┬───────────┘
                                │
                                ▼
         ┌──────────────────────────────────┐   ┌─────────────────────┐
         │          GRADER AGENT            │   │    AGENT TOOLS      │
         │  Think → Act → Observe → Think   │◄─►│  ┌───────────────┐  │
         │                                  │   │  │  Search Doc   │  │
         │   ┌──────────┐  ┌────────────┐   │   │  │   (Qdrant)    │  │
         │   │ RELEVANT │  │ IRRELEVANT │   │   │  └───────────────┘  │
         │   └─────┬────┘  └──────┬─────┘   │   │  ┌───────────────┐  │
         │                        │         │  │   │  │   Web Search  │  
         └─────────┼───────────────┼────────┘   │  │ (DuckDuckGo)  │  │
                   │               │            │  └───────────────┘  │
                   │               │            └─────────────────────┘
        ┌──────────▼──────┐  ┌─────▼──────────┐
        │  Reranker Agent │  │   Web Search   │
        └────────┬────────┘  └───────┬────────┘
                 │                   │
                 └─────────┬─────────┘
                           │
                           ▼
              ┌────────────────────────┐   ┌──────────────────────┐
              │       GENERATOR        │──►│ Observability/Logging│
              │     Final Response     │   │      (MongoDB)       │
              └────────────────────────┘   └──────────────────────┘
```

### LLM & Embedding Models

| Model | Role | Purpose |
|---|---|---|
| `llama-3.1-8b-instant` | ReAct Agent, Query Rewrite | Tool selection, fast routing |
| `llama-3.3-70b-versatile` | Grader, Reranker, Generator | High-quality reasoning and accuracy |
| `BAAI/bge-m3` | Vector Embedding | Multi-functionality, multilingual, multi-granularity |

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph|
| LLM inference | Groq API  |
| Embedding | `BAAI/bge-m3` |
| Vector DB | Qdrant Cloud |
| PDF extraction | Docling |
| Chunking | LangChain |
| Web search | DuckDuckGo |
| API | FastAPI + `asyncio.to_thread` |
| UI | Streamlit |
| Logging | MongoDB Atlas |
| CI/CD | GitHub Actions |
| Containers | Docker Compose |

---

## 📦 Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [Qdrant Cloud](https://cloud.qdrant.io/) account (free tier works)
- [Groq API key](https://console.groq.com/) (free tier works)
- MongoDB Atlas (optional — for query logging)

### 1. Clone the Repository

```bash
git clone https://github.com/eliflula/NTT-DATA-case-study.git
cd NTT-DATA-case-study
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

```env
QDRANT_URL=https://<your-cluster>.qdrant.io:6333
QDRANT_API_KEY=<your-qdrant-api-key>
GROQ_API_KEY=<your-groq-api-key>

# Optional — logging disabled when empty
MONGO_URL=mongodb+srv://<user>:<pass>@cluster.mongodb.net/
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Ingest Documents (first time only)

```bash
python -m ingestion.pipeline --dir doc/ --clean
```

---

## 💻 Usage

### Web Interface

```bash
streamlit run app.py
```

Access at: <http://localhost:8501>

### API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: <http://localhost:8000/docs>

### Docker Compose (API + UI)

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI | <http://localhost:8000> |
| Streamlit UI | <http://localhost:8501> |

---

## 🔌 API Reference

### `POST /ask`

**Request:**
```json
{ "question": "What were NTT DATA's carbon emissions in 2024?" }
```

**Response (RAG):**
```json
{
  "answer": "NTT DATA reduced carbon emissions by...",
  "source_type": "rag",
  "sources": [
    {
      "source": "NTT-DATA_Sustainability-Report_2024_CaseBook",
      "year": 2024,
      "page": 12,
      "score": 0.8712,
      "chunk_id": "chunk_0042",
      "chunk_text": "..."
    }
  ]
}
```

**Response (web fallback):**
```json
{
  "answer": "...",
  "source_type": "web",
  "sources": [
    { "source": "Example Article", "url": "https://example.com" }
  ]
}
```

### `GET /health`

```json
{
  "status": "ok",
  "uptime_seconds": 142.5,
  "embedding_model": "BAAI/bge-m3",
  "llm_model": "llama-3.3-70b-versatile",
  "collection": "ntt_data_sustainability",
  "vector_db": "ok"
}
```

`status` is `"degraded"` when Qdrant is unreachable.

---

## 📁 Project Structure

```
NTT-DATA-case-study/
├── src/
│   ├── rag_graph.py      # RAGGraph — LangGraph agent, tools, grader, reranker
│   ├── api.py            # FastAPI endpoints
│   ├── embedder.py       # EmbeddingService (BAAI/bge-m3)
│   ├── retriever.py      # RetrieverService (Qdrant)
│   ├── generator.py      # GeneratorService (Groq LLM) + system prompt
│   ├── mongo_logger.py   # QueryLogger + SessionStore (MongoDB)
│   └── config.py         # pydantic-settings configuration
├── ingestion/
│   ├── pipeline.py       # DocumentIngestionPipeline (PDF → Qdrant)
│   ├── extract.py        # PDFExtractor (Docling)
│   └── chunker.py        # DocumentChunker (semantic + recursive)
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_embedder.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   └── test_chunker.py
├── doc/                  # Sustainability report PDFs (2020–2025)
├── app.py                # Streamlit demo UI
├── Dockerfile
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .github/workflows/ci.yml
```

---

## ✅ Testing

```bash
pytest tests/ -v
```

| File | Coverage |
|---|---|
| `test_api.py` | FastAPI endpoints, RAG/web responses, validation |
| `test_embedder.py` | Embedding output shape, normalization |
| `test_retriever.py` | Qdrant calls, ping, top-k handling |
| `test_generator.py` | Context building, LLM call structure |
| `test_chunker.py` | ChunkingConfig validation, ChunkData fields |

All 25 tests run without network access or model downloads.

---

## 🔧 Configuration

| Variable | Description | Default |
|---|---|---|
| `QDRANT_URL` | Qdrant cluster URL | required |
| `QDRANT_API_KEY` | Qdrant API key | required |
| `GROQ_API_KEY` | Groq API key | required |
| `MONGO_URL` | MongoDB connection string | `""` (disabled) |
| `MONGO_DB` | MongoDB database name | `ntt_rag` |
| `COLLECTION` | Qdrant collection name | `ntt_data_sustainability` |
| `LLM_MODEL` | Groq model for grading/reranking/generation | `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | Sentence-transformer model | `BAAI/bge-m3` |

---

## 🩺 Troubleshooting

**Qdrant connection error**
```bash
curl -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/collections"
```

**Empty answers / "not available"**
```bash
# Rebuild the collection
python -m ingestion.pipeline --dir doc/ --clean
```

**Groq rate limit**

The agent makes up to 4 Groq calls per query (rewrite, retrieve, grade, rerank, generate). Use `llama-3.1-8b-instant` for all steps if rate limits are an issue by setting `LLM_MODEL=llama-3.1-8b-instant`.
