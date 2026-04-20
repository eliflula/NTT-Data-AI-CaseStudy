"""Streamlit UI for the NTT DATA Sustainability RAG system."""
from __future__ import annotations

import logging
import time
import streamlit as st


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

st.set_page_config(
    page_title="NTT DATA Sustainability RAG",
    page_icon="🌱",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #f8faf8; }
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1.5px solid #c8e6c9;
        padding: 12px 16px;
        font-size: 15px;
    }
    .stButton > button {
        border-radius: 12px;
        background-color: #2e7d32;
        color: white;
        font-weight: 600;
        padding: 10px 32px;
        border: none;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #1b5e20; }
    .answer-box {
        background: white;
        border-left: 4px solid #2e7d32;
        border-radius: 8px;
        padding: 20px 24px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
        margin-top: 8px;
        color: #1a1a1a !important;
    }
    .answer-box p, .answer-box li, .answer-box span {
        color: #1a1a1a !important;
    }
    h1 { color: #1b5e20 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline() -> RAGGraph:
    return RAGGraph()


@st.cache_resource
def load_mongo() -> QueryLogger:
    return QueryLogger()


pipeline = load_pipeline()
mongo = load_mongo()

# ── Header ───
st.markdown("# 🌱 NTT DATA Sustainability RAG")
st.markdown("Ask questions about NTT DATA Sustainability Reports · **2020 · 2022 · 2023 · 2024 · 2025**")
st.divider()

# ── Input ───
question = st.text_input(
    label="Your question",
    placeholder="e.g. What are NTT DATA's carbon emission targets for 2030?",
    label_visibility="collapsed",
)

ask = st.button("Ask", disabled=not question)

# ── Query ───
if ask and question:
    with st.spinner("Searching through sustainability reports…"):
        t0 = time.time()
        answer, source_type, chunks, web_urls = pipeline.ask(question)
        latency = round(time.time() - t0, 3)

    if source_type == "web":
        log_sources = [
            type("S", (), {"source": w.get("title", ""), "score": 0.0, "url": w.get("url", "")})()
            for w in web_urls
        ]
    else:
        log_sources = [
            type("S", (), {
                "source": (point.payload or {}).get("source", ""),
                "score": point.score,
            })()
            for point in chunks
        ]

    mongo.log(
        question=question,
        answer=answer,
        sources=log_sources,
        source_type=source_type,
        latency=latency,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )

    st.divider()
    col_answer, col_sources = st.columns([3, 2])

    with col_answer:
        st.markdown("#### Answer")
        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True,
        )

    with col_sources:
        st.markdown("#### Sources")
        if source_type == "web":
            for i, w in enumerate(web_urls, 1):
                st.markdown(f"**{i}.** [{w.get('title', 'Web Result')}]({w.get('url', '')})")
        else:
            for i, point in enumerate(chunks, 1):
                p = point.payload or {}
                with st.expander(f"{i}. {p.get('year', '')} · Page {p.get('page', '')}  —  Score: {round(point.score, 3):.3f}"):
                    st.caption(p.get("source", ""))
                    content = p.get("content", "")
                    st.markdown(content[:800] + ("..." if len(content) > 800 else ""))
