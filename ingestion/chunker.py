"""
Hybrid document chunker — works on full document text (not page-by-page).

Strategy:
  1. Semantic chunking  — LangChain SemanticChunker with HuggingFace embeddings.
  2. Large chunk control — RecursiveCharacterTextSplitter breaks oversized semantic chunks.
  3. Recursive fallback — RecursiveCharacterTextSplitter when semantic chunking fails.

Page numbers are recovered from <<PAGE N>> markers embedded by extract.py.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from ingestion.extract import ExtractedDocument
from src.config import settings

logger = logging.getLogger(__name__)

_RE_PAGE_MARKER = re.compile(r"<<PAGE\s+(\d+)>>")

@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 2000
    chunk_overlap: int = 400
    min_chunk_size: int = 100
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: float = 95.0
    use_semantic_splitting: bool = True

    def __post_init__(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")


@dataclass
class ChunkData:
    """A single text chunk ready for embedding and storage."""

    chunk_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    source: str
    year: int | None
    page: int | None
    token_count: int
    chunking_method: str
    metadata: dict = field(default_factory=dict)


def build_embeddings(device: str = "cpu") -> HuggingFaceEmbeddings:
    """Create HuggingFaceEmbeddings for BGE-M3."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


class DocumentChunker:
    """
    Splits an ExtractedDocument's full markdown into overlapping chunks.

    Works on the complete document text so topic continuity is preserved
    across page boundaries (e.g. a topic spanning pages 3-5 stays together).
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()

        self._semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=self.config.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.config.breakpoint_threshold_amount,
            min_chunk_size=self.config.min_chunk_size,
        )
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)

    def chunk_document(self, doc: ExtractedDocument) -> list[ChunkData]:
        text = doc.full_markdown.strip()
        if not text:
            logger.warning("'%s' has empty full_markdown — nothing to chunk.", doc.source)
            return []

        page_map = _build_page_map(text)

        clean = _RE_PAGE_MARKER.sub("", text).strip()
        clean = re.sub(r"\n{3,}", "\n\n", clean)

        # Small content → skip semantic embedding, go straight to recursive
        use_semantic = (
            self.config.use_semantic_splitting
            and len(clean) > self.config.chunk_size
        )

        if use_semantic:
            try:
                semantic_chunks = self._semantic_splitter.split_text(clean)
                if not semantic_chunks:
                    raise ValueError("empty result from semantic chunker")
                raw_chunks = []
                for chunk in semantic_chunks:
                    if len(chunk) > self.config.chunk_size:
                        raw_chunks.extend(self._recursive_splitter.split_text(chunk))
                    else:
                        raw_chunks.append(chunk)
                method = "semantic"
            except Exception as exc:
                logger.debug("Semantic chunking failed (%s) — using recursive fallback", exc)
                raw_chunks = self._recursive_splitter.split_text(clean)
                method = "recursive"
        else:
            raw_chunks = self._recursive_splitter.split_text(clean)
            method = "recursive"

        total = len(raw_chunks)
        chunks: list[ChunkData] = []
        offset = 0
        for i, raw in enumerate(raw_chunks):
            raw = raw.strip()
            if len(raw) < self.config.min_chunk_size:
                continue

            pos = clean.find(raw, offset)
            if pos == -1:
                pos = offset
            end = pos + len(raw)
            offset = end

            chunks.append(ChunkData(
                chunk_id=f"{doc.source}_chunk{i:04d}",
                text=raw,
                chunk_index=i,
                start_char=pos,
                end_char=end,
                source=doc.source,
                year=doc.year,
                page=_page_at(pos, page_map),
                token_count=len(self._tokenizer.encode(raw)),
                chunking_method=method,
                metadata={
                    "total_chunks": total,
                    "chunk_size": len(raw),
                },
            ))

        logger.info("Chunked '%s': %d chunks (%s)", doc.source, len(chunks), method)
        return chunks


def _build_page_map(text: str) -> list[tuple[int, int]]:
    return [(m.start(), int(m.group(1))) for m in _RE_PAGE_MARKER.finditer(text)]


def _page_at(offset: int, page_map: list[tuple[int, int]]) -> int | None:
    page = None
    for marker_offset, page_no in page_map:
        if marker_offset <= offset:
            page = page_no
        else:
            break
    return page
