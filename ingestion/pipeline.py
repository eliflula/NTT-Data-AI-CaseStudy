"""
4-stage document ingestion pipeline.

  Stage 1 — Extract  : PDF → ExtractedDocument  (Docling)
  Stage 2 — Chunk    : ExtractedDocument → list[ChunkData]  (semantic / recursive)
  Stage 3 — Embed    : list[ChunkData] → list[PointStruct]  (BAAI/bge-m3)
  Stage 4 — Store    : list[PointStruct] → Qdrant

Usage (CLI):
    python -m ingestion.pipeline --dir doc/

Usage (Python):
    import asyncio
    from pathlib import Path
    from ingestion.pipeline import DocumentIngestionPipeline

    async def main():
        pipeline = DocumentIngestionPipeline()
        results = await pipeline.ingest_directory(Path("doc/"), clean=True)
        print(results)

    asyncio.run(main())
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Callable
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import re
from ingestion.chunker import ChunkData, ChunkingConfig, DocumentChunker, build_embeddings
from ingestion.extract import ExtractedDocument, PDFExtractor
from src.config import settings
from src.embedder import EmbeddingService

logger = logging.getLogger(__name__)

# Type alias for optional progress callbacks
ProgressCallback = Callable[[str, int, int], None]

_DEFAULT_BATCH_SIZE = 32


class DocumentIngestionPipeline:
    """
    Orchestrates the full Extract → Chunk → Embed → Store workflow.

    Parameters
    ----------
    chunk_size:       Max characters per chunk (default 1000).
    chunk_overlap:    Overlap between consecutive chunks (default 200).
    min_chunk_size:   Discard chunks shorter than this (default 100).
    batch_size:       Embedding / upsert batch size (default 32).
    enable_ocr:       Run OCR on scanned pages (default False).
    include_tables:   Extract table structure with TableFormer (default False).
    include_images:   Run vision model to describe figures (default False).
    markdown_dir:     If set, save extracted markdown files here (e.g. Path("data/markdown")).
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        enable_ocr: bool = False,
        include_tables: bool = False,
        include_images: bool = False,
        markdown_dir: Path | None = Path("doc/output-md"),
    ) -> None:
        self._embedder = EmbeddingService()
        self._extractor = PDFExtractor(
            enable_ocr=enable_ocr,
            include_tables=include_tables,
            include_images=include_images,
            output_dir=markdown_dir,
        )
        self._chunker = DocumentChunker(
            build_embeddings(),
            config=ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size,
            ),
        )
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_markdown(
        self,
        md_path: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> int:
        """Chunk → Embed → Store from an already-extracted .md file."""
        year_match = re.search(r"(20\d{2})", md_path.stem)
        doc = ExtractedDocument(
            source=md_path.stem,
            year=int(year_match.group(1)) if year_match else None,
            file_path=md_path,
            full_markdown=md_path.read_text(encoding="utf-8"),
        )

        _log_stage(1, "Chunk", doc.source)
        chunks = self._chunker.chunk_document(doc)
        _notify(progress, f"Created {len(chunks)} chunks", 1, 3)

        if not chunks:
            logger.warning("No chunks produced for %s — skipping.", md_path.name)
            return 0

        _log_stage(2, "Embed", f"{len(chunks)} chunks")
        points: list[PointStruct] = await asyncio.to_thread(self._embed_chunks, chunks)
        _notify(progress, f"Embedded {len(points)} vectors", 2, 3)

        _log_stage(3, "Store", settings.collection)
        await asyncio.to_thread(self._store_points, points)
        _notify(progress, f"Stored {len(points)} points in Qdrant", 3, 3)

        return len(points)

    async def ingest_markdown_directory(
        self,
        md_dir: Path,
        *,
        clean: bool = False,
        progress: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """Ingest all .md files in md_dir (skips files starting with 'chunks_')."""
        md_files = sorted(
            f for f in md_dir.glob("*.md")
            if not f.stem.startswith("chunks_")
        )
        if not md_files:
            logger.warning("No markdown files found in %s", md_dir)
            return {}

        if clean:
            await asyncio.to_thread(self._reset_collection)
        else:
            await asyncio.to_thread(self.ensure_collection)

        results: dict[str, int] = {}
        for md_path in md_files:
            try:
                count = await self.ingest_markdown(md_path, progress=progress)
                results[md_path.stem] = count
                logger.info("OK  %s — %d chunks stored", md_path.name, count)
            except Exception as exc:
                logger.error("FAIL %s — %s", md_path.name, exc, exc_info=True)
                results[md_path.stem] = -1

        total_ok = sum(v for v in results.values() if v >= 0)
        total_fail = sum(1 for v in results.values() if v < 0)
        logger.info(
            "Ingestion complete: %d files OK (%d chunks), %d failed.",
            len(results) - total_fail, total_ok, total_fail,
        )
        return results

    async def ingest_file(
        self,
        pdf_path: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> int:
        """
        Ingest a single PDF file end-to-end.

        Returns the number of chunks stored in Qdrant.
        """
        # Stage 1 — Extract
        _log_stage(1, "Extract", pdf_path.name)
        extracted: ExtractedDocument = await asyncio.to_thread(
            self._extractor.extract, pdf_path
        )
        pages = extracted.metrics.non_empty_pages if extracted.metrics else "?"
        md_info = f" → {extracted.markdown_path.name}" if extracted.markdown_path else ""
        _notify(progress, f"Extracted {pages} pages{md_info}", 1, 4)

        # Stage 2 — Chunk
        _log_stage(2, "Chunk", extracted.source)
        chunks = self._chunker.chunk_document(extracted)
        _notify(progress, f"Created {len(chunks)} chunks", 2, 4)

        if not chunks:
            logger.warning("No chunks produced for %s — skipping.", pdf_path.name)
            return 0

        # Annotate every chunk with document-level metadata from extraction
        if extracted.metrics:
            m = extracted.metrics
            doc_meta = {
                "processing_time": m.duration_seconds,
                "total_pages": m.page_count,
                "texts": m.text_count,
                "pictures": m.picture_count,
                "tables": m.table_count,
                "extraction_method": "docling",
                "content_type": "pdf",
            }
            for chunk in chunks:
                chunk.metadata.update(doc_meta)

        # Stage 3 — Embed
        _log_stage(3, "Embed", f"{len(chunks)} chunks")
        points: list[PointStruct] = await asyncio.to_thread(
            self._embed_chunks, chunks
        )
        _notify(progress, f"Embedded {len(points)} vectors", 3, 4)

        # Stage 4 — Store
        _log_stage(4, "Store", settings.collection)
        await asyncio.to_thread(self._store_points, points)
        _notify(progress, f"Stored {len(points)} points in Qdrant", 4, 4)

        return len(points)

    async def ingest_directory(
        self,
        pdf_dir: Path,
        *,
        clean: bool = False,
        progress: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """
        Ingest all PDF files found in *pdf_dir*.

        Parameters
        ----------
        pdf_dir: Directory that contains the PDF files.
        clean:   If True, drop and recreate the Qdrant collection first.

        Returns a mapping of {filename: chunk_count}.
        A chunk_count of -1 means the file failed.
        """
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in %s", pdf_dir)
            return {}

        if clean:
            await asyncio.to_thread(self._reset_collection)
        else:
            await asyncio.to_thread(self.ensure_collection)

        results: dict[str, int] = {}
        for pdf_path in pdf_files:
            try:
                count = await self.ingest_file(pdf_path, progress=progress)
                results[pdf_path.name] = count
                logger.info("OK  %s — %d chunks stored", pdf_path.name, count)
            except Exception as exc:
                logger.error(
                    "FAIL %s — %s", pdf_path.name, exc, exc_info=True
                )
                results[pdf_path.name] = -1

        total_ok = sum(v for v in results.values() if v >= 0)
        total_fail = sum(1 for v in results.values() if v < 0)
        logger.info(
            "Ingestion complete: %d files OK (%d chunks), %d failed.",
            len(results) - total_fail,
            total_ok,
            total_fail,
        )
        return results

    def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        if not self._client.collection_exists(settings.collection):
            self._client.create_collection(
                collection_name=settings.collection,
                vectors_config=VectorParams(
                    size=settings.vector_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'", settings.collection)
        else:
            logger.info("Collection '%s' already exists", settings.collection)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _embed_chunks(self, chunks: list[ChunkData]) -> list[PointStruct]:
        """Embed chunks in batches and return Qdrant PointStructs."""
        points: list[PointStruct] = []
        total = len(chunks)

        for i in range(0, total, self._batch_size):
            batch = chunks[i : i + self._batch_size]
            embeddings = self._embedder.embed_batch([c.text for c in batch])

            for chunk, vector in zip(batch, embeddings):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "content": chunk.text,
                            "source": chunk.source,
                            "year": chunk.year,
                            "page": chunk.page,
                            "chunk_index": chunk.chunk_index,
                            "chunk_id": chunk.chunk_id,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "token_count": chunk.token_count,
                            "chunking_method": chunk.chunking_method,
                            **chunk.metadata,
                        },
                    )
                )

            done = min(i + self._batch_size, total)
            logger.debug("Embedded %d/%d chunks", done, total)

        return points

    def _store_points(self, points: list[PointStruct]) -> None:
        """Upsert points into Qdrant in batches."""
        self.ensure_collection()
        total = len(points)
        for i in range(0, total, self._batch_size):
            batch = points[i : i + self._batch_size]
            self._client.upsert(
                collection_name=settings.collection,
                points=batch,
            )
            done = min(i + self._batch_size, total)
            logger.debug("Upserted %d/%d points", done, total)

    def _reset_collection(self) -> None:
        """Drop (if exists) and recreate the Qdrant collection."""
        if self._client.collection_exists(settings.collection):
            self._client.delete_collection(settings.collection)
            logger.info("Deleted collection '%s'", settings.collection)
        self._client.create_collection(
            collection_name=settings.collection,
            vectors_config=VectorParams(
                size=settings.vector_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Re-created collection '%s'", settings.collection)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _log_stage(step: int, name: str, detail: str) -> None:
    logger.info("=== Stage %d/%d — %s  [%s]", step, 4, name, detail)


def _notify(
    callback: ProgressCallback | None, message: str, step: int, total: int
) -> None:
    logger.info("[%d/%d] %s", step, total, message)
    if callback:
        callback(message, step, total)


# ------------------------------------------------------------------
# CLI entry-point:  python -m ingestion.pipeline --dir doc/
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    parser.add_argument("--dir", type=Path, default=None,
                        help="Directory containing PDF files.")
    parser.add_argument("--from-markdown", type=Path, default=None,
                        help="Directory containing pre-extracted .md files.")
    parser.add_argument("--clean", action="store_true",
                        help="Drop and recreate the Qdrant collection before ingesting.")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=400)
    return parser.parse_args()


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()

    pipeline = DocumentIngestionPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.from_markdown:
        results = await pipeline.ingest_markdown_directory(
            args.from_markdown, clean=args.clean
        )
    else:
        pdf_dir = args.dir or Path("doc")
        results = await pipeline.ingest_directory(pdf_dir, clean=args.clean)

    print("\n--- Ingestion Summary ---")
    for name, count in results.items():
        status = f"{count} chunks" if count >= 0 else "FAILED"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    asyncio.run(_main())
