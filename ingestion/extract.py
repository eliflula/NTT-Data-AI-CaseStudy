"""PDF extraction using Docling — full-document markdown with text cleaning."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)


@dataclass
class ExtractionMetrics:
    """Metrics collected during PDF extraction."""

    duration_seconds: float
    page_count: int
    non_empty_pages: int
    total_words: int
    text_count: int = 0
    picture_count: int = 0
    table_count: int = 0


@dataclass
class ExtractedDocument:
    """Full extracted content from a PDF file."""

    source: str           # filename without extension
    year: int | None
    file_path: Path
    full_markdown: str = ""
    markdown_path: Path | None = None
    metrics: ExtractionMetrics | None = None


# HTML / Docling artifact tags (<!-- image -->, <!-- figure -->, etc.)
# <<PAGE N>> markers are NOT HTML tags so they are safe from this regex
_RE_TAGS = re.compile(r"<!--.*?-->", re.DOTALL)

# Encoding artifact characters that appear as garbage in PDF extraction
_GARBAGE_CHARS = set("·­€ƒ†‡ˆ‰Š‹ŒŽ''""•–—˜™š›œžŸ¡¢£¤¥¦§¨©ª«¬\xad®¯°±²³´µ¶·¸¹º»¼½¾¿")

_RE_MULTI_SPACE = re.compile(r" {3,}")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
# Table separator rows: lines composed only of |, -, and spaces (no real content)
_RE_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s\-|]+\|\s*$")


def _is_garbage_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) < 3:
        return True
    # For table rows strip surrounding pipes before checking garbage ratio
    content = stripped.strip("|").strip() if stripped.startswith("|") else stripped
    if not content:
        return True
    garbage_count = sum(1 for c in content if c in _GARBAGE_CHARS)
    return garbage_count / len(content) > 0.4


def clean_text(text: str) -> str:
    """
    Remove extraction artifacts from Docling markdown output:
    - HTML comment tags (<!-- image -->, etc.)
    - Table separator rows (|---|) and table rows with garbage cell content
    - Lines with mostly garbage / encoding-artifact characters
    - Excessive whitespace and blank lines
    """
    # 1. Remove HTML/Docling comment tags
    text = _RE_TAGS.sub("", text)

    # 2. Remove table separators and garbage lines
    lines = text.splitlines()
    clean_lines = [
        line for line in lines
        if not _RE_TABLE_SEPARATOR.match(line) and not _is_garbage_line(line)
    ]
    text = "\n".join(clean_lines)

    # 3. Collapse excessive spaces and blank lines
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


def _extract_year(name: str) -> int | None:
    match = re.search(r"(20\d{2})", name)
    return int(match.group(1)) if match else None

class PDFExtractor:
    """
    Extracts a PDF into a single cleaned markdown string with embedded
    <<PAGE N>> markers so downstream chunking can recover page numbers.
    """

    def __init__(
        self,
        *,
        enable_ocr: bool = True,
        include_tables: bool = True,
        include_images: bool = False,
        images_scale: float = 1.0,
        output_dir: Path | None = None,
    ) -> None:
        options = PdfPipelineOptions()
        options.do_ocr = enable_ocr
        options.do_table_structure = include_tables
        options.do_picture_description = include_images
        options.images_scale = images_scale

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options)
            }
        )
        self._output_dir = output_dir

    def extract(self, pdf_path: Path) -> ExtractedDocument:
        """Extract PDF and return a single cleaned markdown with <<PAGE N>> markers."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

        source = pdf_path.stem
        year = _extract_year(source)
        logger.info("Extracting: %s (year=%s)", source, year)

        t0 = time.time()
        result = self._converter.convert(pdf_path)
        doc = result.document
        n_pages = len(doc.pages)

        # Build markdown page by page — one export call per page, markers protected
        parts: list[str] = []
        non_empty = 0
        for page_no in range(1, n_pages + 1):
            md = (doc.export_to_markdown(page_no=page_no) or "").strip()
            if not md or md == "<!-- empty page -->":
                continue
            cleaned = clean_text(md)
            if cleaned:
                parts.append(f"<<PAGE {page_no}>>\n{cleaned}")
                non_empty += 1

        full_markdown = "\n\n".join(parts)
        duration = round(time.time() - t0, 2)
        total_words = len(full_markdown.split())

        metrics = ExtractionMetrics(
            duration_seconds=duration,
            page_count=n_pages,
            non_empty_pages=non_empty,
            total_words=total_words,
            text_count=len(doc.texts),
            picture_count=len(doc.pictures),
            table_count=len(doc.tables),
        )

        logger.info(
            "Extracted '%s': %d/%d pages, %d words, %d tables, %d pictures in %.2fs",
            source, non_empty, n_pages, total_words,
            metrics.table_count, metrics.picture_count, duration,
        )

        md_path: Path | None = None
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            md_path = self._output_dir / f"{source}.md"
            md_path.write_text(full_markdown, encoding="utf-8")
            logger.info("Saved markdown: %s", md_path)

        return ExtractedDocument(
            source=source,
            year=year,
            file_path=pdf_path,
            full_markdown=full_markdown,
            markdown_path=md_path,
            metrics=metrics,
        )
