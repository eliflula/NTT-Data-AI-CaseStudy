from __future__ import annotations
import pytest
from ingestion.chunker import ChunkingConfig, ChunkData


def test_chunking_config_defaults() -> None:
    cfg = ChunkingConfig()
    assert cfg.chunk_size == 2000
    assert cfg.chunk_overlap == 400
    assert cfg.min_chunk_size == 100
    assert cfg.use_semantic_splitting is True


def test_chunking_config_custom_values() -> None:
    cfg = ChunkingConfig(chunk_size=1000, chunk_overlap=100, min_chunk_size=50)
    assert cfg.chunk_size == 1000
    assert cfg.chunk_overlap == 100
    assert cfg.min_chunk_size == 50


def test_chunking_config_rejects_overlap_gte_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        ChunkingConfig(chunk_size=500, chunk_overlap=500)


def test_chunking_config_rejects_zero_min_chunk_size() -> None:
    with pytest.raises(ValueError, match="min_chunk_size must be positive"):
        ChunkingConfig(min_chunk_size=0)


def test_chunk_data_fields() -> None:
    chunk = ChunkData(
        chunk_id="abc-123",
        text="NTT DATA sustainability report content.",
        source="NTT-DATA_Sustainability-Report_2024_CaseBook",
        year=2024,
        page=5,
        chunk_index=0,
        start_char=0,
        end_char=38,
        token_count=7,
        chunking_method="semantic",
    )
    assert chunk.chunk_id == "abc-123"
    assert chunk.year == 2024
    assert chunk.page == 5
    assert chunk.chunking_method == "semantic"


def test_chunk_data_metadata_defaults_to_empty_dict() -> None:
    chunk = ChunkData(
        chunk_id="x",
        text="text",
        source="src",
        year=2023,
        page=1,
        chunk_index=0,
        start_char=0,
        end_char=4,
        token_count=1,
        chunking_method="recursive",
    )
    assert chunk.metadata == {}
