from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    qdrant_url: str
    qdrant_api_key: str
    groq_api_key: str
    tavily_api_key: str = ""
    hf_token: str = ""

    # MongoDB — optional; logging is disabled when mongo_url is empty
    mongo_url: str = ""
    mongo_db: str = "ntt_rag"
    mongo_collection: str = "query_logs"

    collection: str = "ntt_data_sustainability"
    top_k: int = 5
    retrieval_top_k: int = 10
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "BAAI/bge-m3"
    vector_dim: int = 1024

    @property
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "data"


settings = Settings()
