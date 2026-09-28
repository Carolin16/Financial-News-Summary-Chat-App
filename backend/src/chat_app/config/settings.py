"""Application settings, loaded from environment variables (and an optional `.env`).

Every tunable value lives here so no module hard-codes paths, model names, or thresholds.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_CONFIG_DIR = Path(__file__).parent


class Settings(BaseSettings):
    """Typed configuration shared by the offline indexer and the API."""

    # Root .env first, then a local one; running from backend/ still picks up the shared file.
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # --- data ---
    data_path: Path = Field(default=Path("../data/stock_news.json"))
    enrichment_cache_path: Path = Field(default=Path("../data/enrichment_cache.json"))
    tickers_path: Path = Field(default=_PACKAGE_CONFIG_DIR / "tickers.json")

    # --- LLM / embeddings ---
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    openai_base_url: str | None = None
    llm_model: str = "gpt-5.6-terra"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3
    embedding_batch_size: int = 64
    enrichment_max_chars: int = 12000
    enrichment_fallback_min_mentions: int = 3
    enrichment_workers: int = 8

    # --- vector store ---
    qdrant_url: str = "http://localhost:6333"
    # When set, Qdrant runs embedded in-process at this path (no server; single process only).
    qdrant_local_path: Path | None = None
    qdrant_collection: str = "stock_news"
    sparse_model: str = "Qdrant/bm25"

    # --- ingestion ---
    stub_min_words: int = 80
    near_duplicate_threshold: float = 0.6
    title_duplicate_threshold: float = 0.8
    shingle_size: int = 5
    chunk_max_tokens: int = 500
    chunk_overlap_sentences: int = 1
    tokenizer_encoding: str = "cl100k_base"

    # --- retrieval ---
    retrieval_top_k: int = 8
    retrieval_prefetch_k: int = 30
    full_coverage_min_articles: int = 5

    # --- API ---
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    max_query_chars: int = 500
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached so env is read once)."""
    return Settings()
