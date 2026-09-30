"""All app settings in one place: file paths, model names, limits and thresholds.

Each value has a default here and can be overridden by an environment variable or `.env`.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_CONFIG_DIR = Path(__file__).parent


class Settings(BaseSettings):
    """Settings used by both the indexing script and the API."""

    # Reads the project-root .env, then a local one, so running from backend/ still works.
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # --- data ---
    data_path: Path = Field(default=Path("../data/stock_news.json"))
    tickers_path: Path = Field(default=_PACKAGE_CONFIG_DIR / "tickers.json")
    relevance_rules_path: Path = Field(default=_PACKAGE_CONFIG_DIR / "relevance_rules.toml")
    cleaning_rules_path: Path = Field(
        default=_PACKAGE_CONFIG_DIR.parent / "ingestion" / "cleaning" / "cleaning_rules.toml"
    )

    # --- LLM / embeddings ---
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    openai_base_url: str | None = None
    llm_model: str = "gpt-5.6-terra"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3
    embedding_batch_size: int = 64
    enrichment_workers: int = 8

    # --- vector store ---
    qdrant_url: str = "http://localhost:6333"
    # If set, Qdrant runs inside the app and stores data here, with no separate server.
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

    # --- evaluation (claim-support judge and eval-suite rules; not used when serving) ---
    # Resolves to backend/tests/eval/cases.yaml in a source checkout, where evals run.
    eval_cases_path: Path = Field(
        default=_PACKAGE_CONFIG_DIR.parents[2] / "tests" / "eval" / "cases.yaml"
    )
    claim_judge_model: str = "gpt-5.6-terra"
    # Unset by default because GPT 5.6 Terra rejects the parameter (HTTP 400); set it to 0
    # for a judge model that accepts it.
    claim_judge_temperature: float | None = None
    # A shorter quote (e.g. just "Intel") would match almost any chunk and prove nothing.
    claim_judge_min_quote_chars: int = 20
    claim_judge_max_concurrency: int = 8
    # Judge calls per sentence, decided by majority; odd so a vote can never tie.
    claim_judge_votes: int = 3
    # The suite-wide mean must reach the threshold; no single case may fall below the floor.
    claim_support_threshold: float = 0.8
    claim_support_floor: float = 0.6
    # Retrieval recall@k over gold articles. Unset k means the live `retrieval_top_k`, so the
    # eval measures what generation actually sees.
    eval_recall_k: int | None = Field(default=None, gt=0)
    eval_min_mean_recall: float = Field(default=0.8, ge=0.0, le=1.0)
    eval_no_data_phrases: list[str] = Field(
        default_factory=lambda: ["no data", "no information", "no news", "none of the articles"]
    )
    eval_valuation_terms: list[str] = Field(
        default_factory=lambda: [
            "market cap",
            "market capitalization",
            "market value",
            "valuation",
            "valued at",
        ]
    )

    @property
    def recall_k(self) -> int:
        """The k used for retrieval recall: the override if set, else the live top-k."""
        return self.eval_recall_k or self.retrieval_top_k

    @field_validator("claim_judge_votes")
    @classmethod
    def _votes_are_odd(cls, votes: int) -> int:
        """Reject even vote counts, which could tie, instead of inventing a tie rule."""
        if votes < 1 or votes % 2 == 0:
            raise ValueError(f"CLAIM_JUDGE_VOTES must be a positive odd number, got {votes}")
        return votes

    @model_validator(mode="after")
    def _floor_not_above_threshold(self) -> "Settings":
        """A per-case floor above the suite threshold would make the threshold meaningless."""
        if self.claim_support_floor > self.claim_support_threshold:
            raise ValueError("CLAIM_SUPPORT_FLOOR must not exceed CLAIM_SUPPORT_THRESHOLD")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the shared settings, reading the environment only once."""
    return Settings()
