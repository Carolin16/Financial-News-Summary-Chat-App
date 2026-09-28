"""Domain models shared by ingestion, retrieval, generation, and the API.

The same Pydantic types serve as the metadata schema, the Qdrant payload shape, and the
API contract, so a field added here is validated consistently end to end.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Sentiment(StrEnum):
    """Overall tone of an article toward its primary company."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ArticleType(StrEnum):
    """Editorial form of an article; lets retrieval down-rank promotional formats."""

    NEWS = "news"
    ANALYSIS = "analysis"
    OPINION = "opinion"
    PRESS_RELEASE = "press_release"
    LISTICLE = "listicle"
    OTHER = "other"


class EventType(StrEnum):
    """What happened in the article; one article may carry several."""

    EARNINGS = "earnings"
    ANALYST_RATING = "analyst_rating"
    PRICE_TARGET = "price_target"
    STOCK_MOVE = "stock_move"
    PRODUCT = "product"
    PARTNERSHIP = "partnership"
    LEGAL_REGULATORY = "legal_regulatory"
    LEADERSHIP = "leadership"
    MACRO = "macro"
    OTHER = "other"


class RawArticle(BaseModel):
    """One entry exactly as it appears in the source file, keyed under `ticker`."""

    model_config = ConfigDict(frozen=True)

    title: str
    link: str
    ticker: str
    full_text: str


class ArticleMetadata(BaseModel):
    """Content-derived metadata. Tickers come from the text, never from the source key."""

    primary_tickers: list[str] = Field(default_factory=list)
    mentioned_tickers: list[str] = Field(default_factory=list)
    event_types: list[EventType] = Field(default_factory=list)
    sentiment: Sentiment = Sentiment.NEUTRAL
    article_type: ArticleType = ArticleType.OTHER

    @property
    def is_relevant(self) -> bool:
        """An article is on-topic only if it is primarily about at least one company."""
        return bool(self.primary_tickers)


class Article(BaseModel):
    """A cleaned, deduplicated article ready for enrichment and chunking."""

    article_id: str
    title: str
    link: str
    text: str
    source_keys: list[str] = Field(
        default_factory=list,
        description="Source-file ticker keys the article was filed under. Provenance only: "
        "never used to decide relevance.",
    )
    merged_links: list[str] = Field(
        default_factory=list,
        description="Links of near-duplicate versions merged into this article (provenance).",
    )
    is_stub: bool = False
    metadata: ArticleMetadata = Field(default_factory=ArticleMetadata)


HEADER_SEPARATOR = "\n\n"


class Chunk(BaseModel):
    """A retrievable unit of an article, carrying the article-level filter fields."""

    chunk_id: str
    article_id: str
    chunk_index: int
    title: str
    link: str
    text: str = Field(description="Chunk body without the contextual header.")
    header: str = Field(description="Article context prepended before embedding.")
    is_stub: bool
    metadata: ArticleMetadata

    @property
    def contextualized_text(self) -> str:
        """Text that is embedded and shown to the LLM: header plus body."""
        return f"{self.header}{HEADER_SEPARATOR}{self.text}"


class RetrievedChunk(BaseModel):
    """A chunk returned by search together with its fused relevance score."""

    chunk: Chunk
    score: float


class SearchFilters(BaseModel):
    """Optional metadata constraints applied inside the vector search."""

    tickers: list[str] = Field(
        default_factory=list, description="Match if any ticker is primary OR mentioned."
    )
    primary_only: bool = False
    include_stubs: bool = True
    event_types: list[EventType] = Field(default_factory=list)
    sentiment: Sentiment | None = None
    article_type: ArticleType | None = None


class Citation(BaseModel):
    """A source the answer relies on, shown to the user as a link."""

    article_id: str
    title: str
    link: str
    is_stub: bool = False


class TokenUsage(BaseModel):
    """Token counts for an LLM call, recorded in per-query logs."""

    input_tokens: int = 0
    output_tokens: int = 0


class SummaryRequest(BaseModel):
    """Everything a summarizer needs: the question, how to answer it, and the sources."""

    question: str
    intent: str
    sources: str = Field(description="Numbered, formatted source blocks.")
    coverage_guidance: str = ""
