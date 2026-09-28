"""Abstractions that the pipeline depends on (DIP).

Concrete adapters (JSON file, Qdrant, OpenAI) implement these, so core logic and tests
can swap them without touching callers. Protocols keep test fakes free of inheritance.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from pydantic import BaseModel

from chat_app.core.models import (
    Article,
    ArticleMetadata,
    Chunk,
    RawArticle,
    RetrievedChunk,
    SearchFilters,
    SummaryRequest,
    TokenUsage,
)


class StructuredLlm(Protocol):
    """An LLM that returns output validated against a Pydantic schema."""

    def parse[T: BaseModel](self, instructions: str, prompt: str, schema: type[T]) -> T:
        """Return the model's answer as an instance of `schema`."""
        ...


class AsyncStructuredLlm(Protocol):
    """Async variant of `StructuredLlm`, for use inside request handlers."""

    async def aparse[T: BaseModel](self, instructions: str, prompt: str, schema: type[T]) -> T:
        """Return the model's answer as an instance of `schema`."""
        ...


class ScopeGuard(Protocol):
    """Decides whether a question is within the assistant's purpose."""

    async def is_in_scope(self, question: str) -> bool:
        """True if the question should be answered."""
        ...


class StreamingLlm(Protocol):
    """An LLM that streams free-text answers."""

    def stream_text(self, instructions: str, prompt: str, usage: TokenUsage) -> AsyncIterator[str]:
        """Yield text fragments; fill `usage` once the response completes."""
        ...


class ArticleRepository(Protocol):
    """Source of raw articles (a JSON file today; an API or database later)."""

    def load(self) -> list[RawArticle]:
        """Return every raw entry, including duplicates, in source order."""
        ...


class MetadataExtractor(Protocol):
    """Derives content-based metadata (tickers, events, sentiment) for an article."""

    def extract(self, article: Article) -> ArticleMetadata:
        """Return metadata judged from the article text, not its source key."""
        ...


class EmbeddingProvider(Protocol):
    """Turns text into dense vectors."""

    @property
    def dimensions(self) -> int:
        """Length of every vector this provider returns."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving order (offline)."""
        ...

    async def aembed_query(self, text: str) -> list[float]:
        """Embed one query (request time)."""
        ...


class ChunkIndex(Protocol):
    """Write side of the vector store, used only by offline indexing."""

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """Insert or replace chunks by their deterministic ID; return how many were written."""
        ...


class Retriever(Protocol):
    """Read side of the vector store, used at query time."""

    async def retrieve(
        self, query: str, filters: SearchFilters, top_k: int
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks for `query` that satisfy `filters`."""
        ...


class Summarizer(Protocol):
    """Produces a grounded answer from numbered sources."""

    def stream(self, request: SummaryRequest, usage: TokenUsage) -> AsyncIterator[str]:
        """Yield answer text incrementally, using only the request's sources."""
        ...
