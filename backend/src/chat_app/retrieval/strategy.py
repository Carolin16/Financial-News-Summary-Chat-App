"""Decides how to search: dedicated articles about a company first, mentions only if needed."""

from collections.abc import Mapping, Sequence

from chat_app.core.interfaces import Retriever
from chat_app.core.models import RetrievedChunk, SearchFilters


class CompanyFirstRetrieval:
    """Searches articles mainly about the company first, then fills gaps with passing mentions."""

    def __init__(
        self, retriever: Retriever, top_k: int, query_expansions: Mapping[str, str]
    ) -> None:
        """Take the search engine, how many results to return, and extra words per question type."""
        self._retriever = retriever
        self._top_k = top_k
        self._expansions = query_expansions

    async def fetch(
        self, question: str, tickers: Sequence[str], intent: str
    ) -> list[RetrievedChunk]:
        """Return the best matching chunks for the question, up to the result limit."""
        query = f"{question} {self._expansions.get(intent, '')}".strip()
        if not tickers:
            return await self._retriever.retrieve(query, SearchFilters(), self._top_k)

        primary = await self._retriever.retrieve(
            query, SearchFilters(tickers=list(tickers), primary_only=True), self._top_k
        )
        if len(primary) >= self._top_k:
            return primary
        seen = {r.chunk.chunk_id for r in primary}
        with_mentions = await self._retriever.retrieve(
            query, SearchFilters(tickers=list(tickers)), self._top_k
        )
        extra = [r for r in with_mentions if r.chunk.chunk_id not in seen]
        return primary + extra[: self._top_k - len(primary)]
