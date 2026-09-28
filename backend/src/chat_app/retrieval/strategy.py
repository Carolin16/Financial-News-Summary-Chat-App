"""Company-aware retrieval: articles about the company first, passing mentions to fill."""

from collections.abc import Mapping, Sequence

from chat_app.core.interfaces import Retriever
from chat_app.core.models import RetrievedChunk, SearchFilters


class CompanyFirstRetrieval:
    """Wraps a `Retriever` with the policy of preferring primary-subject articles.

    Passing mentions are only used to fill remaining slots, so a well-covered company's
    answer is built from dedicated reporting, while a thinly covered one (e.g. Tesla) can
    still be summarised from what mentions exist.
    """

    def __init__(
        self, retriever: Retriever, top_k: int, query_expansions: Mapping[str, str]
    ) -> None:
        """`query_expansions` maps an intent to extra search terms (e.g. "price target")."""
        self._retriever = retriever
        self._top_k = top_k
        self._expansions = query_expansions

    async def fetch(
        self, question: str, tickers: Sequence[str], intent: str
    ) -> list[RetrievedChunk]:
        """Return up to `top_k` chunks for the question."""
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
