"""Composition root for the API: builds the object graph once per process."""

import asyncio

from qdrant_client import AsyncQdrantClient

from chat_app.config.settings import Settings
from chat_app.core.ticker_registry import TickerRegistry
from chat_app.generation.answer_service import AnswerService
from chat_app.generation.llm_client import OpenAILlmClient
from chat_app.generation.query_analysis import Intent, QueryAnalyzer
from chat_app.generation.scope_guard import LlmScopeGuard
from chat_app.generation.summarizer import LlmSummarizer
from chat_app.retrieval.chunk_catalog import load_chunks
from chat_app.retrieval.coverage import CoverageIndex
from chat_app.retrieval.encoders import Bm25SparseEncoder, OpenAIEmbeddingProvider
from chat_app.retrieval.qdrant_clients import make_async_client
from chat_app.retrieval.qdrant_retriever import QdrantHybridRetriever
from chat_app.retrieval.strategy import CompanyFirstRetrieval

# Extra keyword-search terms per intent: BM25 then favours chunks using the vocabulary
# those questions are answered with, which the user's phrasing often lacks.
QUERY_EXPANSIONS: dict[str, str] = {
    Intent.PRICE_TARGET: "price target analyst rating",
    Intent.ANALYST_VIEW: "analyst rating price target upgrade downgrade",
    Intent.CAUSAL: "shares jumped surged rose report",
    Intent.ADVICE: "analyst rating outlook",
    Intent.PREDICTION: "analyst outlook expects",
}


class VectorStoreUnavailableError(RuntimeError):
    """Raised when the vector store cannot be reached to build the answer service."""


class Container:
    """Holds long-lived clients and builds the answer service lazily when data exists."""

    def __init__(self, settings: Settings) -> None:
        """Create clients; nothing here calls the network."""
        self.settings = settings
        self.registry = TickerRegistry.from_json(settings.tickers_path)
        self.qdrant: AsyncQdrantClient = make_async_client(settings)
        self._embedder = OpenAIEmbeddingProvider(settings)
        self._sparse = Bm25SparseEncoder(settings.sparse_model)
        self._llm = OpenAILlmClient(settings)
        self._service: AnswerService | None = None
        self._lock = asyncio.Lock()

    async def answer_service(self) -> AnswerService:
        """Return the service, building coverage from the index on first use.

        Deferred so the API can start (and report health) before offline indexing has run;
        it is only cached once the index has content.
        """
        if self._service is not None:
            return self._service
        async with self._lock:
            if self._service is None:
                try:
                    chunks = await load_chunks(self.qdrant, self.settings.qdrant_collection)
                except Exception as error:  # any transport/client failure: store unreachable
                    raise VectorStoreUnavailableError(str(error)) from error
                service = self._build_service(
                    CoverageIndex(chunks, self.settings.full_coverage_min_articles)
                )
                if not chunks:
                    return service
                self._service = service
        return self._service

    def _build_service(self, coverage: CoverageIndex) -> AnswerService:
        retriever = QdrantHybridRetriever(
            self.qdrant,
            self.settings.qdrant_collection,
            self._embedder,
            self._sparse,
            self.settings.retrieval_prefetch_k,
        )
        return AnswerService(
            analyzer=QueryAnalyzer(self.registry),
            retrieval=CompanyFirstRetrieval(
                retriever, self.settings.retrieval_top_k, QUERY_EXPANSIONS
            ),
            summarizer=LlmSummarizer(self._llm),
            coverage=coverage,
            registry=self.registry,
            scope_guard=LlmScopeGuard(self._llm),
        )

    async def close(self) -> None:
        """Release network resources."""
        await self.qdrant.close()
