"""Indexer + hybrid retriever against Qdrant's embedded local mode (no server needed).

Dense vectors come from a deterministic bag-of-words fake so results are reproducible;
BM25 encoding, payload filtering, and RRF fusion are the real Qdrant/fastembed code paths.
"""

import hashlib
import math
import re
from collections.abc import Sequence

import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient

from chat_app.core.models import Article, ArticleMetadata, SearchFilters
from chat_app.ingestion.chunker import Chunker
from chat_app.ingestion.indexer import QdrantChunkIndex
from chat_app.retrieval.encoders import Bm25SparseEncoder
from chat_app.retrieval.qdrant_retriever import QdrantHybridRetriever

COLLECTION = "test_news"
DIMENSIONS = 64


class BagOfWordsEmbedder:
    """Hashes words into buckets; similar wording -> similar vectors."""

    dimensions = DIMENSIONS

    def __init__(self) -> None:
        self.embedded_texts = 0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded_texts += len(texts)
        return [self._vector(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vector(text)


def article(article_id: str, title: str, text: str, primary=(), mentioned=(), is_stub=False):
    return Article(
        article_id=article_id,
        title=title,
        link=f"https://news.example/{article_id}",
        text=text,
        is_stub=is_stub,
        metadata=ArticleMetadata(primary_tickers=list(primary), mentioned_tickers=list(mentioned)),
    )


ARTICLES = [
    article(
        "dbs",
        "DBS Bank Adjusts NVIDIA Price Target to $160 From $175",
        "NVIDIA has an average rating of Buy and mean price target of $174.93.",
        primary=["NVDA"],
        is_stub=True,
    ),
    article(
        "intel-jump",
        "Intel stock surges on report of Broadcom, TSMC deals",
        "Intel shares jumped after reports that Broadcom and TSMC are exploring deals.",
        primary=["INTC"],
        mentioned=["AVGO", "TSM"],
    ),
    article(
        "market",
        "Wall Street bullish as cash levels drop",
        "Fund managers are bullish. Nvidia and Intel were among stocks discussed.",
        mentioned=["NVDA", "INTC"],
    ),
    article(
        "orman",
        "Suze Orman on marriage finances",
        "Keep separate checking accounts and credit cards.",
    ),
]


@pytest.fixture(scope="module")
def sparse_encoder() -> Bm25SparseEncoder:
    return Bm25SparseEncoder("Qdrant/bm25")


@pytest.fixture
def chunker() -> Chunker:
    return Chunker(500, 1, "cl100k_base", company_name=lambda t: t)


@pytest.fixture
def storage(tmp_path):
    return str(tmp_path / "qdrant")


def index(storage, chunker, sparse_encoder, articles, embedder=None):
    embedder = embedder or BagOfWordsEmbedder()
    client = QdrantClient(path=storage)
    try:
        chunks = [c for a in articles for c in chunker.chunk(a)]
        report = QdrantChunkIndex(client, COLLECTION, embedder, sparse_encoder).sync(chunks)
    finally:
        client.close()  # local mode locks the folder; release it for the next client
    return report, embedder


@pytest.fixture
async def retriever(storage, chunker, sparse_encoder):
    index(storage, chunker, sparse_encoder, ARTICLES)
    client = AsyncQdrantClient(path=storage)
    yield QdrantHybridRetriever(client, COLLECTION, BagOfWordsEmbedder(), sparse_encoder, 20)
    await client.close()


class TestIndexer:
    def test_reindexing_is_idempotent_and_skips_unchanged(self, storage, chunker, sparse_encoder):
        first, _ = index(storage, chunker, sparse_encoder, ARTICLES)
        second, embedder = index(storage, chunker, sparse_encoder, ARTICLES)

        assert first.written == len(ARTICLES)
        assert (second.written, second.unchanged, second.stale_removed) == (0, len(ARTICLES), 0)
        assert embedder.embedded_texts == 0

        client = QdrantClient(path=storage)
        assert client.count(COLLECTION).count == len(ARTICLES)
        client.close()

    def test_changed_article_is_reembedded_and_removed_article_deleted(
        self, storage, chunker, sparse_encoder
    ):
        index(storage, chunker, sparse_encoder, ARTICLES)
        changed = ARTICLES[1].model_copy(update={"text": "Intel shares jumped 16% on Tuesday."})

        report, embedder = index(storage, chunker, sparse_encoder, [ARTICLES[0], changed])

        assert (report.written, report.unchanged, report.stale_removed) == (1, 1, 2)
        assert embedder.embedded_texts == 1


class TestHybridRetriever:
    async def test_exact_tokens_rank_the_matching_article_first(self, retriever):
        results = await retriever.retrieve("DBS $160 target", SearchFilters(), top_k=3)
        assert results[0].chunk.article_id == "dbs"

    async def test_ticker_filter_matches_primary_or_mentioned(self, retriever):
        results = await retriever.retrieve("Intel", SearchFilters(tickers=["INTC"]), top_k=10)
        assert {r.chunk.article_id for r in results} == {"intel-jump", "market"}

    async def test_primary_only_excludes_passing_mentions(self, retriever):
        filters = SearchFilters(tickers=["INTC"], primary_only=True)
        results = await retriever.retrieve("Intel", filters, top_k=10)
        assert {r.chunk.article_id for r in results} == {"intel-jump"}

    async def test_stubs_can_be_excluded(self, retriever):
        filters = SearchFilters(tickers=["NVDA"], include_stubs=False)
        results = await retriever.retrieve("Nvidia price target", filters, top_k=10)
        assert {r.chunk.article_id for r in results} == {"market"}

    async def test_payload_round_trips_to_chunk(self, retriever):
        [top] = await retriever.retrieve(
            "DBS", SearchFilters(tickers=["NVDA"], primary_only=True), 1
        )
        assert top.chunk.is_stub
        assert top.chunk.link == "https://news.example/dbs"
        assert "Title: DBS Bank" in top.chunk.contextualized_text
