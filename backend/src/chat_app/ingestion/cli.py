"""Loads the news into the search database, and re-running only updates what changed."""

import logging

from chat_app.config.logging import configure_logging
from chat_app.config.settings import get_settings
from chat_app.core.ticker_registry import TickerRegistry
from chat_app.ingestion.chunker import Chunker
from chat_app.ingestion.factory import build_article_pipeline, build_enricher
from chat_app.ingestion.indexer import QdrantChunkIndex
from chat_app.retrieval.encoders import Bm25SparseEncoder, OpenAIEmbeddingProvider
from chat_app.retrieval.qdrant_clients import make_client

logger = logging.getLogger(__name__)


def main() -> None:
    """Prepare the articles, split them into chunks, and bring Qdrant up to date."""
    settings = get_settings()
    configure_logging(settings.log_level)

    articles = build_article_pipeline(settings, build_enricher(settings)).run()

    registry = TickerRegistry.from_json(settings.tickers_path)
    chunker = Chunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_sentences=settings.chunk_overlap_sentences,
        encoding_name=settings.tokenizer_encoding,
        company_name=registry.name_for,
    )
    chunks = [chunk for article in articles for chunk in chunker.chunk(article)]

    client = make_client(settings)
    try:
        index = QdrantChunkIndex(
            client,
            settings.qdrant_collection,
            OpenAIEmbeddingProvider(settings),
            Bm25SparseEncoder(settings.sparse_model),
        )
        report = index.sync(chunks)
    finally:
        client.close()
    logger.info(
        "indexing complete",
        extra={
            "articles": len(articles),
            "chunks": len(chunks),
            "written": report.written,
            "unchanged": report.unchanged,
            "stale_removed": report.stale_removed,
        },
    )


if __name__ == "__main__":
    main()
