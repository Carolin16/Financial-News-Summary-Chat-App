"""Offline indexing entry point: `chat-index` (or `python scripts/index_news.py`).

Safe to re-run: unchanged chunks are skipped, changed ones re-embedded, removed ones deleted.
"""

import logging

from chat_app.config.logging import configure_logging
from chat_app.config.settings import get_settings
from chat_app.core.ticker_registry import TickerRegistry
from chat_app.ingestion.chunker import Chunker
from chat_app.ingestion.enrichment_cache import CachedMetadataExtractor
from chat_app.ingestion.factory import build_article_pipeline, build_metadata_extractor
from chat_app.ingestion.indexer import QdrantChunkIndex
from chat_app.retrieval.encoders import Bm25SparseEncoder, OpenAIEmbeddingProvider
from chat_app.retrieval.qdrant_clients import make_client

logger = logging.getLogger(__name__)


def main() -> None:
    """Prepare articles, chunk them, and sync the vector index."""
    settings = get_settings()
    configure_logging(settings.log_level)

    extractor = build_metadata_extractor(settings)
    articles = build_article_pipeline(settings, extractor).run()
    _save_cache(extractor)

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
            "enrichment_cache_hits": extractor.hits,
            "enrichment_cache_misses": extractor.misses,
        },
    )


def _save_cache(extractor: CachedMetadataExtractor) -> None:
    if extractor.misses == 0:
        return
    try:
        extractor.save()
    except OSError as error:
        # In Docker the data volume is read-only; indexing still succeeds without the cache.
        logger.warning("could not persist enrichment cache: %s", error)


if __name__ == "__main__":
    main()
