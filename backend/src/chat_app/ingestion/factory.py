"""Composition root for offline ingestion: builds concrete stages from settings."""

from chat_app.config.settings import Settings
from chat_app.core.ticker_registry import TickerRegistry
from chat_app.generation.llm_client import OpenAILlmClient
from chat_app.ingestion.article_pipeline import ArticlePipeline
from chat_app.ingestion.cleaner import TextCleaner
from chat_app.ingestion.deduplicator import Deduplicator
from chat_app.ingestion.enrichment_cache import CachedMetadataExtractor
from chat_app.ingestion.loader import JsonArticleRepository
from chat_app.ingestion.metadata_extractors import (
    FallbackMetadataExtractor,
    HeuristicMetadataExtractor,
    LlmMetadataExtractor,
)
from chat_app.ingestion.stub_detector import StubDetector


def build_metadata_extractor(settings: Settings) -> CachedMetadataExtractor:
    """Cache -> LLM -> heuristic fallback, so indexing works even when the LLM is down."""
    registry = TickerRegistry.from_json(settings.tickers_path)
    llm_extractor = LlmMetadataExtractor(
        OpenAILlmClient(settings), registry, settings.enrichment_max_chars
    )
    heuristic = HeuristicMetadataExtractor(registry, settings.enrichment_fallback_min_mentions)
    return CachedMetadataExtractor(
        FallbackMetadataExtractor(llm_extractor, heuristic), settings.enrichment_cache_path
    )


def build_article_pipeline(
    settings: Settings, extractor: CachedMetadataExtractor
) -> ArticlePipeline:
    """Assemble the article preparation pipeline from configuration."""
    return ArticlePipeline(
        repository=JsonArticleRepository(settings.data_path),
        cleaner=TextCleaner(),
        stub_detector=StubDetector(settings.stub_min_words),
        deduplicator=Deduplicator(
            body_threshold=settings.near_duplicate_threshold,
            title_threshold=settings.title_duplicate_threshold,
            shingle_size=settings.shingle_size,
        ),
        extractor=extractor,
        enrichment_workers=settings.enrichment_workers,
    )
