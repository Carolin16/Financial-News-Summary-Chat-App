"""Offline preparation: load -> clean -> flag stubs -> deduplicate -> enrich metadata."""

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from chat_app.core.ids import article_id_for
from chat_app.core.interfaces import ArticleRepository, MetadataExtractor
from chat_app.core.models import Article, RawArticle
from chat_app.ingestion.cleaner import TextCleaner, normalize_characters
from chat_app.ingestion.deduplicator import Deduplicator
from chat_app.ingestion.stub_detector import StubDetector

logger = logging.getLogger(__name__)


class ArticlePipeline:
    """Turns raw feed entries into unique, cleaned, metadata-enriched articles."""

    def __init__(
        self,
        repository: ArticleRepository,
        cleaner: TextCleaner,
        stub_detector: StubDetector,
        deduplicator: Deduplicator,
        extractor: MetadataExtractor,
        enrichment_workers: int,
    ) -> None:
        """Wire the pipeline stages; each is independently replaceable and testable."""
        self._repository = repository
        self._cleaner = cleaner
        self._stub_detector = stub_detector
        self._deduplicator = deduplicator
        self._extractor = extractor
        self._enrichment_workers = enrichment_workers

    def run(self) -> list[Article]:
        """Execute every stage and return articles ready for chunking."""
        raw_articles = self._repository.load()
        cleaned = [self._to_article(raw) for raw in raw_articles]
        unique = self._deduplicator.deduplicate(cleaned)
        enriched = self._enrich(unique)
        logger.info(
            "prepared articles: raw=%d unique=%d stubs=%d off_topic=%d",
            len(raw_articles),
            len(enriched),
            sum(a.is_stub for a in enriched),
            sum(not a.metadata.is_relevant for a in enriched),
        )
        return enriched

    def _to_article(self, raw: RawArticle) -> Article:
        text = self._cleaner.clean(raw.full_text)
        return Article(
            article_id=article_id_for(raw.link),
            title=normalize_characters(raw.title).strip(),
            link=raw.link.strip(),
            text=text,
            source_tickers=[raw.ticker],
            is_stub=self._stub_detector.is_stub(raw.full_text, text),
        )

    def _enrich(self, articles: Sequence[Article]) -> list[Article]:
        # Extraction is I/O-bound (one LLM call per uncached article), so threads suffice.
        with ThreadPoolExecutor(max_workers=self._enrichment_workers) as pool:
            metadata = list(pool.map(self._extractor.extract, articles))
        return [
            article.model_copy(update={"metadata": meta})
            for article, meta in zip(articles, metadata, strict=True)
        ]
