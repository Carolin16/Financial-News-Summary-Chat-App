import pytest
from pydantic import BaseModel

from chat_app.core.models import ArticleMetadata, EventType, Sentiment
from chat_app.core.ticker_registry import Company, TickerRegistry
from chat_app.generation.llm_client import LlmError
from chat_app.ingestion.enrichment_cache import CachedMetadataExtractor
from chat_app.ingestion.metadata_extractors import (
    FallbackMetadataExtractor,
    HeuristicMetadataExtractor,
    LlmMetadataExtractor,
    canonicalize_tickers,
)


@pytest.fixture
def registry() -> TickerRegistry:
    return TickerRegistry(
        {
            "AAPL": Company(name="Apple", aliases=["Apple"]),
            "INTC": Company(name="Intel", aliases=["Intel"]),
            "GOOGL": Company(name="Alphabet", aliases=["Google", "GOOG"]),
        }
    )


class FakeLlm:
    def __init__(self, result: ArticleMetadata | Exception):
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def parse[T: BaseModel](self, instructions: str, prompt: str, schema: type[T]) -> T:
        self.calls.append((instructions, prompt))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result  # type: ignore[return-value]


class CountingExtractor:
    def __init__(self, metadata: ArticleMetadata):
        self.metadata = metadata
        self.calls = 0

    def extract(self, article):
        self.calls += 1
        return self.metadata


class TestLlmExtractor:
    def test_sends_title_and_text_and_lists_tracked_tickers(self, registry, make_article):
        llm = FakeLlm(ArticleMetadata(primary_tickers=["INTC"]))
        LlmMetadataExtractor(llm, registry, max_chars=1000).extract(
            make_article(title="Intel jumps", text="Body text")
        )
        instructions, prompt = llm.calls[0]
        assert "INTC (Intel)" in instructions
        assert prompt.startswith("Title: Intel jumps") and "Body text" in prompt

    def test_truncates_long_articles(self, registry, make_article):
        llm = FakeLlm(ArticleMetadata())
        LlmMetadataExtractor(llm, registry, max_chars=10).extract(make_article(text="x" * 100))
        assert llm.calls[0][1].endswith("x" * 10)
        assert "x" * 11 not in llm.calls[0][1]

    def test_canonicalizes_llm_tickers(self, registry, make_article):
        llm = FakeLlm(
            ArticleMetadata(primary_tickers=["GOOG", "intc"], mentioned_tickers=["Google"])
        )
        metadata = LlmMetadataExtractor(llm, registry, max_chars=100).extract(make_article())
        assert metadata.primary_tickers == ["GOOGL", "INTC"]
        assert metadata.mentioned_tickers == []


def test_canonicalize_keeps_primary_and_mentioned_disjoint(registry):
    metadata = ArticleMetadata(primary_tickers=["AAPL"], mentioned_tickers=["AAPL", "GOOG", "TSM"])
    result = canonicalize_tickers(metadata, registry)
    assert result.mentioned_tickers == ["GOOGL", "TSM"]


class TestHeuristicExtractor:
    def test_title_mention_makes_company_primary(self, registry, make_article):
        article = make_article(title="Intel shares jump", text="Chipmaker rallied. Apple was flat.")
        metadata = HeuristicMetadataExtractor(registry, min_primary_mentions=3).extract(article)
        assert metadata.primary_tickers == ["INTC"]
        assert metadata.mentioned_tickers == ["AAPL"]

    def test_frequent_body_mentions_make_company_primary(self, registry, make_article):
        article = make_article(title="Chip news", text="Intel rose. Intel said. Intel plans.")
        metadata = HeuristicMetadataExtractor(registry, min_primary_mentions=3).extract(article)
        assert metadata.primary_tickers == ["INTC"]

    def test_off_topic_article_has_no_primary(self, registry, make_article):
        article = make_article(title="Suze Orman on marriage", text="Keep separate accounts.")
        metadata = HeuristicMetadataExtractor(registry, min_primary_mentions=3).extract(article)
        assert not metadata.is_relevant


class TestFallback:
    @pytest.mark.parametrize("error", [LlmError("timeout"), LlmError("no output")])
    def test_degrades_to_fallback_on_llm_failure(self, make_article, error):
        fallback = CountingExtractor(ArticleMetadata(primary_tickers=["INTC"]))
        extractor = FallbackMetadataExtractor(
            LlmMetadataExtractor(FakeLlm(error), TickerRegistry({}), 100), fallback
        )
        assert extractor.extract(make_article()).primary_tickers == ["INTC"]
        assert fallback.calls == 1

    def test_uses_primary_when_it_succeeds(self, make_article):
        primary = CountingExtractor(ArticleMetadata(sentiment=Sentiment.POSITIVE))
        fallback = CountingExtractor(ArticleMetadata())
        result = FallbackMetadataExtractor(primary, fallback).extract(make_article())
        assert result.sentiment is Sentiment.POSITIVE
        assert fallback.calls == 0


class TestCache:
    def test_reuses_cached_metadata_across_instances(self, tmp_path, make_article):
        path = tmp_path / "cache.json"
        inner = CountingExtractor(ArticleMetadata(event_types=[EventType.EARNINGS]))
        first = CachedMetadataExtractor(inner, path)
        first.extract(make_article())
        first.save()

        second = CachedMetadataExtractor(inner, path)
        result = second.extract(make_article())

        assert inner.calls == 1
        assert result.event_types == [EventType.EARNINGS]
        assert (second.hits, second.misses) == (1, 0)

    def test_changed_content_invalidates_entry(self, tmp_path, make_article):
        inner = CountingExtractor(ArticleMetadata())
        cache = CachedMetadataExtractor(inner, tmp_path / "cache.json")
        cache.extract(make_article(text="version one"))
        cache.extract(make_article(text="version two"))
        assert inner.calls == 2
