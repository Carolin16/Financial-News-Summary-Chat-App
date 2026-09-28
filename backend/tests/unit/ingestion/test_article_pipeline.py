"""Pipeline wiring on a fake repository, plus a regression check on the real dataset."""

from pathlib import Path

import pytest

from chat_app.core.models import ArticleMetadata, RawArticle
from chat_app.ingestion.article_pipeline import ArticlePipeline
from chat_app.ingestion.cleaning.config import CleaningConfig
from chat_app.ingestion.cleaning.factory import build_text_cleaner
from chat_app.ingestion.deduplicator import Deduplicator
from chat_app.ingestion.loader import JsonArticleRepository
from chat_app.ingestion.stub_detector import StubDetector

DATASET_PATH = Path(__file__).parents[4] / "data" / "stock_news.json"


class FakeRepository:
    def __init__(self, articles: list[RawArticle]):
        self.articles = articles

    def load(self) -> list[RawArticle]:
        return self.articles


class TickerFromTitle:
    """Stands in for the LLM: primary ticker is the title's first word."""

    def extract(self, article):
        return ArticleMetadata(primary_tickers=[article.title.split()[0].upper()])


def build(repository, stub_min_words: int = 5) -> ArticlePipeline:
    return ArticlePipeline(
        repository=repository,
        cleaner=build_text_cleaner(CleaningConfig.from_toml(), company_terms=[]),
        stub_detector=StubDetector(min_words=stub_min_words),
        deduplicator=Deduplicator(body_threshold=0.6, title_threshold=0.8, shingle_size=5),
        extractor=TickerFromTitle(),
        enrichment_workers=2,
    )


def raw(ticker: str, title: str, link: str, text: str) -> RawArticle:
    return RawArticle(ticker=ticker, title=title, link=link, full_text=text)


def test_pipeline_cleans_dedupes_flags_and_enriches():
    body = "Intel shares rose sharply on Tuesday after reports of deal talks with rivals."
    articles = build(
        FakeRepository(
            [
                raw("INTC", "Intel jumps today", "https://x/1", f"{body} View Comments"),
                raw("AAPL", "Intel jumps today", "https://x/1", body),
                raw("NVDA", "Nvidia teaser", "https://x/2", "Nvidia rallied. Continue Reading"),
            ]
        )
    ).run()

    intel, nvidia = articles
    assert intel.title == "Intel jumps today"
    assert intel.text == body
    assert intel.source_keys == ["INTC", "AAPL"]
    assert intel.metadata.primary_tickers == ["INTEL"]
    assert nvidia.is_stub


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset not available")
def test_real_dataset_profile():
    """Guards the ingestion rules against regressions on the actual feed."""
    result = build(JsonArticleRepository(DATASET_PATH), stub_min_words=80).run()

    assert len(result) == 117  # 118 unique links minus one "Update:" re-publication
    # 23 paywalled/teaser articles, plus one Yahoo video blurb that is thin (77 words) once
    # its "Related Videos" carousel is no longer counted as content.
    assert sum(a.is_stub for a in result) == 24
    for noise in ["275%", "READ NEXT", "Trending:", "Don't Miss", "View Comments", "newsletter"]:
        assert not any(noise in a.text for a in result), noise
