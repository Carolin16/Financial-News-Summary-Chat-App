from qdrant_client import models

from chat_app.core.models import ArticleType, EventType, SearchFilters, Sentiment
from chat_app.retrieval.filters import build_filter


def test_no_constraints_means_no_filter():
    assert build_filter(SearchFilters()) is None


def test_tickers_match_primary_or_mentioned():
    result = build_filter(SearchFilters(tickers=["INTC"]))
    [nested] = result.must
    assert {c.key for c in nested.should} == {
        "metadata.primary_tickers",
        "metadata.mentioned_tickers",
    }


def test_primary_only_restricts_to_primary_field():
    [condition] = build_filter(SearchFilters(tickers=["IBM"], primary_only=True)).must
    assert condition.key == "metadata.primary_tickers"
    assert condition.match == models.MatchAny(any=["IBM"])


def test_all_metadata_fields_become_conditions():
    result = build_filter(
        SearchFilters(
            include_stubs=False,
            event_types=[EventType.PRICE_TARGET],
            sentiment=Sentiment.NEGATIVE,
            article_type=ArticleType.NEWS,
        )
    )
    keys = {c.key for c in result.must}
    assert keys == {
        "is_stub",
        "metadata.event_types",
        "metadata.sentiment",
        "metadata.article_type",
    }
