"""The enricher's composition rules, with the real rules config and small texts."""

import pytest

from chat_app.config.relevance import RelevanceConfig
from chat_app.config.settings import Settings
from chat_app.core.models import Article, ArticleType, RelevanceTier
from chat_app.core.ticker_registry import TickerRegistry
from chat_app.ingestion.relevance.article_type import CueArticleTypeClassifier
from chat_app.ingestion.relevance.enricher import RuleBasedEnricher
from chat_app.ingestion.relevance.entities import RegistryEntityMatcher
from chat_app.ingestion.relevance.events import KeywordEventTagger, LexiconSentimentScorer
from chat_app.ingestion.relevance.salience import WeightedSalienceScorer

RULES = RelevanceConfig.from_toml()
REGISTRY = TickerRegistry.from_json(Settings().tickers_path)


def enricher() -> RuleBasedEnricher:
    return RuleBasedEnricher(
        matcher=RegistryEntityMatcher(REGISTRY, RULES.entities),
        scorer=WeightedSalienceScorer(RULES.salience),
        classifier=CueArticleTypeClassifier(RULES.article_type),
        event_tagger=KeywordEventTagger(RULES.events),
        sentiment_scorer=LexiconSentimentScorer(RULES.sentiment),
        tiers=RULES.tiers,
        enumeration_min_companies=RULES.entities.enumeration_min_companies,
    )


def article(title: str, text: str) -> Article:
    return Article(article_id="a", title=title, link="l", text=text)


INTEL = article(
    "Intel stock surges on report of Broadcom, TSMC exploring deals",
    "Intel (NASDAQ:INTC) shares jumped after reports Broadcom and TSMC are exploring deals. "
    "Intel has been restructuring. Intel said it does not comment on rumours. Analysts said "
    "Intel could unlock value.",
)


def test_labels_primary_mentioned_type_events():
    metadata = enricher().extract(INTEL)
    assert metadata.primary_tickers == ["INTC"]
    assert set(metadata.mentioned_tickers) == {"AVGO", "TSM"}
    assert metadata.tiers["INTC"] is RelevanceTier.PRIMARY
    assert metadata.salience["INTC"] > metadata.salience["TSM"]
    assert metadata.article_type is ArticleType.NEWS
    assert {"stock_move", "deal"} <= {e.value for e in metadata.event_types}


def test_off_topic_article_is_kept_and_flagged():
    metadata = enricher().extract(
        article("Suze Orman on marriage", "Keep separate checking accounts and credit cards.")
    )
    assert metadata.primary_tickers == [] and not metadata.is_relevant


def test_passing_prose_mention_stays_mentioned():
    text = "McDonald's (NYSE:MCD) has higher margins. " * 3 + "Tesla trails on profitability."
    metadata = enricher().extract(article("McDonald's margins", text))
    assert "TSLA" in metadata.mentioned_tickers


def test_press_release_side_companies_are_incidental():
    text = (
        "SANTA CLARA, Calif.--(BUSINESS WIRE)--Cohesity today announced a new CMO. "
        "Backed by NVIDIA, IBM, Cisco, and others, Cohesity is headquartered in Santa Clara."
    )
    metadata = enricher().extract(article("Cohesity Appoints Carol Carpenter as CMO", text))
    assert metadata.article_type is ArticleType.PRESS_RELEASE
    assert "IBM" in metadata.incidental_tickers and "IBM" not in metadata.mentioned_tickers


def test_listicle_needs_stronger_evidence():
    text = (
        "We recently compiled a list of AI stocks. Nvidia (NASDAQ:NVDA) makes GPUs. "
        "Intel (NASDAQ:INTC) makes CPUs. Apple (NASDAQ:AAPL) sells phones."
    )
    metadata = enricher().extract(article("10 Best AI Stocks To Buy Now", text))
    assert metadata.article_type is ArticleType.LISTICLE
    assert metadata.primary_tickers == []


@pytest.mark.parametrize("stub_text", ["Intel rose.", "Intel rose. Continue Reading"])
def test_stub_status_is_not_an_input(stub_text):
    # The enricher never sees is_stub; short text is judged on its content alone.
    metadata = enricher().extract(article("Intel Shares Jump After Report", stub_text))
    assert metadata.primary_tickers == ["INTC"]
