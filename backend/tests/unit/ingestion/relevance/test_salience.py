"""Salience scoring on hand-built mentions, so each feature is tested in isolation."""

import pytest

from chat_app.config.relevance import RelevanceConfig
from chat_app.core.ticker_registry import MentionForm
from chat_app.ingestion.relevance.entities import EntityMention
from chat_app.ingestion.relevance.salience import WeightedSalienceScorer

CONFIG = RelevanceConfig.from_toml().salience


def mention(ticker: str, start: int, tagged: bool = False) -> EntityMention:
    return EntityMention(ticker, start, start + 1, MentionForm.NAME, tagged)


def words(n: int) -> list[int]:
    return [i * 6 for i in range(n)]  # a word every 6 characters


@pytest.fixture
def scorer() -> WeightedSalienceScorer:
    return WeightedSalienceScorer(CONFIG)


def test_headline_subject_beats_other_headline_names(scorer):
    title = [mention("INTC", 0), mention("TSM", 30), mention("AVGO", 40)]
    body = [
        mention("INTC", 0),
        *(mention("INTC", 60 * i) for i in range(1, 8)),
        mention("TSM", 300),
    ]
    scores = scorer.score(title, body, words(200))
    assert scores["INTC"].features.title_subject
    assert scores["INTC"].score > scores["TSM"].score > scores["AVGO"].score
    assert scores["INTC"].score >= 0.5 > scores["TSM"].score


def test_headline_subject_with_lead_mention_reaches_primary(scorer):
    scores = scorer.score([mention("MSFT", 0)], [mention("MSFT", 10)], words(20))
    assert scores["MSFT"].score >= 0.5


def test_share_is_relative_to_the_leader(scorer):
    # Two companies discussed equally: neither is capped at ~50% share.
    body = [mention("GOOGL", i * 20) for i in range(5)] + [
        mention("MSFT", 200 + i * 20) for i in range(5)
    ]
    scores = scorer.score([], body, words(300))
    assert scores["GOOGL"].features.relative_share == scores["MSFT"].features.relative_share == 1.0


def test_single_mention_does_not_count_as_frequent(scorer):
    scores = scorer.score([], [mention("MSFT", 5)], words(15))  # 1 mention in a 15-word teaser
    assert scores["MSFT"].features.mentions_per_100_words > CONFIG.frequency_cap_per_100_words
    assert scores["MSFT"].score < 0.5


def test_exchange_tag_and_lead_add_evidence(scorer):
    plain = scorer.score([], [mention("X", 5000), mention("Y", 10)], words(1000))["X"].score
    tagged = scorer.score([], [mention("X", 10, tagged=True), mention("Y", 20)], words(1000))[
        "X"
    ].score
    assert tagged > plain


def test_scores_are_bounded_and_deterministic(scorer):
    title = [mention("INTC", 0)]
    body = [mention("INTC", i, tagged=True) for i in range(0, 600, 6)]
    first = scorer.score(title, body, words(100))["INTC"].score
    assert first == scorer.score(title, body, words(100))["INTC"].score
    assert 0.0 <= first <= 1.0
