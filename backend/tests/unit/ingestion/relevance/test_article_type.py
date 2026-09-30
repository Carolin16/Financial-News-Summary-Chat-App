"""Article-type cues, using real titles and leads from the dataset."""

import pytest

from chat_app.config.relevance import RelevanceConfig
from chat_app.core.models import ArticleType
from chat_app.ingestion.relevance.article_type import CueArticleTypeClassifier


@pytest.fixture(scope="module")
def classifier() -> CueArticleTypeClassifier:
    return CueArticleTypeClassifier(RelevanceConfig.from_toml().article_type)


@pytest.mark.parametrize(
    ("title", "lead", "expected"),
    [
        ("DBS Bank Adjusts NVIDIA Price Target to $160 From $175", "", ArticleType.ANALYST_NOTE),
        (
            "Citic Securities Downgrades Intel to Hold From Buy, Price Target is $24",
            "",
            ArticleType.ANALYST_NOTE,
        ),
        ("Nvidia gains, Baidu earnings, Walgreens: Market Minute", "", ArticleType.MARKET_WRAP),
        ("Sector Update: Tech", "", ArticleType.MARKET_WRAP),
        (
            "Cohesity Appoints Carol Carpenter as Chief Marketing Officer",
            "SANTA CLARA, Calif., February 18, 2025--(BUSINESS WIRE)--Cohesity today announced",
            ArticleType.PRESS_RELEASE,
        ),
        (
            "Clinical Data Analytics Market to Reach $614.7 Billion by 2034",
            "",
            ArticleType.PRESS_RELEASE,
        ),
        ("25 Top AI Stocks That Could Boost Your Portfolio", "", ArticleType.LISTICLE),
        (
            "Jim Cramer on Amazon.com (AMZN): 'Knock Yourself Out'",
            "We recently compiled a list of the Jim Cramer's Bold Predictions.",
            ArticleType.LISTICLE,
        ),
        ("Should You Buy Apple Stock Hand Over Fist Before Feb. 19?", "", ArticleType.OPINION),
        (
            "Prediction: These 2 Quantum Computing Stocks Will Be the Biggest AI Winners",
            "",
            ArticleType.OPINION,
        ),
        ("Intel stock surges on report of Broadcom, TSMC exploring deals", "", ArticleType.NEWS),
    ],
)
def test_real_titles(classifier, title, lead, expected):
    assert classifier.classify(title, lead).article_type is expected


def test_matched_cue_is_recorded_for_audit(classifier):
    verdict = classifier.classify("DBS Bank Adjusts NVIDIA Price Target to $160 From $175", "")
    assert verdict.cue == "Price Target"
    assert classifier.classify("Intel shares jump", "Intel rose.").cue is None


def test_priority_resolves_overlaps(classifier):
    # Zacks blog posts are press releases even though "Highlights A, B and C" is a list cue.
    verdict = classifier.classify(
        "The Zacks Analyst Blog Highlights Apple, Eli Lilly, Shopify and ImmuCell",
        "For Immediate Release Chicago, IL - Zacks.com announces the list of stocks",
    )
    assert verdict.article_type is ArticleType.PRESS_RELEASE


def test_lead_cues_only_apply_where_allowed(classifier):
    # "price target" in the body of a news story does not make it an analyst note.
    lead = "Intel shares jumped. One analyst raised his price target to $30."
    assert (
        classifier.classify("Intel shares jump after deal report", lead).article_type
        is ArticleType.NEWS
    )
