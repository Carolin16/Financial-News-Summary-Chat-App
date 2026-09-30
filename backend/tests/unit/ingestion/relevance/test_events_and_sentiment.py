"""Event keywords (checked against the reference queries that need them) and tone."""

import pytest

from chat_app.config.relevance import RelevanceConfig
from chat_app.core.models import EventType, Sentiment
from chat_app.ingestion.relevance.events import KeywordEventTagger, LexiconSentimentScorer

RULES = RelevanceConfig.from_toml()


@pytest.fixture(scope="module")
def tagger() -> KeywordEventTagger:
    return KeywordEventTagger(RULES.events)


@pytest.fixture(scope="module")
def tone() -> LexiconSentimentScorer:
    return LexiconSentimentScorer(RULES.sentiment)


class TestEventsForReferenceQueries:
    @pytest.mark.parametrize(
        ("title", "sentence"),
        [
            # Q6 "Why did Intel stock jump?": every source must read as stock move + deal.
            (
                "Intel (INTC) Stock Trades Up, Here Is Why",
                "Shares of Intel (NASDAQ:INTC) jumped 10.5% after the WSJ reported that the "
                "company is in talks with Broadcom and TSMC to sell certain assets.",
            ),
            (
                "Intel stock surges 10% because TSMC and Broadcom both might buy a piece of it",
                "One plan could see TSMC acquire Intel's manufacturing business.",
            ),
            (
                "Intel Shares Jump After Report on Potential Deals to Split Company",
                "Intel shares rose after the report.",
            ),
        ],
    )
    def test_q6_intel_jump_is_stock_move_and_deal(self, tagger, title, sentence):
        events = tagger.tag(title, [sentence])
        assert {EventType.STOCK_MOVE, EventType.DEAL} <= set(events)

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("DBS Bank Adjusts NVIDIA Price Target to $160 From $175", {EventType.PRICE_TARGET}),
            (
                "Citic Securities Downgrades Intel to Hold From Buy, Price Target is $24",
                {EventType.PRICE_TARGET, EventType.ANALYST_RATING},
            ),
            (
                "Cantor Fitzgerald Adjusts Price Target on Intel to $29 From $22, "
                "Maintains Neutral Rating",
                {EventType.PRICE_TARGET, EventType.ANALYST_RATING},
            ),
        ],
    )
    def test_q4_q5_price_targets_and_ratings(self, tagger, title, expected):
        assert expected <= set(tagger.tag(title, []))

    def test_executive_mentions_are_not_leadership_changes(self, tagger):
        sentence = "Frank Yeary, Intel's executive chair, met TSMC officials about the deal."
        assert EventType.LEADERSHIP not in tagger.tag("Intel talks", [sentence])

    def test_output_is_in_stable_enum_order(self, tagger):
        events = tagger.tag("Intel shares jumped on a merger report", ["The price target rose."])
        assert events == sorted(events, key=list(EventType).index)


class TestSentiment:
    @pytest.mark.parametrize(
        ("sentences", "expected"),
        [
            (
                ["Shares surged after results beat estimates and margins improved."],
                Sentiment.POSITIVE,
            ),
            (
                ["Shares plunged after it missed estimates and cut guidance amid weak demand."],
                Sentiment.NEGATIVE,
            ),
            (["The company held its annual meeting in Austin."], Sentiment.NEUTRAL),
            (
                ["Profit grew and sales surged, but losses widened, risks rose and shares fell."],
                Sentiment.MIXED,
            ),
        ],
    )
    def test_labels(self, tone, sentences, expected):
        assert tone.score(sentences) is expected

    def test_inflections_match_the_lexicon(self, tone):
        assert tone.score(["It surges, rallies and jumped."]) is Sentiment.POSITIVE

    def test_negation_flips_polarity(self, tone):
        assert (
            tone.score(["It did not beat, did not surge and was not strong."]) is Sentiment.NEGATIVE
        )

    def test_known_limitation_lexical_tone_misreads_bearish_advice(self, tone):
        # "Fade the rally" is bearish advice, but the words read positive. Tone is therefore
        # informational metadata only and never a default search filter.
        assert (
            tone.score(["Fade the rally: shares surged and gained, a chip analyst said."])
            is Sentiment.POSITIVE
        )
