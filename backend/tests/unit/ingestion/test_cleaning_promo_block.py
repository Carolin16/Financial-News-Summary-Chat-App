"""PromoBlockScanner and FooterGuard in isolation."""

import pytest

from chat_app.ingestion.cleaning.footer_guard import FooterGuard
from chat_app.ingestion.cleaning.promo_block import PromoBlockScanner


@pytest.fixture(scope="module")
def scanner(cleaning_config) -> PromoBlockScanner:
    return PromoBlockScanner(cleaning_config.promo_block)


def remainder(scanner, text: str, marker: str, allow_foreign: bool = False) -> str:
    end = scanner.block_end(text, text.index(marker) + len(marker), allow_foreign)
    return text[end:].strip()


@pytest.mark.parametrize(
    ("text", "marker", "expected"),
    [
        # headline mode: ends where lowercase prose begins, backing up to "The"
        (
            "READ ALSO: Top 14 AI Stocks on Wall Street The startup may win.",
            "READ ALSO:",
            "The startup may win.",
        ),
        # headline mode: ends at a known next marker
        (
            "READ NEXT: 20 Best AI Stocks To Buy Now Disclosure: None.",
            "READ NEXT:",
            "Disclosure: None.",
        ),
        # sentence mode: ends at the sentence end
        (
            "Trending: Would you invest in a fund like this? Why It Matters: rules.",
            "Trending:",
            "Why It Matters: rules.",
        ),
        # sentence mode: unpunctuated boundary before a sentence opener
        (
            "Read more: Google ends hiring targets tied to diversity It's part of a trend.",
            "Read more:",
            "It's part of a trend.",
        ),
    ],
)
def test_block_end(scanner, text, marker, expected):
    assert remainder(scanner, text, marker) == expected


def test_proper_noun_runs_do_not_end_a_sentence_promo(scanner):
    text = "Trending: If there was a new fund backed by Jeff Bezos offering yields would you? Next."
    assert remainder(scanner, text, "Trending:") == "Next."


def test_foreign_headlines_skipped_only_when_allowed(scanner):
    text = (
        "Most Read: Big News Today Por qué Barcelona compró el edificio Old Firms Rise "
        "Companies like Cisco are winning."
    )
    assert remainder(scanner, text, "Most Read:", allow_foreign=True).startswith("Companies like")
    assert remainder(scanner, text, "Most Read:").startswith("Por qu")


def test_word_cap_bounds_runaway_blocks(scanner, cleaning_config):
    words = " ".join(["Headline"] * (cleaning_config.promo_block.max_words + 20))
    text = f"READ NEXT: {words}"
    kept = remainder(scanner, text, "READ NEXT:")
    assert len(kept.split()) == 20


class TestFooterGuard:
    @pytest.fixture
    def guard(self, cleaning_config) -> FooterGuard:
        return FooterGuard(cleaning_config.footer_safety, ["Intel", "INTC", "Apple"])

    def test_boilerplate_only_tail_has_no_content(self, guard, snippets):
        tail = snippets["motley_fool_footer"].split("investors. ", 1)[1]
        assert guard.content_sentences(tail, title="25 Top AI Stocks") == []

    @pytest.mark.parametrize(
        "sentence",
        [
            "Intel reported quarterly revenue that beat estimates.",
            "Revenue grew 12% year over year in the quarter.",
            "Shares of (NASDAQ:INTC) rose sharply after the report.",
        ],
    )
    def test_company_ticker_or_number_in_full_sentence_is_content(self, guard, sentence):
        assert guard.content_sentences(sentence) == [sentence]

    @pytest.mark.parametrize(
        "text",
        [
            "Apple Inc. (AAPL) : Free Stock Analysis Report",  # boilerplate
            "Read More on AAPL: Apple restores TikTok",  # headline fragment, no terminator
            "Should You Buy Apple Stock Before Feb. 19?",  # echo of the article title
        ],
    )
    def test_boilerplate_fragments_and_title_echoes_are_not_content(self, guard, text):
        assert (
            guard.content_sentences(text, title="Should You Buy Apple Stock Before Feb. 19?") == []
        )
