import pytest

from chat_app.ingestion.cleaner import TextCleaner, normalize_characters, repair_mojibake


@pytest.fixture
def cleaner() -> TextCleaner:
    return TextCleaner()


class TestEncodingRepair:
    def test_repairs_fully_misdecoded_text(self):
        assert repair_mojibake("Appleâ€™s shares") == "Apple’s shares"

    def test_falls_back_to_known_sequences_for_mixed_text(self):
        # "★" cannot be encoded as cp1252, so the round trip fails and the map is used.
        assert repair_mojibake("Itâ€™s ★ rated") == "It’s ★ rated"

    def test_cleaned_mojibake_ends_up_as_ascii(self):
        assert normalize_characters("Appleâ€™sÂ shares") == "Apple's shares"

    def test_leaves_clean_text_untouched(self):
        assert repair_mojibake("Plain text, $4.5 billion") == "Plain text, $4.5 billion"

    def test_normalizes_non_breaking_spaces_and_curly_quotes(self):
        assert normalize_characters("Don’t Miss “this”") == 'Don\'t Miss "this"'


class TestNoiseRemoval:
    def test_strips_newsletter_performance_claims(self, cleaner):
        text = (
            "Apple beat estimates. Our quarterly newsletter's strategy selects 14 stocks and "
            "has returned 275% since May 2014. Shares rose."
        )
        cleaned = cleaner.clean(text)
        assert "275%" not in cleaned
        assert cleaned == "Apple beat estimates. Shares rose."

    def test_keeps_genuine_figures_that_share_promo_vocabulary(self, cleaner):
        text = "The company returned over $30 billion to shareholders in the quarter."
        assert cleaner.clean(text) == text

    @pytest.mark.parametrize(
        "promo",
        [
            "READ NEXT: 20 Best AI Stocks To Buy Now.",
            "READ ALSO: 7 Best Stocks to Buy For Long-Term.",
            "Trending: Would you invest in a fund with a 7-9% target yield?",
            "Don't Miss: Deloitte's fastest-growing software company partners with Amazon.",
            "Disclosure: None.",
            "This article is originally published at Insider Monkey.",
            "Sign up for Yahoo Finance's Week in Tech newsletter.",
        ],
    )
    def test_drops_cross_promotion_sentences(self, cleaner, promo):
        assert cleaner.clean(f"Nvidia rallied. {promo} Meta slid.") == "Nvidia rallied. Meta slid."

    def test_removes_paywall_and_ui_chrome(self, cleaner):
        text = (
            "Intel (INTC) has a mean price target of $22.03, according to analysts po PREMIUM "
            "Upgrade to read this MT Newswires article and get so much more. A Silver or Gold "
            "subscription plan is required. Upgrade Already have a subscription? Sign in"
        )
        cleaned = cleaner.clean(text)
        assert "PREMIUM" not in cleaned
        assert "$22.03" in cleaned

    @pytest.mark.parametrize("marker", ["Continue Reading", "View Comments |", "Story Continues"])
    def test_removes_inline_markers(self, cleaner, marker):
        assert marker.split()[0] not in cleaner.clean(f"Intel stock rose Tuesday. {marker}")

    def test_removes_rating_glyphs(self, cleaner):
        assert "★" not in cleaner.clean("Super Micro Computer 29.07% ★★★★★★")

    def test_accepts_custom_patterns(self):
        import re

        custom = TextCleaner(
            inline_noise=[re.compile("ACME AD")], promo_prefixes=[], promo_content=[]
        )
        assert custom.clean("News ACME AD here.") == "News here."
