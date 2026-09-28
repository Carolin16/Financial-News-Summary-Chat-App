"""Noise-block rules, run through the full cleaner on real snippets."""

import pytest

from chat_app.core.models import RawArticle
from chat_app.ingestion.cleaning.config import CleaningConfig
from chat_app.ingestion.cleaning.factory import build_text_cleaner


def clean(cleaner, text: str, title: str = "Unrelated headline"):
    return cleaner.clean(RawArticle(title=title, link="https://x/1", ticker="AAPL", full_text=text))


class TestTruncationMarkers:
    @pytest.mark.parametrize(
        ("key", "rule", "kept"),
        [
            ("continue_reading_teaser", "continue_reading", "Intel stock rose Tuesday."),
            ("paywall_stub", "paywall_premium", "mean price target of $174.93"),
        ],
    )
    def test_marker_removed_but_reported(self, cleaner, snippets, key, rule, kept):
        result = clean(cleaner, snippets[key])
        assert rule in result.signals.truncation_markers
        assert result.signals.truncation_marker_found
        assert kept in result.text
        assert "Continue Reading" not in result.text and "PREMIUM" not in result.text

    def test_clean_article_has_no_truncation_signal(self, cleaner, snippets):
        assert not clean(cleaner, snippets["curly_quote"]).signals.truncation_marker_found


class TestCrossPromotion:
    def test_headline_list_removed_and_following_prose_kept(self, cleaner, snippets):
        text = clean(cleaner, snippets["read_also_headlines"]).text
        assert "READ ALSO" not in text and "Analysts Are Watching" not in text
        assert "MSCI China to 85 from 75." in text
        assert text.endswith(
            "The Chinese startup may be all the rage for Chinese stocks, but "
            "some countries are exercising caution all the same."
        )

    def test_bloomberg_most_read_list_fully_removed(self, cleaner, snippets):
        # Regression: leftover headlines would inject Trump, NYC, Winnipeg... into relevance.
        text = clean(cleaner, snippets["bloomberg_most_read"]).text
        for headline in [
            "Most Read from Bloomberg",
            "Why Barcelona Bought the Building",
            "Por qué Barcelona compró el edificio",
            "Trump Child Refugee Agency",
            "Surreal Journey Into His Own Private Winnipeg",
            "NYC Restaurants Are Still Waiting",
        ]:
            assert headline not in text, headline
        assert "Companies like Cisco Systems Inc., International Business Machines Corp." in text
        assert "legacy names are drawing renewed attention." in text

    def test_title_case_promo_keeps_sentence_opener_after_it(self, cleaner, snippets):
        text = clean(cleaner, snippets["trending_headline"]).text
        assert "Which Bucket" not in text and "Retirement Accounts" not in text
        assert text.endswith("Instead, Orman recommends that each spouse maintain control")

    @pytest.mark.parametrize(
        ("key", "removed", "kept"),
        [
            (
                "read_more_sentence_case",
                "Google ends hiring targets tied to diversity",
                "It's part of a broader retrenchment across the business community.",
            ),
            (
                "read_more_unpunctuated",
                "Mark Zuckerberg pivots toward the president",
                "Tech companies have pointed to legal risks",
            ),
            (
                "see_also_promo",
                "average American couple has saved",
                "The impact from the February 28 economic blackout may be minimal",
            ),
        ],
    )
    def test_sentence_case_promo_ends_at_unpunctuated_boundary(
        self, cleaner, snippets, key, removed, kept
    ):
        text = clean(cleaner, snippets[key]).text
        assert removed not in text
        assert kept in text

    def test_chained_sponsored_lines_removed_genuine_content_next_to_them_kept(
        self, cleaner, snippets
    ):
        text = clean(cleaner, snippets["dont_miss_chain"]).text
        for promo in ["Don't Miss", "$0.26/share", "$60,000 foldable home", "$0.80 per share"]:
            assert promo not in text, promo
        assert "approximately $64.83 billion." in text
        assert "Hon Hai reported a 3.16% year-over-year increase" in text


class TestPromotionalClaims:
    def test_newsletter_return_claim_removed(self, cleaner, snippets):
        text = clean(cleaner, snippets["newsletter_claim"]).text
        assert "275%" not in text and "newsletter" not in text

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_genuine_return_figures_survive(self, cleaner, snippets, index):
        snippet = snippets["genuine_returns"][index]
        assert clean(cleaner, snippet).text == cleaner.clean_text(snippet)
        assert "returned" in clean(cleaner, snippet).text


class TestFooters:
    def test_longest_real_footer_tail_is_stripped(self, cleaner, snippets):
        result = clean(cleaner, snippets["zacks_longest_footer"])
        assert result.text == (
            "Is Coca-Cola still a buy in 2025? What Else Should You Know About Buffett's Sales "
            "of Apple Shares? Tune into this week's podcast to find out."
        )
        assert result.signals.chars_removed_by_rule["zacks_footer"] > 3000

    def test_motley_fool_footer_stripped(self, cleaner, snippets):
        result = clean(cleaner, snippets["motley_fool_footer"], "25 Top AI Stocks")
        assert result.text == "handsomely reward patient investors."
        assert not result.signals.footers_kept

    GENUINE = "Intel reported revenue of $14.3 billion for the quarter, beating analyst estimates."
    FOOTERS = {
        "simply_wall_st_footer": (
            "Have feedback on this article? Concerned about the content? Get in touch with us "
            "directly. Simply Wall St has no position in any stocks mentioned."
        ),
        "motley_fool_footer": (
            "Should you invest $1,000 in Intel right now? The Motley Fool has a disclosure "
            "policy. The 10 stocks that made the cut could produce monster returns."
        ),
    }

    @pytest.mark.parametrize("rule", sorted(FOOTERS))
    def test_guard_keeps_and_logs_tail_with_genuine_content(self, cleaner, caplog, rule):
        text = f"Intro sentence about chips. {self.FOOTERS[rule]} {self.GENUINE}"
        with caplog.at_level("WARNING", logger="chat_app.ingestion.cleaning.steps.noise_blocks"):
            result = clean(cleaner, text)
        assert result.signals.footers_kept == [rule]
        assert self.GENUINE in result.text
        [record] = [r for r in caplog.records if r.getMessage().startswith("footer kept")]
        assert record.rule == rule
        assert record.content == [self.GENUINE]

    @pytest.mark.parametrize("rule", sorted(FOOTERS))
    def test_same_tail_without_content_is_stripped_silently(self, cleaner, caplog, rule):
        with caplog.at_level("WARNING"):
            result = clean(cleaner, f"Intro sentence about chips. {self.FOOTERS[rule]}")
        assert result.text == "Intro sentence about chips."
        assert not result.signals.footers_kept
        assert not [r for r in caplog.records if r.getMessage().startswith("footer kept")]

    def test_link_text_before_footer_is_promo_residue_not_a_heading(self, cleaner, snippets):
        # Regression for tail #3: "Breaking Down the Current Earnings Outlook" is the link text
        # of a Zacks cross-promo, removed with its sentence; the real last sentence is kept.
        text = clean(cleaner, snippets["zacks_cta_before_footer"]).text
        assert "Breaking Down the Current Earnings Outlook" not in text
        assert "Earnings Trends report" not in text
        assert text.endswith("would be up +8.3% on +4.5% higher revenues.")

    def test_parenthetical_promo_keeps_host_sentence_punctuation(self, cleaner, snippets):
        text = clean(cleaner, snippets["read_more_parenthetical"]).text
        assert "read more" not in text and "Palantir" not in text
        assert "growth for the semiconductor giant. NVIDIA has a Zacks Rank #2" in text


class TestCreditsContactsAndRankings:
    @pytest.mark.parametrize(
        ("key", "credit", "kept"),
        [
            (
                "credit_getty_parenthetical",
                "Getty Images",
                "Intel store at Promenade Street in Davos, Switzerland, on Jan. 21, 2025.",
            ),
            ("credit_slash_getty", "Getty Images", "Nvidia called off"),
            ("credit_image_source_repeated", "Image Source", "CRNC Stock's Performance"),
            ("credit_image_contributor", "Getty Images", "Almost 20 years after Netflix"),
            ("credit_photo_parenthetical", "(Photo:", "Cohesity"),
        ],
    )
    def test_credit_tokens_removed_caption_text_kept(self, cleaner, snippets, key, credit, kept):
        text = clean(cleaner, snippets[key]).text
        assert credit not in text
        assert kept in text

    def test_author_contact_lines_removed(self, cleaner, snippets):
        assert clean(cleaner, snippets["author_contact"]).text == ""

    def test_zacks_contact_block_loses_phone_and_email(self, cleaner, snippets):
        text = clean(cleaner, snippets["zacks_contact_block"]).text
        assert "800-767-3771" not in text and "support@zacks.com" not in text

    def test_ranking_label_kept_by_default(self, cleaner, snippets):
        assert "Zacks Rank #3 (Hold)" in clean(cleaner, snippets["zacks_rank_label"]).text

    def test_ranking_label_stripped_when_configured(self, cleaning_config, company_terms, snippets):
        config = cleaning_config.model_copy(update={"keep_ranking_labels": False})
        cleaner = build_text_cleaner(config, company_terms)
        text = clean(cleaner, snippets["zacks_rank_label"]).text
        assert "Zacks Rank" not in text
        assert "Cerence currently has a" in text


def test_rules_file_is_valid_and_names_are_unique(cleaning_config: CleaningConfig):
    names = [rule.name for rule in cleaning_config.noise_rules]
    assert len(names) == len(set(names))
    for rule in cleaning_config.noise_rules:
        assert rule.regex  # compiles
