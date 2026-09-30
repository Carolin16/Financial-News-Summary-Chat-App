"""Entity matching: registry names/products/tickers and untracked companies in formal forms."""

import pytest

from chat_app.config.relevance import RelevanceConfig
from chat_app.config.settings import Settings
from chat_app.core.ticker_registry import MentionForm, TickerRegistry
from chat_app.ingestion.relevance.entities import RegistryEntityMatcher


@pytest.fixture(scope="module")
def matcher() -> RegistryEntityMatcher:
    registry = TickerRegistry.from_json(Settings().tickers_path)
    return RegistryEntityMatcher(registry, RelevanceConfig.from_toml().entities)


def tickers(mentions) -> list[str]:
    return [m.ticker for m in mentions]


class TestRegistry:
    @pytest.mark.parametrize(
        ("text", "ticker", "form"),
        [
            ("Intel shares rose.", "INTC", MentionForm.NAME),
            ("Buy INTC today.", "INTC", MentionForm.TICKER),
            ("Sales of the iPhone grew.", "AAPL", MentionForm.PRODUCT),
            ("AWS revenue rose.", "AMZN", MentionForm.PRODUCT),
            ("The news on Tesla.", "TSLA", MentionForm.NAME),  # non-indexed companies count
            ("Google and Alphabet", "GOOGL", MentionForm.NAME),
        ],
    )
    def test_registry_forms(self, matcher, text, ticker, form):
        [mention] = matcher.find("", text).body[:1]
        assert (mention.ticker, mention.form) == (ticker, form)

    def test_products_are_case_sensitive(self, matcher):
        assert matcher.find("", "Open the windows and the gemini constellation.").body == []

    def test_exchange_tag_is_marked(self, matcher):
        body = matcher.find("", "Apple Inc. (NASDAQ:AAPL) rose.").body
        assert tickers(body) == ["AAPL", "AAPL"]
        assert [m.exchange_tagged for m in body] == [False, True]


class TestUntrackedCompanies:
    @pytest.mark.parametrize(
        "text",
        [
            "Retailer Walmart (NYSE:WMT) rose.",
            "Shares of Cerence Inc. (CRNC) jumped.",
            "Cerence CRNC shares have returned 75%. CRNC stock rose.",  # Zacks style
            "Cal-Maine Foods, Inc. CALM as the Bull of the Day.",  # legal suffix, single use
        ],
    )
    def test_formal_forms_are_recognised(self, matcher, text):
        assert matcher.find("", text).body

    @pytest.mark.parametrize(
        "text",
        [
            "Natural Language Processing (NLP) tools.",  # acronym of the words before it
            "deep ultraviolet (DUV) machines",  # no capitalised name before it
            "The Chief Executive Officer (CEO) said.",  # configured not-a-ticker
            "Nvidia GPU demand.",  # bare symbol that never recurs
        ],
    )
    def test_acronyms_and_noise_are_not_companies(self, matcher, text):
        assert "INTC" not in tickers(matcher.find("", text).body)
        untracked = [m for m in matcher.find("", text).body if m.ticker not in {"NVDA"}]
        assert untracked == []

    def test_learned_name_is_found_in_the_title(self, matcher):
        found = matcher.find(
            "CyberArk Rises 8% Since Q4 Earnings Beat",
            "CyberArk Software Ltd. CYBR shares have gained. Shares of CYBR have rallied.",
        )
        assert tickers(found.title) == ["CYBR"]
        assert found.learned_names["CyberArk"] == "CYBR"

    def test_sentence_initial_filler_is_not_part_of_the_name(self, matcher):
        found = matcher.find(
            "Surprise: McDonald's Has Higher Profit Margins Than Tesla",
            "Big Tech has high margins. But McDonald's (NYSE: MCD), the world's largest chain...",
        )
        assert tickers(found.title) == ["MCD", "TSLA"]

    def test_learned_names_match_case_insensitively(self, matcher):
        found = matcher.find(
            "Enpro Earnings: What To Look For From NPO",
            "Industrial provider EnPro Industries (NYSE:NPO) will report.",
        )
        assert tickers(found.title)[0] == "NPO"
