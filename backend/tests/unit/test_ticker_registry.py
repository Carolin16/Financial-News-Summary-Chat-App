import pytest

from chat_app.config.settings import Settings
from chat_app.core.ticker_registry import Company, TickerRegistry


@pytest.fixture
def registry() -> TickerRegistry:
    return TickerRegistry(
        {
            "GOOGL": Company(name="Alphabet (Google)", aliases=["Google", "Alphabet", "GOOG"]),
            "INTC": Company(name="Intel", aliases=["Intel"]),
            "AMD": Company(name="AMD", aliases=["Advanced Micro Devices"]),
        }
    )


@pytest.mark.parametrize("symbol", ["GOOGL", "goog", "Google", " alphabet "])
def test_canonicalizes_aliases(registry, symbol):
    assert registry.canonical(symbol) == "GOOGL"


def test_unknown_symbols_pass_through_uppercased(registry):
    assert registry.canonical("tsm") == "TSM"


def test_finds_names_case_insensitively_and_exchange_style_tickers(registry):
    text = "what's the news on google? Also NASDAQ:INTC and $AMD."
    assert set(registry.find_mentions(text)) == {"GOOGL", "INTC", "AMD"}


def test_does_not_match_inside_other_words(registry):
    assert registry.find_mentions("Artificial intelligence and damDAMAGE") == []


def test_bare_ticker_must_be_uppercase(registry):
    assert registry.find_mentions("the amd in lowercase prose") == []


def test_orders_by_mention_count(registry):
    assert registry.find_mentions("Intel. Google. Intel said Intel") == ["INTC", "GOOGL"]


def test_bundled_registry_covers_dataset_and_partial_coverage_companies():
    registry = TickerRegistry.from_json(Settings().tickers_path)
    for ticker in ["AAPL", "MSFT", "AMZN", "NFLX", "NVDA", "INTC", "IBM", "TSLA", "GOOGL"]:
        assert ticker in registry.tickers
