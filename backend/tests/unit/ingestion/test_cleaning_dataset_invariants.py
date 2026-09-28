"""Hard invariants and per-noise-type regressions over the whole dataset."""

import re
from collections import Counter
from pathlib import Path

import pytest

from chat_app.core.text_normalization import fold_punctuation
from chat_app.ingestion.cleaning.config import CleaningConfig
from chat_app.ingestion.cleaning.factory import build_text_cleaner
from chat_app.ingestion.cleaning.models import CleanedArticle
from chat_app.ingestion.loader import JsonArticleRepository

DATASET_PATH = Path(__file__).parents[4] / "data" / "stock_news.json"

pytestmark = pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset not available")

_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_EXCHANGE_TICKER = re.compile(r"\((?:NASDAQ|NYSE|NYSEARCA|OTC|TSX|LSE)\s*:\s*[A-Z.]+\)")
# Tickers that only appear inside Insider Monkey promo sentences, which are removed whole.
_TICKERS_IN_REMOVED_PROMO = {"Here's Why Microsoft (MSFT) Stock Returned 13% in Q4": 2}


@pytest.fixture(scope="module")
def articles():
    return list({r.link: r for r in JsonArticleRepository(DATASET_PATH).load()}.values())


@pytest.fixture(scope="module")
def cleaned(articles, cleaner) -> list[tuple[str, CleanedArticle]]:
    return [(raw.full_text, cleaner.clean(raw)) for raw in articles]


def _numbers(text: str) -> set[str]:
    return {m.group(0) for m in _NUMBER.finditer(text)}


def test_cleaning_is_idempotent(cleaned, cleaner):
    for _, article in cleaned:
        assert cleaner.clean_text(article.text, article.title) == article.text, article.title


def test_cleaning_never_creates_numbers(cleaned):
    for raw_text, article in cleaned:
        created = _numbers(article.text) - _numbers(fold_punctuation(raw_text))
        assert not created, (article.title, created)


def test_exchange_tickers_preserved_outside_removed_promos(cleaned):
    for raw_text, article in cleaned:
        before = Counter(_EXCHANGE_TICKER.findall(fold_punctuation(raw_text)))
        after = Counter(_EXCHANGE_TICKER.findall(article.text))
        lost = sum((before - after).values())
        expected = _TICKERS_IN_REMOVED_PROMO.get(article.title.replace("’", "'"), 0)
        assert lost == expected, (article.title, before - after)


def test_non_noise_steps_preserve_every_ticker(articles, cleaning_config):
    config = cleaning_config.model_copy(
        update={"step_order": [s for s in cleaning_config.step_order if s != "noise_blocks"]}
    )
    cleaner = build_text_cleaner(config, [])
    for raw in articles:
        before = Counter(_EXCHANGE_TICKER.findall(raw.full_text))
        assert Counter(_EXCHANGE_TICKER.findall(cleaner.clean(raw).text)) == before


@pytest.mark.parametrize("symbol", ["€", "¥"])
def test_non_dollar_currency_signs_survive_exactly(cleaned, symbol):
    # Euro and yen figures only occur in article content, so every one must survive.
    for raw_text, article in cleaned:
        assert article.text.count(symbol) == raw_text.count(symbol), article.title


@pytest.mark.parametrize(
    "noise",
    [
        "View Comments",
        "Continue Reading",
        "Story Continues",
        "PREMIUM Upgrade",
        "READ NEXT",
        "Read Next:",
        "READ ALSO",
        "Don't Miss",
        "Trending:",
        "See Also:",
        "Most Read from Bloomberg",
        "Related Videos",
        "Yahoo Finance Video",
        "newsletter",
        "275%",
        "Stock Advisor",
        "Getty Images",
        "Image Source:",
        "https://",
        "www.",
        "@yahoofinance.com",
        "800-767-3771",
        "Disclosure:",
        "All rights reserved",
        "originally published",
        "originally appeared",
        "Click here",
        "GuruFocus has detected",
        "★",
        "\U0001f4b0",
        " ",
        "’",
    ],
)
def test_noise_type_is_gone_everywhere(cleaned, noise):
    offenders = [a.title for _, a in cleaned if noise in a.text]
    assert not offenders, offenders


def test_ranking_labels_kept_by_default(cleaned):
    assert sum("Zacks Rank #" in a.text for _, a in cleaned) >= 5


def test_truncation_signals_match_teasers_and_paywalls(cleaned):
    flagged = [a for _, a in cleaned if a.signals.truncation_marker_found]
    assert len(flagged) == 24  # 11 "Continue Reading" teasers + 13 paywalled stubs (pre-dedupe)


def test_no_footer_needed_the_safety_guard(cleaned):
    assert not [a.title for _, a in cleaned if a.signals.footers_kept]


def test_rules_file_loads_from_settings_path(cleaning_config):
    from chat_app.config.settings import Settings

    assert CleaningConfig.from_toml(Settings().cleaning_rules_path) == cleaning_config
