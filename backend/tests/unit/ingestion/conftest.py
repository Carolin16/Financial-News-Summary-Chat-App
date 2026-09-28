import json
from pathlib import Path

import pytest

from chat_app.config.settings import Settings
from chat_app.core.ticker_registry import TickerRegistry
from chat_app.ingestion.cleaning.config import CleaningConfig
from chat_app.ingestion.cleaning.factory import build_text_cleaner
from chat_app.ingestion.cleaning.models import CleaningContext
from chat_app.ingestion.cleaning.text_cleaner import TextCleaner

FIXTURES = Path(__file__).parents[2] / "fixtures"
DATASET_PATH = Path(__file__).parents[4] / "data" / "stock_news.json"


@pytest.fixture(scope="session")
def snippets() -> dict:
    """Real text snippets from stock_news.json (mojibake cases are synthetic: none exist)."""
    return json.loads((FIXTURES / "cleaning_snippets.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def cleaning_config() -> CleaningConfig:
    return CleaningConfig.from_toml()


@pytest.fixture(scope="session")
def company_terms() -> list[str]:
    return TickerRegistry.from_json(Settings().tickers_path).company_terms()


@pytest.fixture(scope="session")
def cleaner(cleaning_config, company_terms) -> TextCleaner:
    return build_text_cleaner(cleaning_config, company_terms)


@pytest.fixture
def context() -> CleaningContext:
    return CleaningContext(title="Unrelated headline")
