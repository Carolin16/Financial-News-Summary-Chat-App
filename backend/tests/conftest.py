from pathlib import Path

import pytest

from chat_app.core.models import Article, ArticleMetadata

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
DATASET_PATH = TESTS_DIR.parents[1] / "data" / "stock_news.json"


@pytest.fixture
def make_article():
    """Factory for Article objects with sensible defaults."""

    def _make(
        title: str = "Intel shares jump",
        link: str = "https://example.com/a",
        text: str = "Intel shares rose 5% after a report.",
        source_keys: list[str] | None = None,
        is_stub: bool = False,
        metadata: ArticleMetadata | None = None,
        article_id: str | None = None,
    ) -> Article:
        return Article(
            article_id=article_id or link.rsplit("/", 1)[-1],
            title=title,
            link=link,
            text=text,
            source_keys=source_keys or ["INTC"],
            is_stub=is_stub,
            metadata=metadata or ArticleMetadata(),
        )

    return _make
