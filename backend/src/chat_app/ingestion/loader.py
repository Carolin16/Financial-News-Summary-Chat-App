"""Reads the raw news dataset from its JSON file."""

import json
from pathlib import Path

from chat_app.core.models import RawArticle


class JsonArticleRepository:
    """`ArticleRepository` backed by a `{ticker: [article, ...]}` JSON file."""

    def __init__(self, path: Path) -> None:
        """Bind the repository to the dataset file at `path`."""
        self._path = path

    def load(self) -> list[RawArticle]:
        """Return every entry in file order; duplicates are kept for the deduplicator."""
        with self._path.open(encoding="utf-8") as handle:
            data: dict[str, list[dict[str, str]]] = json.load(handle)
        return [
            RawArticle.model_validate({**entry, "ticker": entry.get("ticker", key)})
            for key, entries in data.items()
            for entry in entries
        ]
