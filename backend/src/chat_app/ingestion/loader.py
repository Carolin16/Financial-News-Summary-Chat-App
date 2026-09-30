"""Reads stock_news.json and hands back every news entry exactly as it appears."""

import json
from pathlib import Path

from chat_app.core.models import RawArticle


class JsonArticleRepository:
    """Supplies raw articles from a JSON file grouped by ticker."""

    def __init__(self, path: Path) -> None:
        """Remember which dataset file to read."""
        self._path = path

    def load(self) -> list[RawArticle]:
        """Return all entries in file order, duplicates included (deduplication happens later)."""
        with self._path.open(encoding="utf-8") as handle:
            data: dict[str, list[dict[str, str]]] = json.load(handle)
        return [
            RawArticle.model_validate({**entry, "ticker": entry.get("ticker", key)})
            for key, entries in data.items()
            for entry in entries
        ]
