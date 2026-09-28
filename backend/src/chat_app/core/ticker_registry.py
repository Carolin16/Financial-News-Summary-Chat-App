"""Known companies and how they are referred to in text.

Loaded from a JSON data file so tracking a new company is a data change, not a code change.
Used offline (canonicalising enrichment output) and at query time (spotting companies in
the user's question).
"""

import json
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel


class Company(BaseModel):
    """A tracked company: display name and the surface forms that refer to it."""

    name: str
    aliases: list[str]


class TickerRegistry:
    """Maps tickers, aliases, and exchange-style references (NASDAQ:AAPL) to tickers."""

    def __init__(self, companies: dict[str, Company]) -> None:
        """Build lookup patterns for every ticker and alias."""
        self._companies = {ticker.upper(): company for ticker, company in companies.items()}
        self._alias_to_ticker = {
            alias.upper(): ticker
            for ticker, company in self._companies.items()
            for alias in company.aliases
        }
        self._patterns = {
            ticker: _mention_pattern(ticker, company) for ticker, company in self._companies.items()
        }

    @classmethod
    def from_json(cls, path: Path) -> "TickerRegistry":
        """Load a registry from `{ticker: {name, aliases}}` JSON."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls({ticker: Company.model_validate(entry) for ticker, entry in raw.items()})

    @property
    def tickers(self) -> list[str]:
        """All tracked tickers."""
        return list(self._companies)

    def name_for(self, ticker: str) -> str:
        """Display name for a ticker, or the ticker itself if untracked."""
        company = self._companies.get(ticker.upper())
        return company.name if company else ticker.upper()

    def canonical(self, symbol: str) -> str:
        """Normalise a ticker or alias (e.g. GOOG, Google) to its registry ticker."""
        key = symbol.strip().upper()
        if key in self._companies:
            return key
        return self._alias_to_ticker.get(key, key)

    def mention_counts(self, text: str) -> Counter[str]:
        """How many times each tracked company is referenced in `text`."""
        counts: Counter[str] = Counter()
        for ticker, pattern in self._patterns.items():
            hits = len(pattern.findall(text))
            if hits:
                counts[ticker] = hits
        return counts

    def find_mentions(self, text: str) -> list[str]:
        """Tracked tickers referenced in `text`, most-mentioned first."""
        return [ticker for ticker, _ in self.mention_counts(text).most_common()]


def _mention_pattern(ticker: str, company: Company) -> re.Pattern[str]:
    # Aliases match case-insensitively ("nvidia"), but bare tickers must be upper case so
    # short symbols such as "AMD" or "IBM" in prose don't collide with ordinary words.
    alias_alternatives = "|".join(re.escape(alias) for alias in company.aliases)
    return re.compile(
        rf"(?:(?i:\b(?:{alias_alternatives})\b)|(?<![A-Za-z])\$?{re.escape(ticker)}\b)"
    )
