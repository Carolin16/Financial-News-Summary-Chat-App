"""List of known companies and the names they go by (e.g. "Google" or "GOOG" → GOOGL).

Used to spot which companies an article or a user's question is about. Companies are
listed in `config/tickers.json`, so adding one needs no code change.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Company(BaseModel):
    """A tracked company: display name and the surface forms that refer to it."""

    name: str
    aliases: list[str]
    products: list[str] = Field(
        default_factory=list, description="Product names that imply the company (exact case)."
    )
    indexed: bool = Field(
        default=False, description="True if the dataset has a feed key for this company."
    )


class MentionForm(StrEnum):
    """How a company was referred to."""

    NAME = "name"
    PRODUCT = "product"
    TICKER = "ticker"


@dataclass(frozen=True)
class Mention:
    """One reference to a tracked company in a text."""

    ticker: str
    start: int
    end: int
    form: MentionForm


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
        self._product_patterns = {
            ticker: re.compile(rf"\b(?:{'|'.join(map(re.escape, company.products))})\b")
            for ticker, company in self._companies.items()
            if company.products
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

    @property
    def indexed_tickers(self) -> list[str]:
        """Tickers the dataset has a feed key for (the others appear only as mentions)."""
        return [t for t, c in self._companies.items() if c.indexed]

    def mentions(self, text: str) -> list[Mention]:
        """Every reference to a tracked company in `text`, in reading order."""
        found: list[Mention] = []
        for ticker, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                form = (
                    MentionForm.TICKER if match.group().lstrip("$") == ticker else MentionForm.NAME
                )
                found.append(Mention(ticker, match.start(), match.end(), form))
        for ticker, pattern in self._product_patterns.items():
            found.extend(
                Mention(ticker, m.start(), m.end(), MentionForm.PRODUCT)
                for m in pattern.finditer(text)
            )
        return sorted(found, key=lambda m: m.start)

    def company_terms(self) -> list[str]:
        """Every ticker and alias, e.g. for detecting company mentions in text."""
        return [
            term
            for ticker, company in self._companies.items()
            for term in (ticker, *company.aliases)
        ]

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
