"""Checks how much news we have on each company, so answers can say when coverage is thin."""

from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel

from chat_app.core.models import Chunk


class CoverageLevel(StrEnum):
    """How much news we have on a company, from plenty down to nothing."""

    FULL = "full"  # enough full articles about it for a proper summary
    LIMITED = "limited"  # a few articles about it, or only paywalled ones
    MENTIONS_ONLY = "mentions_only"  # no articles about it, only passing mentions
    NONE = "none"  # not in the news at all


class CoverageReport(BaseModel):
    """A company's coverage level plus the article counts that led to it."""

    ticker: str
    level: CoverageLevel
    primary_articles: int
    primary_full_articles: int
    mention_articles: int


class _Counts(BaseModel):
    """IDs of the articles about a company, the full (non-paywalled) ones, and mentions."""

    primary: set[str] = set()
    primary_full: set[str] = set()
    mentioned: set[str] = set()


class CoverageIndex:
    """Counts each company's articles once at startup, then answers coverage questions fast."""

    def __init__(self, chunks: Iterable[Chunk], full_coverage_min_articles: int) -> None:
        """Tally articles per company. `full_coverage_min_articles` is the bar for FULL."""
        self._min_full = full_coverage_min_articles
        self._counts: dict[str, _Counts] = defaultdict(_Counts)
        self._articles: set[str] = set()
        # Count articles, not chunks: a long article split into 5 chunks still counts once.
        for chunk in chunks:
            self._articles.add(chunk.article_id)
            for ticker in chunk.metadata.primary_tickers:
                self._counts[ticker].primary.add(chunk.article_id)
                if not chunk.is_stub:
                    self._counts[ticker].primary_full.add(chunk.article_id)
            for ticker in chunk.metadata.mentioned_tickers:
                self._counts[ticker].mentioned.add(chunk.article_id)

    @property
    def total_articles(self) -> int:
        """How many different articles the index holds."""
        return len(self._articles)

    def assess(self, ticker: str) -> CoverageReport:
        """Return how well a company is covered, with the counts behind it."""
        counts = self._counts.get(ticker, _Counts())
        # Articles that mention the company without being about it.
        mentions_only = counts.mentioned - counts.primary
        if len(counts.primary_full) >= self._min_full:
            level = CoverageLevel.FULL
        elif counts.primary:
            level = CoverageLevel.LIMITED
        elif mentions_only:
            level = CoverageLevel.MENTIONS_ONLY
        else:
            level = CoverageLevel.NONE
        return CoverageReport(
            ticker=ticker,
            level=level,
            primary_articles=len(counts.primary),
            primary_full_articles=len(counts.primary_full),
            mention_articles=len(mentions_only),
        )

    def most_covered(self, limit: int, among: Iterable[str]) -> list[CoverageReport]:
        """The companies with the most articles about them, e.g. to list what the news covers."""
        candidates = [t for t in among if self._counts.get(t) and self._counts[t].primary]
        ranked = sorted(candidates, key=lambda t: len(self._counts[t].primary), reverse=True)
        return [self.assess(t) for t in ranked[:limit]]
