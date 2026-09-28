"""How well the dataset covers each company, derived from content-based article metadata.

Drives honest framing: a full summary for well-covered companies, an explicit "coverage is
limited" for thin ones (IBM, Google, Tesla), and "no articles" only when truly none exist.
"""

from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel

from chat_app.core.models import Chunk


class CoverageLevel(StrEnum):
    """Coverage tiers, from dedicated reporting down to nothing at all."""

    FULL = "full"
    LIMITED = "limited"
    MENTIONS_ONLY = "mentions_only"
    NONE = "none"


class CoverageReport(BaseModel):
    """Article counts behind a company's coverage level."""

    ticker: str
    level: CoverageLevel
    primary_articles: int
    primary_full_articles: int
    mention_articles: int


class _Counts(BaseModel):
    primary: set[str] = set()
    primary_full: set[str] = set()
    mentioned: set[str] = set()


class CoverageIndex:
    """Per-ticker article counts, built once from the indexed chunks."""

    def __init__(self, chunks: Iterable[Chunk], full_coverage_min_articles: int) -> None:
        """A company has FULL coverage with at least this many non-stub primary articles."""
        self._min_full = full_coverage_min_articles
        self._counts: dict[str, _Counts] = defaultdict(_Counts)
        self._articles: set[str] = set()
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
        """Number of unique articles in the index."""
        return len(self._articles)

    def assess(self, ticker: str) -> CoverageReport:
        """Classify how well `ticker` is covered."""
        counts = self._counts.get(ticker, _Counts())
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
        """Companies from `among` with the most primary articles, for coverage summaries."""
        candidates = [t for t in among if self._counts.get(t) and self._counts[t].primary]
        ranked = sorted(candidates, key=lambda t: len(self._counts[t].primary), reverse=True)
        return [self.assess(t) for t in ranked[:limit]]
