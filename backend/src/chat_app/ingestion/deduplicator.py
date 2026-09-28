"""Collapses repeated articles so coverage is not overstated.

Two passes: exact duplicates share a link (the same story filed under several tickers);
near-duplicates are re-published "Update:" versions of the same headline.
"""

import itertools
import re
from collections.abc import Sequence

from chat_app.core.models import Article

_UPDATE_PREFIX = re.compile(r"^\s*update\s*:\s*", re.IGNORECASE)
_WORD = re.compile(r"\w+")


class Deduplicator:
    """Merges exact and near-duplicate articles, keeping the best version of each."""

    def __init__(self, body_threshold: float, title_threshold: float, shingle_size: int) -> None:
        """Configure similarity thresholds (Jaccard, 0-1) and body shingle length in words."""
        self._body_threshold = body_threshold
        self._title_threshold = title_threshold
        self._shingle_size = shingle_size

    def deduplicate(self, articles: Sequence[Article]) -> list[Article]:
        """Return unique articles in first-seen order with `source_tickers` merged."""
        return self._merge_near_duplicates(self._merge_by_link(articles))

    def _merge_by_link(self, articles: Sequence[Article]) -> list[Article]:
        by_link: dict[str, Article] = {}
        for article in articles:
            existing = by_link.get(article.link)
            by_link[article.link] = (
                article if existing is None else _merge(keep=existing, drop=article)
            )
        return list(by_link.values())

    def _merge_near_duplicates(self, articles: list[Article]) -> list[Article]:
        survivors = list(articles)
        for first, second in itertools.combinations(articles, 2):
            if first not in survivors or second not in survivors:
                continue
            if not self.are_near_duplicates(first, second):
                continue
            keep, drop = _preferred_version(first, second)
            survivors[survivors.index(keep)] = _merge(keep=keep, drop=drop)
            survivors.remove(drop)
        return survivors

    def are_near_duplicates(self, first: Article, second: Article) -> bool:
        """Same headline (ignoring an "Update:" prefix) and matching bodies.

        The headline check is required because templated series (analyst-note teasers,
        "intrinsic value" reports) share most of their body text yet are different news.
        Stub bodies are truncated mid-sentence, so for stubs the headline alone decides.
        """
        title_similarity = _jaccard(_title_tokens(first.title), _title_tokens(second.title))
        if title_similarity < self._title_threshold:
            return False
        if first.is_stub or second.is_stub:
            return True
        body_similarity = _jaccard(
            _shingles(first.text, self._shingle_size), _shingles(second.text, self._shingle_size)
        )
        return body_similarity >= self._body_threshold


def _preferred_version(first: Article, second: Article) -> tuple[Article, Article]:
    """Prefer an explicitly updated version, then the longer text."""
    first_updated = bool(_UPDATE_PREFIX.match(first.title))
    second_updated = bool(_UPDATE_PREFIX.match(second.title))
    if first_updated != second_updated:
        return (first, second) if first_updated else (second, first)
    return (first, second) if len(first.text) >= len(second.text) else (second, first)


def _merge(keep: Article, drop: Article) -> Article:
    tickers = list(dict.fromkeys([*keep.source_tickers, *drop.source_tickers]))
    return keep.model_copy(update={"source_tickers": tickers})


def _title_tokens(title: str) -> set[str]:
    return set(_WORD.findall(_UPDATE_PREFIX.sub("", title).lower()))


def _shingles(text: str, size: int) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) < size:
        return {tuple(words)}
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def _jaccard[T](first: set[T], second: set[T]) -> float:
    if not first and not second:
        return 1.0
    return len(first & second) / len(first | second)
