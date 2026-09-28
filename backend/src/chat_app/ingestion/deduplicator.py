"""Collapses repeated articles so coverage is not overstated.

Two passes over *cleaned* articles:

1. Exact duplicates share a link: the same story filed under several ticker keys (13 links
   carry 20 extra entries in the dataset). Copies can differ in scrape-time details (video
   timestamps, a live price quote), so the most complete copy is kept.
2. Near-duplicates are re-publications of the same headline under a new link (e.g. an
   "Update:" version). They must match on headline *and* body; stub bodies are truncated
   mid-sentence, so for stubs the headline alone decides.

The kept article records every ticker key it was filed under (`source_keys`) and the links
it absorbed (`merged_links`). Both are provenance only and never used for relevance.
"""

import itertools
import re
from collections.abc import Sequence

from chat_app.core.models import Article

_UPDATE_PREFIX = re.compile(r"^\s*update\s*:\s*", re.IGNORECASE)
_WORD = re.compile(r"\w+")


class Deduplicator:
    """Merges exact and near-duplicate articles, keeping the most complete version."""

    def __init__(self, body_threshold: float, title_threshold: float, shingle_size: int) -> None:
        """Configure similarity thresholds (Jaccard, 0-1) and body shingle length in words."""
        self._body_threshold = body_threshold
        self._title_threshold = title_threshold
        self._shingle_size = shingle_size

    def deduplicate(self, articles: Sequence[Article]) -> list[Article]:
        """Return unique articles in first-seen order with provenance merged."""
        return self._merge_near_duplicates(self._merge_by_link(articles))

    def _merge_by_link(self, articles: Sequence[Article]) -> list[Article]:
        by_link: dict[str, Article] = {}
        for article in articles:
            existing = by_link.get(article.link)
            if existing is None:
                by_link[article.link] = article
            else:
                keep, drop = preferred_version(existing, article)
                by_link[article.link] = _merge(keep=keep, drop=drop)
        return list(by_link.values())

    def _merge_near_duplicates(self, articles: list[Article]) -> list[Article]:
        survivors = list(articles)
        for first, second in itertools.combinations(articles, 2):
            current = {a.link: a for a in survivors}
            if first.link not in current or second.link not in current:
                continue
            first, second = current[first.link], current[second.link]
            if not self.are_near_duplicates(first, second):
                continue
            keep, drop = preferred_version(first, second)
            survivors[survivors.index(keep)] = _merge(keep=keep, drop=drop)
            survivors.remove(drop)
        return survivors

    def are_near_duplicates(self, first: Article, second: Article) -> bool:
        """Same headline (ignoring an "Update:" prefix) and matching bodies.

        The headline check is required because templated series (analyst-note teasers,
        "intrinsic value" reports) share most of their body text yet are different news.
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


def preferred_version(first: Article, second: Article) -> tuple[Article, Article]:
    """Return (keep, drop), preferring the version with more content.

    On a tie an "Update:" version wins, then the one seen first. More content wins over
    recency because only the text we hold can ground an answer.
    """
    first_words, second_words = len(first.text.split()), len(second.text.split())
    if first_words != second_words:
        return (first, second) if first_words > second_words else (second, first)
    if _is_update(second) and not _is_update(first):
        return second, first
    return first, second


def _is_update(article: Article) -> bool:
    return bool(_UPDATE_PREFIX.match(article.title))


def _merge(keep: Article, drop: Article) -> Article:
    keys = list(dict.fromkeys([*keep.source_keys, *drop.source_keys]))
    links = [link for link in [*keep.merged_links, drop.link, *drop.merged_links]]
    merged_links = [link for link in dict.fromkeys(links) if link != keep.link]
    return keep.model_copy(update={"source_keys": keys, "merged_links": merged_links})


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
