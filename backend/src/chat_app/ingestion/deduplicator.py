"""Removes repeated articles so the same story is never counted twice."""

import itertools
import re
from collections.abc import Sequence

from chat_app.core.models import Article

_UPDATE_PREFIX = re.compile(r"^\s*update\s*:\s*", re.IGNORECASE)
_WORD = re.compile(r"\w+")


class Deduplicator:
    """Merges copies of the same story and keeps the most complete one."""

    def __init__(self, body_threshold: float, title_threshold: float, shingle_size: int) -> None:
        """Set how similar headlines and bodies must be (0-1) to count as the same story."""
        self._body_threshold = body_threshold
        self._title_threshold = title_threshold
        self._shingle_size = shingle_size

    def deduplicate(self, articles: Sequence[Article]) -> list[Article]:
        """Return one article per story, in original order, noting every copy merged in."""
        return self._merge_near_duplicates(self._merge_by_link(articles))

    def _merge_by_link(self, articles: Sequence[Article]) -> list[Article]:
        """Pass 1: merge entries with the exact same link (one story under several tickers)."""
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
        """Pass 2: merge re-publications of a story under a different link."""
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
        """True if two articles are the same story: near-identical headline and body."""
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
    """Pick which copy to keep: the longer one, then an "Update:" version, then the first."""
    first_words, second_words = len(first.text.split()), len(second.text.split())
    if first_words != second_words:
        return (first, second) if first_words > second_words else (second, first)
    if _is_update(second) and not _is_update(first):
        return second, first
    return first, second


def _is_update(article: Article) -> bool:
    """True if the headline starts with "Update:"."""
    return bool(_UPDATE_PREFIX.match(article.title))


def _merge(keep: Article, drop: Article) -> Article:
    """Keep one copy, adding the other's ticker keys and links so nothing is lost."""
    keys = list(dict.fromkeys([*keep.source_keys, *drop.source_keys]))
    links = [link for link in [*keep.merged_links, drop.link, *drop.merged_links]]
    merged_links = [link for link in dict.fromkeys(links) if link != keep.link]
    return keep.model_copy(update={"source_keys": keys, "merged_links": merged_links})


def _title_tokens(title: str) -> set[str]:
    """The headline's words, lowercased and without any "Update:" prefix."""
    return set(_WORD.findall(_UPDATE_PREFIX.sub("", title).lower()))


def _shingles(text: str, size: int) -> set[tuple[str, ...]]:
    """Every run of `size` consecutive words, used to compare two bodies."""
    words = _WORD.findall(text.lower())
    if len(words) < size:
        return {tuple(words)}
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def _jaccard[T](first: set[T], second: set[T]) -> float:
    """Overlap between two sets, from 0 (nothing shared) to 1 (identical)."""
    if not first and not second:
        return 1.0
    return len(first & second) / len(first | second)
