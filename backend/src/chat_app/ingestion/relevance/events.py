"""Rule-based event tagging and tone scoring."""

import re
from collections.abc import Sequence
from typing import Protocol

from chat_app.config.relevance import SentimentConfig
from chat_app.core.models import EventType, Sentiment

_WORD = re.compile(r"[A-Za-z']+")


class EventTagger(Protocol):
    """Finds the kinds of events an article reports."""

    def tag(self, title: str, focus_sentences: Sequence[str]) -> list[EventType]:
        """Event types found in the title or in sentences about the article's companies."""
        ...


class KeywordEventTagger:
    """Keyword patterns per event type, from configuration."""

    def __init__(self, patterns: dict[str, list[str]]) -> None:
        """`patterns` maps an EventType value to its regexes."""
        self._patterns = {
            EventType(name): [re.compile(p) for p in regexes] for name, regexes in patterns.items()
        }

    def tag(self, title: str, focus_sentences: Sequence[str]) -> list[EventType]:
        """Event types in enum order, so output is stable."""
        texts = [title, *focus_sentences]
        found = {
            event
            for event, regexes in self._patterns.items()
            if any(r.search(t) for r in regexes for t in texts)
        }
        return [event for event in EventType if event in found]


class SentimentScorer(Protocol):
    """Estimates an article's tone."""

    def score(self, focus_sentences: Sequence[str]) -> Sentiment:
        """Tone of the given sentences."""
        ...


class LexiconSentimentScorer:
    """Counts finance tone words, flipping those shortly after a negator.

    Article-level and lexical: good enough to describe tone, not to filter on. It cannot
    tell whose news is good in a comparison (e.g. AMD gains at Intel's expense).
    """

    def __init__(self, config: SentimentConfig) -> None:
        """Word lists and thresholds come from the relevance rules file."""
        self._config = config
        self._positive = {_stem(w) for w in config.positive}
        self._negative = {_stem(w) for w in config.negative}
        self._negators = {w.lower() for w in config.negators}

    def score(self, focus_sentences: Sequence[str]) -> Sentiment:
        """Positive, negative, mixed, or neutral."""
        positive = negative = 0
        for sentence in focus_sentences:
            words = [w.lower() for w in _WORD.findall(sentence)]
            for index, word in enumerate(words):
                stem = _stem(word)
                polarity = (stem in self._positive) - (stem in self._negative)
                if not polarity:
                    continue
                window = words[max(0, index - self._config.negation_window) : index]
                if any(w in self._negators for w in window):
                    polarity = -polarity
                positive += polarity > 0
                negative += polarity < 0
        return self._label(positive, negative)

    def _label(self, positive: int, negative: int) -> Sentiment:
        config = self._config
        hits = positive + negative
        if hits < config.min_hits:
            return Sentiment.NEUTRAL
        polarity = (positive - negative) / hits
        if polarity >= config.polarity_threshold:
            return Sentiment.POSITIVE
        if polarity <= -config.polarity_threshold:
            return Sentiment.NEGATIVE
        if min(positive, negative) >= config.mixed_min_each:
            return Sentiment.MIXED
        return Sentiment.NEUTRAL


_SUFFIXES = ("ing", "ies", "ied", "es", "ed", "s")


def _stem(word: str) -> str:
    """Crude inflection folding so "surges", "surged", "surging" all match "surge"."""
    word = word.lower().rstrip("'")
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            base = word[: -len(suffix)]
            return base + "y" if suffix in ("ies", "ied") else base.rstrip("e")
    return word.rstrip("e")
