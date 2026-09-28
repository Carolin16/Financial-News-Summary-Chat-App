"""Flags thin articles (paywalls, teasers) whose answers must be marked low-confidence.

A stub is low-confidence, not irrelevant: "DBS Bank Adjusts NVIDIA Price Target to $160
From $175" is a stub but clearly about Nvidia. Stub status is therefore a separate field
that never feeds into relevance.
"""

from enum import StrEnum

from chat_app.ingestion.cleaning.models import CleanedArticle


class StubReason(StrEnum):
    """Why an article counts as thin content."""

    TRUNCATED = "truncated"  # the cleaner found a paywall or "Continue Reading" marker
    TOO_SHORT = "too_short"  # complete, but below the configured length


class StubDetector:
    """Decides whether an article has too little real content to be relied on.

    Consumes the cleaner's signals rather than re-scanning raw text: the cleaner reports a
    truncation marker, this class makes the decision (SRP).
    """

    def __init__(self, min_words: int) -> None:
        """`min_words` is the cleaned-length floor for a full article."""
        self._min_words = min_words

    def reason(self, article: CleanedArticle) -> StubReason | None:
        """Why `article` is a stub, or None if it is a full article."""
        if article.signals.truncation_marker_found:
            return StubReason.TRUNCATED
        if len(article.text.split()) < self._min_words:
            return StubReason.TOO_SHORT
        return None

    def is_stub(self, article: CleanedArticle) -> bool:
        """True if cleaning found a truncation marker or the cleaned text is too short."""
        return self.reason(article) is not None
