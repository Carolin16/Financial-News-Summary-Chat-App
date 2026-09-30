"""Flags thin articles (paywalled or teaser-only) so answers using them are low-confidence."""

from enum import StrEnum

from chat_app.ingestion.cleaning.models import CleanedArticle


class StubReason(StrEnum):
    """The reason an article was flagged as thin."""

    TRUNCATED = "truncated"  # cut off by a paywall or "Continue Reading"
    TOO_SHORT = "too_short"  # complete, but too short to be useful


class StubDetector:
    """Decides if an article has too little real text to rely on, using what cleaning found."""

    def __init__(self, min_words: int) -> None:
        """Set the minimum word count a full article must have."""
        self._min_words = min_words

    def reason(self, article: CleanedArticle) -> StubReason | None:
        """Return why the article is thin, or None if it is a full article."""
        if article.signals.truncation_marker_found:
            return StubReason.TRUNCATED
        if len(article.text.split()) < self._min_words:
            return StubReason.TOO_SHORT
        return None

    def is_stub(self, article: CleanedArticle) -> bool:
        """True if the article was cut off or is too short."""
        return self.reason(article) is not None
