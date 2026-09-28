"""Flags thin articles (paywalls, teasers) whose answers must be marked low-confidence."""

from chat_app.ingestion.cleaning.models import CleanedArticle


class StubDetector:
    """Decides whether an article has too little real content to be relied on.

    Consumes the cleaner's signals rather than re-scanning raw text: the cleaner reports a
    truncation marker, this class makes the decision (SRP).
    """

    def __init__(self, min_words: int) -> None:
        """`min_words` is the cleaned-length floor for a full article."""
        self._min_words = min_words

    def is_stub(self, article: CleanedArticle) -> bool:
        """True if cleaning found a truncation marker or the cleaned text is too short."""
        too_short = len(article.text.split()) < self._min_words
        return article.signals.truncation_marker_found or too_short
