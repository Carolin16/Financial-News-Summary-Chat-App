"""Flags thin articles (paywalls, teasers) whose answers must be marked low-confidence."""

import re
from collections.abc import Iterable

from chat_app.ingestion import noise_patterns as noise


class StubDetector:
    """Decides whether an article has too little real content to be relied on."""

    def __init__(
        self,
        min_words: int,
        teaser_markers: Iterable[re.Pattern[str]] = noise.TEASER_MARKERS,
    ) -> None:
        """Create a detector; `min_words` is the cleaned-length floor for a full article."""
        self._min_words = min_words
        self._teaser_markers = tuple(teaser_markers)

    def is_stub(self, raw_text: str, cleaned_text: str) -> bool:
        """True if the raw text is a truncated teaser or the cleaned text is too short.

        Markers are checked on the raw text because cleaning removes them.
        """
        is_teaser = any(marker.search(raw_text) for marker in self._teaser_markers)
        return is_teaser or len(cleaned_text.split()) < self._min_words
