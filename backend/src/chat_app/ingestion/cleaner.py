"""Text normalisation: repair encoding damage and strip publisher boilerplate and promos."""

import re
import unicodedata
from collections.abc import Iterable

from chat_app.ingestion import noise_patterns as noise
from chat_app.ingestion.sentences import split_sentences

_WHITESPACE = re.compile(r"\s+")


class TextCleaner:
    """Turns raw scraped article text into clean prose suitable for indexing."""

    def __init__(
        self,
        inline_noise: Iterable[re.Pattern[str]] = noise.INLINE_NOISE,
        promo_prefixes: Iterable[re.Pattern[str]] = noise.PROMO_SENTENCE_PREFIXES,
        promo_content: Iterable[re.Pattern[str]] = noise.PROMO_SENTENCE_CONTENT,
    ) -> None:
        """Create a cleaner; pattern sets are injectable so publishers can be added freely."""
        self._inline_noise = tuple(inline_noise)
        self._promo_prefixes = tuple(promo_prefixes)
        self._promo_content = tuple(promo_content)

    def clean(self, text: str) -> str:
        """Return `text` with encoding repaired, noise removed, and whitespace collapsed."""
        text = normalize_characters(text)
        for pattern in self._inline_noise:
            text = pattern.sub(" ", text)
        text = _collapse_whitespace(text)
        kept = [s for s in split_sentences(text) if not self._is_promotional(s)]
        return _collapse_whitespace(" ".join(kept))

    def _is_promotional(self, sentence: str) -> bool:
        return any(p.search(sentence) for p in self._promo_prefixes) or any(
            p.search(sentence) for p in self._promo_content
        )


def normalize_characters(text: str) -> str:
    """Repair mojibake, apply NFKC (e.g. non-breaking spaces), and fold typographic marks."""
    text = repair_mojibake(text)
    text = unicodedata.normalize("NFKC", text)
    for source, target in noise.TYPOGRAPHIC_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text


def repair_mojibake(text: str) -> str:
    """Undo UTF-8 text that was decoded as cp1252 (e.g. `â€™` -> `'`).

    A full round trip is exact when the whole string was mis-decoded; mixed text cannot
    round-trip, so fall back to replacing the known sequences.
    """
    if not any(marker in text for marker in noise.MOJIBAKE_MARKERS):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        for source, target in noise.MOJIBAKE_REPLACEMENTS.items():
            text = text.replace(source, target)
        return text


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()
