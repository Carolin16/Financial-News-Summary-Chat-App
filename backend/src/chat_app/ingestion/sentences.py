"""Sentence segmentation shared by the cleaner and the chunker."""

from functools import lru_cache

import pysbd


@lru_cache
def _segmenter() -> pysbd.Segmenter:
    # pysbd handles abbreviations like "U.S." and "Inc." and decimals like "$4.5" that a
    # naive split on ". " would break, which matters for keeping figures intact.
    return pysbd.Segmenter(language="en", clean=False)


def split_sentences(text: str) -> list[str]:
    """Split text into trimmed, non-empty sentences."""
    return [s.strip() for s in _segmenter().segment(text) if s.strip()]
