"""Sentence segmentation shared by cleaning, chunking, and answer verification."""

from functools import lru_cache

import pysbd


@lru_cache
def _segmenter() -> pysbd.Segmenter:
    # pysbd handles abbreviations like "U.S." and "Inc." and decimals like "$4.5" that a
    # naive split on ". " would break, which matters for keeping figures intact.
    return pysbd.Segmenter(language="en", clean=False)


@lru_cache
def _span_segmenter() -> pysbd.Segmenter:
    return pysbd.Segmenter(language="en", clean=False, char_span=True)


def split_sentences(text: str) -> list[str]:
    """Split text into trimmed, non-empty sentences."""
    return [s.strip() for s in _segmenter().segment(text) if s.strip()]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character offsets (start, end) of each sentence, covering the text in order."""
    return [(span.start, span.end) for span in _span_segmenter().segment(text)]
