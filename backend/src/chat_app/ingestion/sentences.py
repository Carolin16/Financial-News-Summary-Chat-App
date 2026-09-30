"""Sentence segmentation shared by cleaning, chunking, relevance, and answer verification."""

import threading

import pysbd

# pysbd segmenters keep per-call state and are not thread-safe; sharing one across the
# enrichment thread pool returned wrong spans intermittently. One instance per thread.
_local = threading.local()


def _segmenter() -> pysbd.Segmenter:
    # pysbd handles abbreviations like "U.S." and "Inc." and decimals like "$4.5" that a
    # naive split on ". " would break, which matters for keeping figures intact.
    if not hasattr(_local, "segmenter"):
        _local.segmenter = pysbd.Segmenter(language="en", clean=False)
    segmenter: pysbd.Segmenter = _local.segmenter
    return segmenter


def _span_segmenter() -> pysbd.Segmenter:
    if not hasattr(_local, "span_segmenter"):
        _local.span_segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    segmenter: pysbd.Segmenter = _local.span_segmenter
    return segmenter


def split_sentences(text: str) -> list[str]:
    """Split text into trimmed, non-empty sentences."""
    return [s.strip() for s in _segmenter().segment(text) if s.strip()]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character offsets (start, end) of each sentence, covering the text in order."""
    return [(span.start, span.end) for span in _span_segmenter().segment(text)]
