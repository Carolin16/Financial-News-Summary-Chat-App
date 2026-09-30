"""Replaces punctuation (curly quotes, long dashes, special spaces) with plain ASCII.

Applied to both the articles and the LLM's answer, so numbers can be matched exactly
(e.g. "−5%" in an answer matches "-5%" in an article). Letters and digits are never
changed, so this can never create a number that wasn't there.
"""

PUNCTUATION_FOLDS: dict[str, str] = {
    # quotes and primes
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "′": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "″": '"',
    # hyphens, dashes, and the minus sign
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    # ellipsis
    "…": "...",
    # space variants (non-breaking, figure, narrow, thin, en/em spaces)
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
}

_TRANSLATION = str.maketrans(PUNCTUATION_FOLDS)


def fold_punctuation(text: str) -> str:
    """Replace typographic punctuation and space variants with ASCII equivalents."""
    return text.translate(_TRANSLATION)
