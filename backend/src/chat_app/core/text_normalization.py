"""Punctuation folding shared by ingestion and answer verification.

Articles are folded when cleaned, and LLM output is folded before its numbers are checked
against the sources, so both sides compare the same characters (e.g. a Unicode minus in an
answer still matches the ASCII hyphen in the article).

Only punctuation and space variants are folded. Letters (é, Ö, Korean) are never touched,
and nothing is mapped to a digit, so folding can never create a number. Full NFKC is avoided
for exactly that reason: it turns "²" into "2" and "½" into "1⁄2".
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
