"""Safety check before stripping a publisher footer to the end of the text.

Footers are recognised by their opening line, but a misfire would delete real article text
after it. The guard ignores sentences that match known footer boilerplate or merely echo the
article's title, and refuses the strip if any remaining *full sentence* carries article
content: a company name, a ticker, or a number. Headline fragments without terminal
punctuation (e.g. "Read More on AAPL:" lists) are not treated as content.
"""

import re
from collections.abc import Iterable

from chat_app.ingestion.cleaning.config import FooterSafetyConfig
from chat_app.ingestion.sentences import split_sentences

_EXCHANGE_TICKER = r"\((?:NASDAQ|NYSE|NYSEARCA|NYSEAMERICAN|OTC|TSX|LSE)\s*:\s*[A-Z.]+\)"
_NUMBER = re.compile(r"\d")
_SENTENCE_END = re.compile(r"[.!?][\"')]*$")


class FooterGuard:
    """Decides whether a footer tail may be removed."""

    def __init__(self, config: FooterSafetyConfig, company_terms: Iterable[str]) -> None:
        """`company_terms` are names and tickers whose presence signals article content."""
        self._config = config
        alternatives = [re.escape(t) for t in sorted(set(company_terms), key=len, reverse=True)]
        self._content = re.compile(
            rf"{_EXCHANGE_TICKER}|\b(?:{'|'.join(alternatives)})\b"
            if alternatives
            else _EXCHANGE_TICKER
        )

    def content_sentences(self, tail: str, title: str = "") -> list[str]:
        """Sentences in `tail` that are not boilerplate and carry content signals."""
        title_key = _key(title)
        return [
            s
            for s in split_sentences(tail)
            if not (title_key and _key(s) in title_key) and self._carries_content(s)
        ]

    def _carries_content(self, sentence: str) -> bool:
        if any(p.search(sentence) for p in self._config.boilerplate_regexes):
            return False
        is_full_sentence = (
            len(sentence.split()) >= self._config.min_sentence_words
            and _SENTENCE_END.search(sentence) is not None
        )
        has_signal = self._content.search(sentence) or _NUMBER.search(sentence)
        return is_full_sentence and has_signal is not None


def _key(text: str) -> str:
    """Comparison key for title echoes: lowercase letters and digits only."""
    return "".join(ch for ch in text.lower() if ch.isalnum())
