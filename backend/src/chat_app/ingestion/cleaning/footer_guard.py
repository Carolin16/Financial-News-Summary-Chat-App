"""Makes sure cutting a publisher footer never deletes real article text after it."""

import re
from collections.abc import Iterable

from chat_app.ingestion.cleaning.config import FooterSafetyConfig
from chat_app.ingestion.sentences import split_sentences

_EXCHANGE_TICKER = r"\((?:NASDAQ|NYSE|NYSEARCA|NYSEAMERICAN|OTC|TSX|LSE)\s*:\s*[A-Z.]+\)"
_NUMBER = re.compile(r"\d")
_SENTENCE_END = re.compile(r"[.!?][\"')]*$")


class FooterGuard:
    """Says whether a footer is safe to cut or hides real content that must stay."""

    def __init__(self, config: FooterSafetyConfig, company_terms: Iterable[str]) -> None:
        """Take the company names and tickers whose presence means "this is real content"."""
        self._config = config
        alternatives = [re.escape(t) for t in sorted(set(company_terms), key=len, reverse=True)]
        self._content = re.compile(
            rf"{_EXCHANGE_TICKER}|\b(?:{'|'.join(alternatives)})\b"
            if alternatives
            else _EXCHANGE_TICKER
        )

    def content_sentences(self, tail: str, title: str = "") -> list[str]:
        """Return the sentences after the footer marker that look like real article content."""
        title_key = _key(title)
        return [
            s
            for s in split_sentences(tail)
            if not (title_key and _key(s) in title_key) and self._carries_content(s)
        ]

    def _carries_content(self, sentence: str) -> bool:
        """True for a full, non-boilerplate sentence that names a company or has a number."""
        if any(p.search(sentence) for p in self._config.boilerplate_regexes):
            return False
        is_full_sentence = (
            len(sentence.split()) >= self._config.min_sentence_words
            and _SENTENCE_END.search(sentence) is not None
        )
        has_signal = self._content.search(sentence) or _NUMBER.search(sentence)
        return is_full_sentence and has_signal is not None


def _key(text: str) -> str:
    """Simplify text to lowercase letters and digits, to spot sentences repeating the title."""
    return "".join(ch for ch in text.lower() if ch.isalnum())
