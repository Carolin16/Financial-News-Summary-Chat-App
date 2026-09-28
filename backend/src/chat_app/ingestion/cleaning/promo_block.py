"""Finds where a cross-promotion block ends.

A marker such as "READ ALSO:" is followed by other articles' headlines, which often run
straight into the real text with no punctuation ("…Analyst Ratings The Chinese startup may
be…"). Ending the block at the next full stop would delete genuine content, so the end is
the earliest of:

* a known next marker (e.g. "Disclosure:"),
* the configured word limit (a safety cap),
* in *headline mode* (block starts in Title Case): the first lowercase content word, which
  marks the switch to prose; the prose starts at the capitalised word before it (or at a
  sentence opener such as "The" just before that),
* in *sentence mode* (block starts in sentence case): the end of the sentence, or an
  unpunctuated sentence boundary ("…tied to diversity It's part of…").
"""

import re
from dataclasses import dataclass

from chat_app.ingestion.cleaning.config import PromoBlockConfig
from chat_app.ingestion.sentences import sentence_spans

_TOKEN = re.compile(r"\S+")
_EDGE_PUNCTUATION = "\"'([{)]}.,;:!?-"
_NON_ASCII_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    raw: str

    @property
    def core(self) -> str:
        return self.raw.strip(_EDGE_PUNCTUATION)

    @property
    def ends_clause(self) -> bool:
        return self.raw.rstrip("\"')").endswith((".", "!", "?", ":", ";", ","))


class PromoBlockScanner:
    """Computes the end offset of a promo block that starts at a marker."""

    def __init__(self, config: PromoBlockConfig) -> None:
        """Configure word lists and limits (all from the rules file)."""
        self._config = config
        self._connectors = {c.lower() for c in config.connectors}
        self._openers = set(config.sentence_openers)
        self._end_markers = [re.compile(re.escape(m)) for m in config.end_markers]

    def block_end(self, text: str, marker_end: int, allow_foreign: bool = False) -> int:
        """Offset where the promo block that begins before `marker_end` stops."""
        tokens = [_Token(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(text, marker_end)][
            : self._config.max_words
        ]
        if not tokens:
            return len(text)
        candidates = [tokens[-1].end, self._next_end_marker(text, marker_end, tokens[-1].end)]
        if self._is_headline(tokens):
            candidates.append(self._headline_end(tokens, allow_foreign))
        else:
            candidates.append(self._sentence_end(text, tokens))
        return min(c for c in candidates if c is not None)

    def _next_end_marker(self, text: str, start: int, limit: int) -> int | None:
        found = [m.start() for p in self._end_markers if (m := p.search(text, start, limit))]
        return min(found, default=None)

    def _is_headline(self, tokens: list[_Token]) -> bool:
        content = [t.core for t in tokens if t.core and t.core.lower() not in self._connectors]
        probe = content[: self._config.title_case_probe_words]
        return bool(probe) and all(w[0].isupper() or w[0].isdigit() for w in probe)

    def _headline_end(self, tokens: list[_Token], allow_foreign: bool) -> int:
        index = 0
        while index < len(tokens):
            word = tokens[index].core
            if self._is_prose_word(word):
                if allow_foreign and _is_foreign(word):
                    index = self._skip_foreign_run(tokens, index)
                    continue
                return tokens[self._prose_start(tokens, index)].start
            index += 1
        return tokens[-1].end

    def _is_prose_word(self, word: str) -> bool:
        return bool(word) and word[0].islower() and word.lower() not in self._connectors

    def _prose_start(self, tokens: list[_Token], index: int) -> int:
        """Index of the token that begins the prose sentence containing `tokens[index]`."""
        start = index - 1 if index > 0 and tokens[index - 1].core[:1].isupper() else index
        if start > 0 and tokens[start - 1].core in self._openers:
            start -= 1
        return start

    def _skip_foreign_run(self, tokens: list[_Token], index: int) -> int:
        """Advance past a non-English headline to where Title Case headlines resume."""
        for next_index in range(index + 1, len(tokens) - 1):
            pair = tokens[next_index].core, tokens[next_index + 1].core
            if all(w[:1].isupper() for w in pair) and not any(_is_foreign(w) for w in pair):
                return next_index
        return len(tokens)

    def _sentence_end(self, text: str, tokens: list[_Token]) -> int:
        start = tokens[0].start
        sentence_end = next((end for s, end in sentence_spans(text) if s <= start < end), None)
        for index in range(self._config.min_words_before_boundary, len(tokens)):
            if self._starts_unpunctuated_sentence(tokens, index):
                return min(tokens[index].start, sentence_end or tokens[index].start)
        return sentence_end if sentence_end is not None else tokens[-1].end

    def _starts_unpunctuated_sentence(self, tokens: list[_Token], index: int) -> bool:
        """True where prose begins right after a headline with no punctuation between.

        E.g. "...tied to diversity It's part of" or "...toward the president Tech companies
        have". The word before must be a lowercase content word (so "a Trump critic" or
        "backed by Jeff Bezos" don't count), and the capitalised word must be a sentence
        opener or be followed by a lowercase word (ruling out proper-noun runs).
        """
        previous, current = tokens[index - 1], tokens[index]
        after_lowercase_word = (
            not previous.ends_clause
            and previous.core[:1].islower()
            and previous.core.lower() not in self._connectors
        )
        if not after_lowercase_word or not current.core[:1].isupper():
            return False
        if current.core in self._openers:
            return True
        following = tokens[index + 1].core if index + 1 < len(tokens) else ""
        return following[:1].islower() and following.lower() not in self._connectors


def _is_foreign(word: str) -> bool:
    """True if the word contains a non-ASCII letter (e.g. Spanish "qué", "compró")."""
    return any(not ch.isascii() for ch in _NON_ASCII_LETTER.findall(word))
