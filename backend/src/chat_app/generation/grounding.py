"""Checks the LLM's answer and removes any sentence that is uncited or uses a made-up number."""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel

from chat_app.core.text_normalization import fold_punctuation
from chat_app.generation.context import Source
from chat_app.ingestion.sentences import split_sentences

# A citation marker such as [3].
_CITATION = re.compile(r"\[(\d+)\]")

# The same marker with the space before it, for removing citations cleanly.
_CITATION_WITH_SPACE = re.compile(r"\s*\[\d+\]")

# Words like "source 3" or "sources 1 and 4" point to sources, so the digits are not figures.
_SOURCE_REFERENCE = re.compile(r"\bsources? \d+(?:\s*(?:,|and|&)\s*\d+)*", re.IGNORECASE)

# A standalone number such as "1,024.05", but not the digits in "Q4" or "H100".
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")

# A number plus its unit, if any: "15%", "1.2 billion", "$1.2B" (a lone letter must be
# attached, so "Model 3 M" is not 3 million).
_QUANTITY = re.compile(
    r"(?<![\w.])(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?"
    r"(?:(?P<pct>\s*%|\s*(?i:per\s?cent)\b)"
    r"|\s*(?P<word>(?i:trillion|billion|million|thousand|tn|bn|mn))\b"
    r"|(?P<letter>[TBMK])\b)?"
)
# What each scale word or letter multiplies by.
_SCALES: dict[str, int] = {
    "trillion": 10**12,
    "tn": 10**12,
    "t": 10**12,
    "billion": 10**9,
    "bn": 10**9,
    "b": 10**9,
    "million": 10**6,
    "mn": 10**6,
    "m": 10**6,
    "thousand": 10**3,
    "k": 10**3,
}
# The "-" or "to" between the two ends of a range, as in "7-9%" or "1 to 2 billion".
_RANGE_JOINER = re.compile(r"\s*(?:-|to)\s*")

# A list marker at the start of a line: "-", "*" or "1.".
_BULLET = re.compile(r"^(\s*(?:[-*]|\d+\.)\s+)")

# Citations sitting at the very start of a sentence, e.g. "[2] Shares rose...".
_LEADING_CITATIONS = re.compile(r"^((?:\s*\[\d+\])+)\s*")


class RemovalReason(StrEnum):
    """Why a sentence was hidden from the user."""

    UNCITED = "uncited"  # it points to no source
    UNSUPPORTED_NUMBER = "unsupported_number"  # a figure its sources do not contain


class RemovedSentence(BaseModel):
    """A hidden sentence and why it was hidden, kept for the logs and the tests."""

    sentence: str
    reason: RemovalReason
    detail: str = ""


class GroundingReport(BaseModel):
    """The checked answer, which sources it cites, and what was removed."""

    text: str
    cited_numbers: list[int]
    removed: list[RemovedSentence]

    @property
    def is_empty(self) -> bool:
        """True if no cited sentence survived, i.e. there is no backed-up answer."""
        return not self.cited_numbers


@dataclass(frozen=True)
class Quantity:
    """A number in a comparable form, so "$1.2B", "1.2 billion" and "1,200 million" match."""

    value: Decimal
    is_percent: bool = False

    def __str__(self) -> str:
        """Show the figure plainly for logs, e.g. "1200000000" or "15%"."""
        return f"{self.value:f}{'%' if self.is_percent else ''}"


def extract_numbers(text: str) -> set[str]:
    """Every plain number in the text, e.g. "$1,024.05" becomes "1024.05" (units ignored)."""
    return {
        match.group(1).replace(",", "") + (f".{match.group(2)}" if match.group(2) else "")
        for match in _NUMBER.finditer(_without_references(text))
    }


def extract_quantities(text: str) -> set[Quantity]:
    """Every figure in the text with its unit understood, e.g. "7-9%" gives 7% and 9%."""
    text = _without_references(text)
    matches = list(_QUANTITY.finditer(text))
    quantities = set()
    for index, match in enumerate(matches):
        # In a range like "7-9%", the first number has no unit of its own, so borrow the
        # unit of the number after it.
        unit_source = match
        following = matches[index + 1] if index + 1 < len(matches) else None
        if (
            not _has_unit(match)
            and following is not None
            and _has_unit(following)
            and _RANGE_JOINER.fullmatch(text, match.end(), following.start())
        ):
            unit_source = following
        quantities.add(_quantity(match, unit_source))
    return quantities


def split_list_marker(line: str) -> tuple[str, str]:
    """Split off a line's bullet or number marker, so it can be put back after checking."""
    marker = _BULLET.match(line)
    prefix = marker.group(1) if marker else ""
    return prefix, line[len(prefix) :]


def citation_numbers(sentence: str) -> list[int]:
    """Return the source numbers a sentence cites, e.g. [1, 3]."""
    return [int(n) for n in _CITATION.findall(sentence)]


def strip_citations(sentence: str) -> str:
    """Return the sentence without its [n] markers."""
    return " ".join(_CITATION_WITH_SPACE.sub("", sentence).split())


def is_lead_in(sentence: str) -> bool:
    """True for an intro line like "Here's what the articles report:", which needs no citation."""
    return not extract_quantities(sentence) and sentence.rstrip().endswith(":")


def split_answer_sentences(text: str) -> list[str]:
    """Split a line into sentences, moving a stray "[2]" after a full stop back where it belongs."""
    merged: list[str] = []
    for sentence in split_sentences(text):
        # "x. [2] Next" splits as "x." and "[2] Next", so give the [2] back to "x.".
        leading = _LEADING_CITATIONS.match(sentence)
        if merged and leading:
            merged[-1] = f"{merged[-1]} {leading.group(1).strip()}"
            sentence = sentence[leading.end() :]
        if sentence.strip():
            merged.append(sentence.strip())
    return merged


def answer_sentences(answer: str) -> Iterator[str]:
    """Every sentence in the answer, across all lines and bullet points."""
    for line in fold_punctuation(answer).splitlines():
        _, body = split_list_marker(line)
        yield from split_answer_sentences(body)


def verify_answer(answer: str, sources: Sequence[Source]) -> GroundingReport:
    """Keep only sentences backed by their cited sources, preserving lines and bullets."""
    by_number = {s.number: s for s in sources}
    # Read the figures in each source once, up front.
    source_quantities = {s.number: extract_quantities(s.text) for s in sources}
    kept_lines: list[str] = []
    cited: list[int] = []
    removed: list[RemovedSentence] = []

    for line in fold_punctuation(answer).splitlines():
        prefix, body = split_list_marker(line)
        kept_sentences = []
        for sentence in split_answer_sentences(body):
            # Ignore citations to sources that don't exist, e.g. [9] when there are 8.
            sentence = _drop_unknown_citations(sentence, by_number)
            refs = citation_numbers(sentence)
            reason = _check(sentence, refs, source_quantities)
            if reason is None:
                kept_sentences.append(sentence)
                cited.extend(r for r in refs if r not in cited)
            else:
                removed.append(reason)
        if kept_sentences:
            kept_lines.append(prefix + " ".join(kept_sentences))
        elif not line.strip() and kept_lines and kept_lines[-1]:
            kept_lines.append("")  # keep the blank line between paragraphs

    return GroundingReport(text="\n".join(kept_lines).strip(), cited_numbers=cited, removed=removed)


def _check(
    sentence: str, refs: list[int], source_quantities: dict[int, set[Quantity]]
) -> RemovedSentence | None:
    """Return why a sentence must be hidden, or None if it passes both checks."""
    # Check 1: every claim must cite a source.
    if not refs:
        if is_lead_in(sentence):
            return None
        return RemovedSentence(sentence=sentence, reason=RemovalReason.UNCITED)
    # Check 2: every figure must appear in at least one of the sources it cites.
    supported: set[Quantity] = set().union(*(source_quantities[r] for r in refs))
    unsupported = sorted(str(q) for q in extract_quantities(sentence) - supported)
    if unsupported:
        return RemovedSentence(
            sentence=sentence,
            reason=RemovalReason.UNSUPPORTED_NUMBER,
            detail=", ".join(unsupported),
        )
    return None


def _without_references(text: str) -> str:
    """Remove [n] markers and "source 3" wording, so their digits are not read as figures."""
    # Tidy punctuation the same way ingestion did, so both sides are written alike.
    return _SOURCE_REFERENCE.sub(" ", _CITATION.sub(" ", fold_punctuation(text)))


def _has_unit(match: re.Match[str]) -> bool:
    """True if the number came with a unit such as %, billion or B."""
    return any(match.group(g) for g in ("pct", "word", "letter"))


def _quantity(number: re.Match[str], unit_source: re.Match[str]) -> Quantity:
    """Turn a matched number into a Quantity, applying its unit (e.g. 1.2B -> 1200000000)."""
    digits = number.group("int").replace(",", "")
    if number.group("frac"):
        digits += f".{number.group('frac')}"
    scale_word = unit_source.group("word") or unit_source.group("letter") or ""
    value = Decimal(digits) * _SCALES.get(scale_word.lower(), 1)
    return Quantity(value=value.normalize(), is_percent=bool(unit_source.group("pct")))


def _drop_unknown_citations(sentence: str, sources: dict[int, Source]) -> str:
    """Remove citation markers that point to a source number that doesn't exist."""
    return _CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in sources else "", sentence)
