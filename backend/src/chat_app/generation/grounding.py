"""Post-generation grounding checks: citations must exist, numbers must be in cited sources.

Prompt rules reduce hallucination; this module enforces the non-negotiable ones in code. A
sentence that cites nothing, or quotes a figure its cited sources don't contain, is removed
rather than shown, and the removal is reported for logging and evaluation.
"""

import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel

from chat_app.core.text_normalization import fold_punctuation
from chat_app.generation.context import Source
from chat_app.ingestion.sentences import split_sentences

_CITATION = re.compile(r"\[(\d+)\]")
# Prose references such as "source 3" or "sources 1 and 4" point at sources; not figures.
_SOURCE_REFERENCE = re.compile(r"\bsources? \d+(?:\s*(?:,|and|&)\s*\d+)*", re.IGNORECASE)
# Digits not glued to a preceding letter/digit/dot, so "Q4", "H100" and "3.5" (as "5")
# are not extracted; thousands separators and decimals are kept as one number.
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
_BULLET = re.compile(r"^(\s*(?:[-*]|\d+\.)\s+)")
_LEADING_CITATIONS = re.compile(r"^((?:\s*\[\d+\])+)\s*")


class RemovalReason(StrEnum):
    """Why a sentence was withheld from the user."""

    UNCITED = "uncited"
    UNSUPPORTED_NUMBER = "unsupported_number"


class RemovedSentence(BaseModel):
    """A withheld sentence and the reason, for logs and the eval suite."""

    sentence: str
    reason: RemovalReason
    detail: str = ""


class GroundingReport(BaseModel):
    """The verified answer plus what was cited and what was removed."""

    text: str
    cited_numbers: list[int]
    removed: list[RemovedSentence]

    @property
    def is_empty(self) -> bool:
        """True if nothing substantive survived verification."""
        return not self.cited_numbers


def extract_numbers(text: str) -> set[str]:
    """Canonical numbers in `text` ("$1,024.05" -> "1024.05"), ignoring citation markers."""
    # Same folding as ingestion, so e.g. a Unicode minus or NBSP in LLM output compares
    # equal to the cleaned article text.
    text = _SOURCE_REFERENCE.sub(" ", _CITATION.sub(" ", fold_punctuation(text)))
    return {
        match.group(1).replace(",", "") + (f".{match.group(2)}" if match.group(2) else "")
        for match in _NUMBER.finditer(text)
    }


def verify_answer(answer: str, sources: Sequence[Source]) -> GroundingReport:
    """Remove ungrounded sentences from `answer`, keeping its line/bullet structure."""
    by_number = {s.number: s for s in sources}
    source_numbers = {s.number: extract_numbers(s.text) for s in sources}
    kept_lines: list[str] = []
    cited: list[int] = []
    removed: list[RemovedSentence] = []

    for line in fold_punctuation(answer).splitlines():
        prefix_match = _BULLET.match(line)
        prefix = prefix_match.group(1) if prefix_match else ""
        kept_sentences = []
        for sentence in _sentences(line[len(prefix) :]):
            sentence = _drop_unknown_citations(sentence, by_number)
            refs = [int(n) for n in _CITATION.findall(sentence)]
            reason = _check(sentence, refs, source_numbers)
            if reason is None:
                kept_sentences.append(sentence)
                cited.extend(r for r in refs if r not in cited)
            else:
                removed.append(reason)
        if kept_sentences:
            kept_lines.append(prefix + " ".join(kept_sentences))
        elif not line.strip() and kept_lines and kept_lines[-1]:
            kept_lines.append("")  # preserve paragraph breaks between surviving blocks

    return GroundingReport(text="\n".join(kept_lines).strip(), cited_numbers=cited, removed=removed)


def _check(
    sentence: str, refs: list[int], source_numbers: dict[int, set[str]]
) -> RemovedSentence | None:
    numbers = extract_numbers(sentence)
    if not refs:
        if _is_lead_in(sentence, numbers):
            return None
        return RemovedSentence(sentence=sentence, reason=RemovalReason.UNCITED)
    supported = set().union(*(source_numbers[r] for r in refs))
    unsupported = sorted(numbers - supported)
    if unsupported:
        return RemovedSentence(
            sentence=sentence,
            reason=RemovalReason.UNSUPPORTED_NUMBER,
            detail=", ".join(unsupported),
        )
    return None


def _is_lead_in(sentence: str, numbers: set[str]) -> bool:
    """Framing like "Here's what the articles report:" carries no claim to cite."""
    return not numbers and sentence.rstrip().endswith(":")


def _drop_unknown_citations(sentence: str, sources: dict[int, Source]) -> str:
    return _CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in sources else "", sentence)


def _sentences(text: str) -> list[str]:
    """Split into sentences, re-attaching misplaced citations.

    A citation written after the full stop ("x. [2] Next") belongs to the sentence before.
    """
    merged: list[str] = []
    for sentence in split_sentences(text):
        leading = _LEADING_CITATIONS.match(sentence)
        if merged and leading:
            merged[-1] = f"{merged[-1]} {leading.group(1).strip()}"
            sentence = sentence[leading.end() :]
        if sentence.strip():
            merged.append(sentence.strip())
    return merged
