"""Tidies whitespace and punctuation spacing left behind by removals."""

import re

from chat_app.ingestion.cleaning.models import CleaningContext, StepResult

_WHITESPACE = re.compile(r"\s+")
# " ." -> "." but never before a digit: "5 .3" must not become the new number "5.3".
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?)\]])(?!\d)")
_SPACE_AFTER_OPENING = re.compile(r"([(\[])\s+")
# Punctuation orphaned by a removal, e.g. "stock. . Next". Whitespace between the marks is
# required so a genuine ellipsis ("now... and") is left alone.
_DANGLING_PUNCTUATION = re.compile(r"([.!?])(?:\s+[.,;:])+(?=\s|$)")
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]")


class WhitespaceStep:
    """Collapses runs of whitespace and fixes spacing around punctuation."""

    name = "whitespace"

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        """Return tidied text; idempotent by construction (each rule is a fixpoint)."""
        text = _EMPTY_BRACKETS.sub(" ", text)
        text = _WHITESPACE.sub(" ", text)
        text = _DANGLING_PUNCTUATION.sub(r"\1", text)
        text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
        text = _SPACE_AFTER_OPENING.sub(r"\1", text)
        return StepResult(text=text.strip())
