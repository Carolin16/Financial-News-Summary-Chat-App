"""Removes every match of a configured list of patterns (used for symbols and URLs)."""

import re
from collections.abc import Iterable

from chat_app.ingestion.cleaning.models import CleaningContext, StepResult


class PatternRemovalStep:
    """Replaces each match with a space; whitespace is tidied by a later step.

    Replacing with a space (not the empty string) keeps neighbouring tokens apart, so a
    removal can never glue two numbers into a new one.
    """

    def __init__(self, name: str, patterns: Iterable[str]) -> None:
        """Name the step (as referenced in the rules file) and compile its patterns."""
        self.name = name
        self._patterns = [re.compile(p) for p in patterns]

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        """Return `text` without any pattern matches."""
        removed = 0
        for pattern in self._patterns:
            removed += sum(len(m.group()) for m in pattern.finditer(text))
            text = pattern.sub(" ", text)
        return StepResult(text=text, chars_removed_by_rule={self.name: removed} if removed else {})
