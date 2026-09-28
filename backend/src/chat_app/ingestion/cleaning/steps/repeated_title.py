"""Removes the headline when the body repeats it as its first words."""

import re

from chat_app.core.text_normalization import fold_punctuation
from chat_app.ingestion.cleaning.models import CleaningContext, StepResult

_WHITESPACE = re.compile(r"\s+")


class RepeatedTitleStep:
    """Strips a leading copy of the title; the title itself is kept on the article."""

    name = "repeated_title"

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        """Return `text` without a leading repetition of `context.title`."""
        title = _normalise(fold_punctuation(context.title))
        if not title:
            return StepResult(text=text)
        body = text.lstrip()
        # Compare case-insensitively with whitespace collapsed, then cut the same number
        # of words from the original so its exact characters are preserved.
        if not _normalise(body).lower().startswith(title.lower()):
            return StepResult(text=text)
        title_words = len(title.split())
        remainder = body.split(maxsplit=title_words)
        rest = remainder[title_words] if len(remainder) > title_words else ""
        if rest and _normalise(" ".join(remainder[:title_words])).lower() != title.lower():
            return StepResult(text=text)  # word boundary didn't line up; leave it alone
        return StepResult(text=rest, chars_removed_by_rule={self.name: len(text) - len(rest)})


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()
