"""Restores the missing space after sentence punctuation ("compare?The impact")."""

import re

from chat_app.ingestion.cleaning.models import CleaningContext, StepResult

# A lowercase letter, sentence punctuation, then a capitalised word. Requiring letters on
# both sides leaves decimals ("5.3"), tickers ("BRK.B"), and domains ("Amazon.com") alone,
# and inserting a space can never create a number.
_GLUED_SENTENCE = re.compile(r"(?<=[a-z][.!?])(?=[A-Z][a-z])")


class SentenceSpacingStep:
    """Runs before noise removal so sentence-based rules see correct boundaries."""

    name = "sentence_spacing"

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        """Return `text` with a space inserted between glued sentences."""
        return StepResult(text=_GLUED_SENTENCE.sub(" ", text))
