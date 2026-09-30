"""Turns fancy quotes, dashes, ellipses, and odd spaces into plain keyboard characters."""

from chat_app.core.text_normalization import fold_punctuation
from chat_app.ingestion.cleaning.models import CleaningContext, StepResult


class PunctuationNormalizationStep:
    """Uses the same helper as answer checking, so quotes and numbers match exactly."""

    name = "punctuation"

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        """Return the text with all punctuation in plain form."""
        return StepResult(text=fold_punctuation(text))
