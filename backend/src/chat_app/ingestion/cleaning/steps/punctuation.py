"""Folds typographic punctuation and space variants to ASCII."""

from chat_app.core.text_normalization import fold_punctuation
from chat_app.ingestion.cleaning.models import CleaningContext, StepResult


class PunctuationNormalizationStep:
    """Uses the shared normaliser, so answer verification folds text identically."""

    name = "punctuation"

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        """Return `text` with quotes, dashes, ellipses, and odd spaces folded."""
        return StepResult(text=fold_punctuation(text))
