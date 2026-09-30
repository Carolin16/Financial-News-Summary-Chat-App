"""Fixes garbled characters, e.g. turns "â€™" back into "’"."""

import ftfy

from chat_app.ingestion.cleaning.models import CleaningContext, StepResult


class EncodingRepairStep:
    """Fixes character-encoding damage only, kept as a safeguard for future data."""

    name = "encoding_repair"

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        """Return the text with any garbled characters fixed."""
        return StepResult(text=ftfy.fix_encoding(text))
