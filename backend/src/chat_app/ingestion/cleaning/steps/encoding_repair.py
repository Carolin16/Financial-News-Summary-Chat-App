"""Repairs mis-decoded text (mojibake such as "â€™" for "’")."""

import ftfy

from chat_app.ingestion.cleaning.models import CleaningContext, StepResult


class EncodingRepairStep:
    """Delegates to ftfy's encoding repair.

    Defensive: profiling found no mojibake in the current `stock_news.json` (despite the
    brief expecting it), but scraped feeds routinely contain it, so the step stays in the
    pipeline for other files and future sources. Only `fix_encoding` is used, not
    `fix_text`, so this step repairs encoding and nothing else. Wrapped behind the
    `CleaningStep` interface so the library can be swapped without touching callers.
    """

    name = "encoding_repair"

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        """Return `text` with encoding damage repaired."""
        return StepResult(text=ftfy.fix_encoding(text))
