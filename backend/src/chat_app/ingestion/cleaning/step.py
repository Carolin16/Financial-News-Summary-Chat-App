"""The interface every cleaning step implements."""

from typing import Protocol

from chat_app.ingestion.cleaning.models import CleaningContext, StepResult


class CleaningStep(Protocol):
    """One independent transformation of article text.

    Steps must be idempotent (applying twice equals applying once) and must never create
    text: they may only remove, or replace characters without introducing digits.
    """

    @property
    def name(self) -> str:
        """Identifier used in configuration and in cleaning signals."""
        ...

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        """Return the transformed text and any signals observed."""
        ...
