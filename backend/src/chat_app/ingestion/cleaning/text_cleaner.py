"""Composes cleaning steps into the article cleaner."""

from collections.abc import Sequence

from chat_app.core.models import RawArticle
from chat_app.ingestion.cleaning.models import CleanedArticle, CleaningContext, CleaningSignals
from chat_app.ingestion.cleaning.step import CleaningStep


class TextCleaner:
    """Runs body steps and title steps in their configured order and merges signals."""

    def __init__(self, steps: Sequence[CleaningStep], title_steps: Sequence[CleaningStep]) -> None:
        """Steps are injected (built from config by the factory), so order is data."""
        self._steps = list(steps)
        self._title_steps = list(title_steps)

    def clean(self, raw: RawArticle) -> CleanedArticle:
        """Clean one source entry; runs before deduplication and relevance scoring."""
        title = self._run(self._title_steps, raw.title, CleaningContext(title=raw.title))[0]
        text, signals = self._run(self._steps, raw.full_text, CleaningContext(title=title))
        return CleanedArticle(
            title=title, link=raw.link.strip(), ticker=raw.ticker, text=text, signals=signals
        )

    def clean_text(self, text: str, title: str = "") -> str:
        """Clean free text with the body steps (used for idempotency checks and tools)."""
        return self._run(self._steps, text, CleaningContext(title=title))[0]

    @staticmethod
    def _run(
        steps: Sequence[CleaningStep], text: str, context: CleaningContext
    ) -> tuple[str, CleaningSignals]:
        signals = CleaningSignals()
        for step in steps:
            before = len(text)
            result = step.apply(text, context)
            text = result.text
            signals.chars_removed_by_step[step.name] = before - len(text)
            for rule, count in result.chars_removed_by_rule.items():
                signals.chars_removed_by_rule[rule] = (
                    signals.chars_removed_by_rule.get(rule, 0) + count
                )
            signals.truncation_markers.extend(result.truncation_markers)
            signals.footers_kept.extend(result.footers_kept)
        return text, signals
