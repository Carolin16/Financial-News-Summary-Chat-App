"""Cleans each article by running its title and body through the cleaning steps in order."""

from collections.abc import Sequence

from chat_app.core.models import RawArticle
from chat_app.ingestion.cleaning.models import (
    CleanedArticle,
    CleaningContext,
    CleaningSignals,
    StepResult,
)
from chat_app.ingestion.cleaning.step import CleaningStep


class TextCleaner:
    """Applies the cleaning steps one after another and records what each one removed."""

    def __init__(self, steps: Sequence[CleaningStep], title_steps: Sequence[CleaningStep]) -> None:
        """Take the body and title steps, already built and ordered from the rules file."""
        self._steps = list(steps)
        self._title_steps = list(title_steps)

    def clean(self, raw: RawArticle) -> CleanedArticle:
        """Clean one raw article's title and body, before deduplication and relevance scoring."""
        title = self._run(self._title_steps, raw.title, CleaningContext(title=raw.title))[0]
        text, signals = self._run(self._steps, raw.full_text, CleaningContext(title=title))
        return CleanedArticle(
            title=title, link=raw.link.strip(), ticker=raw.ticker, text=text, signals=signals
        )

    def clean_text(self, text: str, title: str = "") -> str:
        """Clean any piece of text with the body steps (handy for tests and tools)."""
        return self._run(self._steps, text, CleaningContext(title=title))[0]

    @staticmethod
    def _run(
        steps: Sequence[CleaningStep], text: str, context: CleaningContext
    ) -> tuple[str, CleaningSignals]:
        """Pass the text through each step in turn and collect what every step reported."""
        signals = CleaningSignals()
        for step in steps:
            result = step.apply(text, context)
            _record(signals, step.name, len(text) - len(result.text), result)
            text = result.text
        return text, signals


def _record(
    signals: CleaningSignals, step_name: str, chars_removed: int, result: StepResult
) -> None:
    """Add one step's report to the running totals."""
    signals.chars_removed_by_step[step_name] = chars_removed
    for rule, count in result.chars_removed_by_rule.items():
        signals.chars_removed_by_rule[rule] = signals.chars_removed_by_rule.get(rule, 0) + count
    signals.truncation_markers.extend(result.truncation_markers)
    signals.footers_kept.extend(result.footers_kept)
