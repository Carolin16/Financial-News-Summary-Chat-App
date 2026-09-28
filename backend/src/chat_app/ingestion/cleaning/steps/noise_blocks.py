"""Removes boilerplate and promotional blocks defined in the rules file."""

import logging
import re
from collections.abc import Sequence

from chat_app.ingestion.cleaning.config import NoiseRule, Strategy
from chat_app.ingestion.cleaning.footer_guard import FooterGuard
from chat_app.ingestion.cleaning.models import CleaningContext, StepResult
from chat_app.ingestion.cleaning.promo_block import PromoBlockScanner
from chat_app.ingestion.sentences import sentence_spans

logger = logging.getLogger(__name__)

Span = tuple[int, int]


class NoiseBlockStep:
    """Applies each configured rule in order, extending matches per the rule's strategy."""

    name = "noise_blocks"

    def __init__(
        self,
        rules: Sequence[NoiseRule],
        scanner: PromoBlockScanner,
        footer_guard: FooterGuard,
        keep_ranking_labels: bool,
    ) -> None:
        """Rules come from configuration; ranking-label rules are skipped when kept."""
        self._rules = [r for r in rules if not (r.ranking_label and keep_ranking_labels)]
        self._scanner = scanner
        self._footer_guard = footer_guard

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        """Return `text` with every rule applied, plus per-rule signals."""
        removed: dict[str, int] = {}
        truncation: list[str] = []
        footers_kept: list[str] = []
        for rule in self._rules:
            matches = list(rule.regex.finditer(text))
            if not matches:
                continue
            if rule.truncation:
                truncation.append(rule.name)
            spans = self._spans(rule, text, matches, context)
            if spans is None:
                footers_kept.append(rule.name)
                continue
            text, count = _remove(text, spans)
            if count:
                removed[rule.name] = removed.get(rule.name, 0) + count
        return StepResult(
            text=text,
            chars_removed_by_rule=removed,
            truncation_markers=truncation,
            footers_kept=footers_kept,
        )

    def _spans(
        self, rule: NoiseRule, text: str, matches: list[re.Match[str]], context: CleaningContext
    ) -> list[Span] | None:
        """Spans to remove, or None when a footer must be kept for safety."""
        if rule.strategy is Strategy.SPAN:
            return [m.span() for m in matches]
        if rule.strategy is Strategy.SENTENCE:
            sentences = sentence_spans(text)
            return [_containing(sentences, m.start()) for m in matches]
        if rule.strategy is Strategy.PROMO_BLOCK:
            return [(m.start(), self._promo_end(rule, text, m)) for m in matches]
        return self._footer_span(rule, text, matches[0], context)

    def _promo_end(self, rule: NoiseRule, text: str, match: re.Match[str]) -> int:
        """End of a promo block; a parenthesised aside ends at its closing parenthesis.

        Without this, "…giant (read more: X?). NVIDIA has…" would lose the host
        sentence's full stop along with the aside.
        """
        if text[match.start()] == "(":
            close = text.find(")", match.end())
            if close != -1:
                return close + 1
        return self._scanner.block_end(text, match.end(), rule.allow_foreign_headlines)

    def _footer_span(
        self, rule: NoiseRule, text: str, match: re.Match[str], context: CleaningContext
    ) -> list[Span] | None:
        content = self._footer_guard.content_sentences(text[match.start() :], context.title)
        if content:
            logger.warning(
                "footer kept: tail after marker carries article content",
                extra={"rule": rule.name, "title": context.title, "content": content[:3]},
            )
            return None
        return [(match.start(), len(text))]


def _containing(sentences: list[Span], offset: int) -> Span:
    return next(((s, e) for s, e in sentences if s <= offset < e), (offset, offset))


def _remove(text: str, spans: list[Span]) -> tuple[str, int]:
    """Replace merged spans with a single space each; return new text and chars removed."""
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    pieces, cursor, removed = [], 0, 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        pieces.append(" ")
        removed += end - start
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), removed
