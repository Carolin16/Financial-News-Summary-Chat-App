"""Cuts ads, promos, page clutter, and publisher footers listed in the rules file."""

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
    """Runs each noise rule in turn, deciding how much text around every match to cut."""

    name = "noise_blocks"

    def __init__(
        self,
        rules: Sequence[NoiseRule],
        scanner: PromoBlockScanner,
        footer_guard: FooterGuard,
        keep_ranking_labels: bool,
    ) -> None:
        """Load the noise rules, leaving out ranking-label rules when those labels are kept."""
        self._rules = [r for r in rules if not (r.ranking_label and keep_ranking_labels)]
        self._scanner = scanner
        self._footer_guard = footer_guard

    def apply(self, text: str, context: CleaningContext) -> StepResult:
        """Return the cleaned text and a report of what each rule removed or flagged."""
        removed: dict[str, int] = {}
        truncation: list[str] = []
        footers_kept: list[str] = []
        # Rules run in file order, and each one sees the text left by the rule before it.
        for rule in self._rules:
            matches = list(rule.regex.finditer(text))
            if not matches:
                continue
            # A paywall or "Continue Reading" marker means the article was cut off, so record
            # it for the stub detector even though the marker itself is deleted.
            if rule.truncation:
                truncation.append(rule.name)
            spans = self._spans(rule, text, matches, context)
            if spans is None:
                # The footer guard found real content after the marker, so nothing is cut.
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
        """Work out which parts of the text to cut, or None if a footer is too risky to cut."""
        # Each rule's strategy says how far the cut reaches beyond the matched phrase.
        if rule.strategy is Strategy.SPAN:
            # Just the matched words, e.g. "View Comments".
            return [m.span() for m in matches]
        if rule.strategy is Strategy.SENTENCE:
            # The whole sentence the phrase sits in, e.g. a copyright line.
            sentences = sentence_spans(text)
            return [_containing(sentences, m.start()) for m in matches]
        if rule.strategy is Strategy.PROMO_BLOCK:
            # From the marker ("Trending:", "READ NEXT") to where the article prose resumes.
            return [(m.start(), self._promo_end(rule, text, m)) for m in matches]
        # Otherwise it is a footer: everything from the first match to the end of the article.
        return self._footer_span(rule, text, matches[0], context)

    def _promo_end(self, rule: NoiseRule, text: str, match: re.Match[str]) -> int:
        """Find where a promo ends, stopping at ")" for asides so the real sentence survives."""
        # "...giant (read more: X?). NVIDIA has..." must keep the full stop after ")".
        if text[match.start()] == "(":
            close = text.find(")", match.end())
            if close != -1:
                return close + 1
        return self._scanner.block_end(text, match.end(), rule.allow_foreign_headlines)

    def _footer_span(
        self, rule: NoiseRule, text: str, match: re.Match[str], context: CleaningContext
    ) -> list[Span] | None:
        """Cut from the footer marker to the end, unless real article content follows it."""
        # Look at everything after the marker; if any of it is real reporting, keep it all.
        content = self._footer_guard.content_sentences(text[match.start() :], context.title)
        if content:
            logger.warning(
                "footer kept: tail after marker carries article content",
                extra={"rule": rule.name, "title": context.title, "content": content[:3]},
            )
            return None
        return [(match.start(), len(text))]


def _containing(sentences: list[Span], offset: int) -> Span:
    """Return the sentence that contains the given position."""
    # An empty span (nothing cut) if the position falls outside every sentence.
    return next(((s, e) for s, e in sentences if s <= offset < e), (offset, offset))


def _remove(text: str, spans: list[Span]) -> tuple[str, int]:
    """Cut the given spans (merging overlaps) and return the new text and chars removed."""
    # Merge overlapping spans first, so the same characters are never cut or counted twice.
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    pieces, cursor, removed = [], 0, 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        # Leave a space so the words on either side of the cut don't run together.
        pieces.append(" ")
        removed += end - start
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), removed
