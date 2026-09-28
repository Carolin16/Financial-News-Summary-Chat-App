"""Answers one question end to end: route -> retrieve -> summarise -> verify -> frame.

The service applies the answering rules that must hold regardless of what the LLM writes:
fixed notices for advice, timing, live data and thin coverage, and grounding verification
of the generated text before it is shown as final.
"""

import logging
import time
from collections.abc import AsyncIterator, Sequence

from chat_app.core.interfaces import ScopeGuard, Summarizer
from chat_app.core.models import SummaryRequest, TokenUsage
from chat_app.core.ticker_registry import TickerRegistry
from chat_app.generation import messages
from chat_app.generation.context import Source, format_sources, number_sources
from chat_app.generation.events import (
    AnswerEvent,
    DeltaEvent,
    ErrorEvent,
    FinalEvent,
    MetaEvent,
    NumberedCitation,
    SourcesEvent,
)
from chat_app.generation.grounding import GroundingReport, verify_answer
from chat_app.generation.llm_client import LlmError
from chat_app.generation.query_analysis import Intent, QueryAnalyzer, QueryPlan
from chat_app.retrieval.coverage import CoverageIndex, CoverageLevel, CoverageReport
from chat_app.retrieval.encoders import EmbeddingError
from chat_app.retrieval.strategy import CompanyFirstRetrieval

logger = logging.getLogger(__name__)

# Intents whose answers must end with the not-advice statement.
_ADVICE_LIKE = frozenset({Intent.ADVICE, Intent.PREDICTION})
_COVERAGE_SUMMARY_SIZE = 7


class AnswerService:
    """Coordinates the query pipeline and emits `AnswerEvent`s for streaming."""

    def __init__(
        self,
        analyzer: QueryAnalyzer,
        retrieval: CompanyFirstRetrieval,
        summarizer: Summarizer,
        coverage: CoverageIndex,
        registry: TickerRegistry,
        scope_guard: ScopeGuard,
    ) -> None:
        """Wire collaborators; all are replaceable (e.g. fakes in tests)."""
        self._scope_guard = scope_guard
        self._analyzer = analyzer
        self._retrieval = retrieval
        self._summarizer = summarizer
        self._coverage = coverage
        self._registry = registry

    async def answer(self, question: str) -> AsyncIterator[AnswerEvent]:
        """Yield meta, sources, draft deltas, and finally the verified answer."""
        started = time.perf_counter()
        plan = self._analyzer.analyze(question)
        reports = [self._coverage.assess(t) for t in plan.tickers]
        yield MetaEvent(intent=plan.intent, tickers=plan.tickers, coverage=reports)

        log = _QueryLog(question=question, plan=plan)
        async for event in self._respond(question, plan, reports, log):
            if isinstance(event, FinalEvent):
                log.removed = event.removed_sentences
            yield event
        log.emit(started)

    async def _respond(
        self, question: str, plan: QueryPlan, reports: list[CoverageReport], log: "_QueryLog"
    ) -> AsyncIterator[AnswerEvent]:
        if not plan.clearly_in_scope and not await self._scope_guard.is_in_scope(question):
            log.out_of_scope = True
            yield _final([], messages.OUT_OF_SCOPE, [])
            return

        if plan.intent is Intent.TIMING and not plan.tickers:
            yield self._timing_answer()
            return

        uncovered = [r for r in reports if r.level is CoverageLevel.NONE]
        if reports and len(uncovered) == len(reports):
            notices = [messages.coverage_notice(r, self._name(r.ticker)) or "" for r in reports]
            yield _final(notices, body="", citations=[])
            return

        before = self._leading_notices(plan, reports)
        try:
            retrieved = await self._retrieval.fetch(question, plan.tickers, plan.intent)
        except EmbeddingError as error:
            logger.exception("retrieval failed")
            yield ErrorEvent(message=str(error))
            yield _final(before, messages.ANSWER_UNAVAILABLE, [])
            return
        log.chunk_ids = [r.chunk.chunk_id for r in retrieved]
        if not retrieved:
            yield _final(before, messages.NO_GROUNDED_ANSWER, [])
            return

        sources = number_sources(retrieved)
        yield SourcesEvent(sources=[_numbered(s) for s in sources])
        request = SummaryRequest(
            question=question,
            intent=plan.intent,
            sources=format_sources(sources, plan.tickers),
            coverage_guidance=self._coverage_guidance(reports),
        )

        draft: list[str] = []
        pending = ""
        try:
            async for delta in self._summarizer.stream(request, log.usage):
                draft.append(delta)
                *complete, pending = (pending + delta).split("\n")
                for line in complete:
                    if (shown := _verified_line(line, sources)) is not None:
                        yield DeltaEvent(text=shown + "\n")
        except LlmError as error:
            logger.exception("generation failed")
            yield ErrorEvent(message=str(error))
            yield _final(before, messages.ANSWER_UNAVAILABLE, [])
            return

        if pending and (shown := _verified_line(pending, sources)) is not None:
            yield DeltaEvent(text=shown)
        report = verify_answer("".join(draft), sources)
        yield self._verified_final(plan, before, report, sources)

    def _verified_final(
        self,
        plan: QueryPlan,
        before: list[str],
        report: GroundingReport,
        sources: Sequence[Source],
    ) -> FinalEvent:
        by_number = {s.number: s for s in sources}
        cited = [by_number[n] for n in sorted(report.cited_numbers)]
        after = []
        if any(s.retrieved.chunk.is_stub for s in cited):
            after.append(messages.PARTIAL_SOURCES)
        if plan.intent in _ADVICE_LIKE:
            after.append(messages.NOT_ADVICE)
        body = messages.NO_GROUNDED_ANSWER if report.is_empty else report.text
        return _final(
            before,
            body,
            [_numbered(s) for s in cited],
            after=after,
            removed=len(report.removed),
        )

    def _leading_notices(self, plan: QueryPlan, reports: list[CoverageReport]) -> list[str]:
        notices = []
        if plan.intent is Intent.LIVE_DATA:
            company = self._name(plan.tickers[0]) if plan.tickers else None
            notices.append(messages.live_data_notice(company, plan.live_metric or ""))
        if plan.intent is Intent.TIMING:
            notices.append(messages.NO_DATES)
        for report in reports:
            notice = messages.coverage_notice(report, self._name(report.ticker))
            if notice:
                notices.append(notice)
        return notices

    def _coverage_guidance(self, reports: list[CoverageReport]) -> str:
        return "\n".join(
            messages.coverage_guidance(r, self._name(r.ticker))
            for r in reports
            if r.level is not CoverageLevel.FULL
        )

    def _timing_answer(self) -> FinalEvent:
        lines = [
            f"- {self._name(r.ticker)}: {r.primary_articles} articles"
            for r in self._coverage.most_covered(_COVERAGE_SUMMARY_SIZE, self._registry.tickers)
        ]
        body = (
            f"Instead, here's what the {self._coverage.total_articles} articles cover most:\n"
            + "\n".join(lines)
            + "\n\nAsk me about any of these companies for a summary of their news."
        )
        return _final([messages.NO_DATES], body, [])

    def _name(self, ticker: str) -> str:
        return self._registry.name_for(ticker)


def _verified_line(line: str, sources: Sequence[Source]) -> str | None:
    """A completed draft line as it may be shown while streaming, or None to withhold it.

    Lines are verified before the user sees them, so an ungrounded claim (a figure not in
    its sources, an uncited statement, an off-topic answer) never appears even briefly.
    """
    if not line.strip():
        return ""  # keep paragraph breaks
    return verify_answer(line, sources).text or None


def _numbered(source: Source) -> NumberedCitation:
    citation = source.citation
    return NumberedCitation(
        number=source.number, title=citation.title, link=citation.link, is_partial=citation.is_stub
    )


def _final(
    before: list[str],
    body: str,
    citations: list[NumberedCitation],
    after: list[str] | None = None,
    removed: int = 0,
) -> FinalEvent:
    after = after or []
    parts = [*before, body, *after]
    return FinalEvent(
        answer="\n\n".join(p for p in parts if p),
        citations=citations,
        notices=[*before, *after],
        removed_sentences=removed,
    )


class _QueryLog:
    """Accumulates the per-query structured log line."""

    def __init__(self, question: str, plan: QueryPlan) -> None:
        self.question = question
        self.plan = plan
        self.chunk_ids: list[str] = []
        self.usage = TokenUsage()
        self.removed = 0
        self.out_of_scope = False

    def emit(self, started: float) -> None:
        logger.info(
            "query answered",
            extra={
                "query": self.question,
                "intent": self.plan.intent.value,
                "tickers": self.plan.tickers,
                "retrieved_chunk_ids": self.chunk_ids,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "removed_sentences": self.removed,
                "out_of_scope": self.out_of_scope,
            },
        )
