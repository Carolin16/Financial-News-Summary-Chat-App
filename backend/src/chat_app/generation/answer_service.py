"""Answers questions using only the news articles, and drops any claim they do not support."""

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

# Advice or prediction questions ("Should I buy?", "Will it go up?") end with a not-advice notice.
_ADVICE_LIKE = frozenset({Intent.ADVICE, Intent.PREDICTION})
# How many top companies to list when we answer "what happened yesterday?" with coverage.
_COVERAGE_SUMMARY_SIZE = 7


class AnswerService:
    """Runs each question through the answer pipeline and streams events to the client."""

    def __init__(
        self,
        analyzer: QueryAnalyzer,
        retrieval: CompanyFirstRetrieval,
        summarizer: Summarizer,
        coverage: CoverageIndex,
        registry: TickerRegistry,
        scope_guard: ScopeGuard,
    ) -> None:
        """Store the pipeline's parts; each can be swapped (e.g. for fakes in tests)."""
        self._scope_guard = scope_guard
        self._analyzer = analyzer
        self._retrieval = retrieval
        self._summarizer = summarizer
        self._coverage = coverage
        self._registry = registry

    async def answer(self, question: str) -> AsyncIterator[AnswerEvent]:
        """Stream how the question was read, its sources, checked lines, then the final answer."""
        started = time.perf_counter()
        # Step 1: understand the question (which companies, what kind of question).
        plan = self._analyzer.analyze(question)
        # Step 2: check how much news we have on each company.
        reports = [self._coverage.assess(t) for t in plan.tickers]
        # Let the screen show how we read the question before the answer arrives.
        yield MetaEvent(intent=plan.intent, tickers=plan.tickers, coverage=reports)

        log = _QueryLog(question=question, plan=plan)
        async for event in self._respond(question, plan, reports, log):
            if isinstance(event, FinalEvent):
                log.removed = event.removed_sentences
            yield event
        # Record this query (sources used, time taken, tokens) in the logs.
        log.emit(started)

    async def _respond(
        self, question: str, plan: QueryPlan, reports: list[CoverageReport], log: "_QueryLog"
    ) -> AsyncIterator[AnswerEvent]:
        """Pick the right kind of answer, stopping early when no LLM summary is needed."""
        # Not about financial news? Politely decline without searching or calling the LLM.
        if not plan.clearly_in_scope and not await self._scope_guard.is_in_scope(question):
            log.out_of_scope = True
            yield _final([], messages.OUT_OF_SCOPE, [])
            return

        # Articles have no reliable dates, so for "yesterday" questions show what we cover instead.
        if plan.intent is Intent.TIMING and not plan.tickers:
            yield self._timing_answer()
            return

        # A live figure can't come from dated articles, and any figure the LLM added next to
        # the notice would read as the current one, so answer with the notice alone.
        if plan.intent is Intent.LIVE_DATA:
            yield _final(self._leading_notices(plan, reports), body="", citations=[])
            return

        # We have no news on any company asked about, so say that rather than guess.
        uncovered = [r for r in reports if r.level is CoverageLevel.NONE]
        if reports and len(uncovered) == len(reports):
            notices = [messages.coverage_notice(r, self._name(r.ticker)) or "" for r in reports]
            yield _final(notices, body="", citations=[])
            return

        # Warnings to show above the answer, e.g. "coverage of IBM is limited".
        before = self._leading_notices(plan, reports)
        try:
            # Step 3: search the news, preferring articles mainly about these companies.
            retrieved = await self._retrieval.fetch(question, plan.tickers, plan.intent)
        except EmbeddingError as error:
            # Search is unavailable, so show an error instead of answering without sources.
            logger.exception("retrieval failed")
            yield ErrorEvent(message=str(error))
            yield _final(before, messages.ANSWER_UNAVAILABLE, [])
            return
        log.chunk_ids = [r.chunk.chunk_id for r in retrieved]
        # Found nothing relevant, so say so instead of asking the LLM to make something up.
        if not retrieved:
            yield _final(before, messages.NO_GROUNDED_ANSWER, [])
            return

        # Number the sources [1], [2], ... so the answer can point to each one.
        sources = number_sources(retrieved)
        yield SourcesEvent(sources=[_numbered(s) for s in sources])
        request = SummaryRequest(
            question=question,
            intent=plan.intent,
            sources=format_sources(sources, plan.tickers),
            coverage_guidance=self._coverage_guidance(reports),
        )

        # Step 4: the LLM writes the answer piece by piece. Each finished line is checked
        # against the sources before it is shown. `pending` is the line not finished yet.
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
            # The LLM failed, so show an error instead of half an answer.
            logger.exception("generation failed")
            yield ErrorEvent(message=str(error))
            yield _final(before, messages.ANSWER_UNAVAILABLE, [])
            return

        # Check the last line too, since it has no line break to trigger the check above.
        if pending and (shown := _verified_line(pending, sources)) is not None:
            yield DeltaEvent(text=shown)
        # Step 5: check the whole answer once more and send the final, cleaned version.
        report = verify_answer("".join(draft), sources)
        yield self._verified_final(plan, before, report, sources)

    def _verified_final(
        self,
        plan: QueryPlan,
        before: list[str],
        report: GroundingReport,
        sources: Sequence[Source],
    ) -> FinalEvent:
        """Build the final answer from the checked draft, adding closing notices as needed."""
        # Show only the sources the answer actually used.
        by_number = {s.number: s for s in sources}
        cited = [by_number[n] for n in sorted(report.cited_numbers)]
        after = []
        # Add a warning if any source used was a paywalled or partial article.
        if any(s.retrieved.chunk.is_stub for s in cited):
            after.append(messages.PARTIAL_SOURCES)
        # Add "not investment advice" to buy/sell and prediction answers.
        if plan.intent in _ADVICE_LIKE:
            after.append(messages.NOT_ADVICE)
        # If every line failed the check, say we have no answer backed by the news.
        body = messages.NO_GROUNDED_ANSWER if report.is_empty else report.text
        return _final(
            before,
            body,
            [_numbered(s) for s in cited],
            after=after,
            removed=len(report.removed),
        )

    def _leading_notices(self, plan: QueryPlan, reports: list[CoverageReport]) -> list[str]:
        """Caveats shown above the answer: no live data, no dates, thin coverage."""
        notices = []
        # Questions like "current market cap" need live data the news doesn't have.
        if plan.intent is Intent.LIVE_DATA:
            company = self._name(plan.tickers[0]) if plan.tickers else None
            notices.append(messages.live_data_notice(company, plan.live_metric or ""))
        if plan.intent is Intent.TIMING:
            notices.append(messages.NO_DATES)
        # Add a note for each company we have little or no news on.
        for report in reports:
            notice = messages.coverage_notice(report, self._name(report.ticker))
            if notice:
                notices.append(notice)
        return notices

    def _coverage_guidance(self, reports: list[CoverageReport]) -> str:
        """Tell the LLM not to pad answers about thinly covered companies."""
        return "\n".join(
            messages.coverage_guidance(r, self._name(r.ticker))
            for r in reports
            if r.level is not CoverageLevel.FULL
        )

    def _timing_answer(self) -> FinalEvent:
        """Explain that dates can't be verified and list the most-covered companies instead."""
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
        """Company name for a ticker, for user-facing wording."""
        return self._registry.name_for(ticker)


def _verified_line(line: str, sources: Sequence[Source]) -> str | None:
    """Return the line if its sources back it up, or None so it never appears on screen."""
    if not line.strip():
        return ""  # empty lines are paragraph breaks, keep them
    return verify_answer(line, sources).text or None


def _numbered(source: Source) -> NumberedCitation:
    """The citation as the client sees it: its [n] number, title, link and passage ID."""
    citation = source.citation
    return NumberedCitation(
        number=source.number,
        title=citation.title,
        link=citation.link,
        is_partial=citation.is_stub,
        chunk_id=source.retrieved.chunk.chunk_id,
    )


def _final(
    before: list[str],
    body: str,
    citations: list[NumberedCitation],
    after: list[str] | None = None,
    removed: int = 0,
) -> FinalEvent:
    """Join notices above, the body, and notices below into one final answer."""
    after = after or []
    parts = [*before, body, *after]
    return FinalEvent(
        answer="\n\n".join(p for p in parts if p),
        citations=citations,
        notices=[*before, *after],
        removed_sentences=removed,
    )


class _QueryLog:
    """Collects what happened during one query and writes it as one structured log line."""

    def __init__(self, question: str, plan: QueryPlan) -> None:
        """Start an empty record; fields are filled in as the pipeline runs."""
        self.question = question
        self.plan = plan
        self.chunk_ids: list[str] = []
        self.usage = TokenUsage()
        self.removed = 0
        self.out_of_scope = False

    def emit(self, started: float) -> None:
        """Log the query, sources used, latency, token usage and removed sentences."""
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
