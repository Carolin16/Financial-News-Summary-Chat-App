"""AnswerService policy tests with fake retrieval and a scripted summarizer (no network)."""

import pytest

from chat_app.core.models import ArticleMetadata, Chunk, RetrievedChunk
from chat_app.core.ticker_registry import Company, TickerRegistry
from chat_app.generation import messages
from chat_app.generation.answer_service import AnswerService
from chat_app.generation.events import (
    DeltaEvent,
    ErrorEvent,
    FinalEvent,
    MetaEvent,
    SourcesEvent,
)
from chat_app.generation.llm_client import LlmError
from chat_app.generation.query_analysis import QueryAnalyzer
from chat_app.retrieval.coverage import CoverageIndex

REGISTRY = TickerRegistry(
    {
        "NVDA": Company(name="Nvidia", aliases=["Nvidia"]),
        "TSLA": Company(name="Tesla", aliases=["Tesla"]),
        "XOM": Company(name="Exxon", aliases=["Exxon"]),
    }
)


def chunk(article_id: str, text: str, primary=(), mentioned=(), is_stub=False) -> Chunk:
    return Chunk(
        chunk_id=f"{article_id}-0",
        article_id=article_id,
        chunk_index=0,
        title=f"Title {article_id}",
        link=f"https://x/{article_id}",
        text=text,
        header=f"Title: Title {article_id}",
        is_stub=is_stub,
        metadata=ArticleMetadata(primary_tickers=list(primary), mentioned_tickers=list(mentioned)),
    )


NVIDIA_CHUNKS = [
    chunk(f"nv{i}", f"Nvidia rose {i}% on AI demand.", primary=["NVDA"]) for i in range(1, 6)
]
DBS = chunk("dbs", "DBS cut its Nvidia target to $160.", primary=["NVDA"], is_stub=True)
TESLA_MENTION = chunk("mkt", "Tesla fell alongside other megacaps.", mentioned=["TSLA"])
ALL_CHUNKS = [*NVIDIA_CHUNKS, DBS, TESLA_MENTION]


class FakeRetrieval:
    def __init__(self, results: list[Chunk] | Exception):
        self.results = results
        self.calls: list[tuple[str, list[str], str]] = []

    async def fetch(self, question, tickers, intent):
        self.calls.append((question, list(tickers), intent))
        if isinstance(self.results, Exception):
            raise self.results
        return [RetrievedChunk(chunk=c, score=1.0) for c in self.results]


class ScriptedSummarizer:
    def __init__(self, text: str = "", error: Exception | None = None):
        self.text = text
        self.error = error
        self.requests = []

    async def stream(self, request, usage):
        self.requests.append(request)
        for word in self.text.split(" "):
            yield word + " "
        if self.error:
            raise self.error
        usage.input_tokens, usage.output_tokens = 100, 20


class FakeScopeGuard:
    def __init__(self, in_scope: bool = True):
        self.in_scope = in_scope
        self.checked: list[str] = []

    async def is_in_scope(self, question):
        self.checked.append(question)
        return self.in_scope


def service(retrieval, summarizer, scope_guard=None) -> AnswerService:
    return AnswerService(
        scope_guard=scope_guard or FakeScopeGuard(),
        analyzer=QueryAnalyzer(REGISTRY),
        retrieval=retrieval,
        summarizer=summarizer,
        coverage=CoverageIndex(ALL_CHUNKS, full_coverage_min_articles=5),
        registry=REGISTRY,
    )


async def run(svc: AnswerService, question: str):
    return [event async for event in svc.answer(question)]


def final(events) -> FinalEvent:
    assert isinstance(events[-1], FinalEvent)
    return events[-1]


async def test_streams_meta_sources_deltas_then_verified_final():
    svc = service(FakeRetrieval(NVIDIA_CHUNKS[:2]), ScriptedSummarizer("Nvidia rose 1% [1]."))
    events = await run(svc, "What's the news on Nvidia?")

    assert isinstance(events[0], MetaEvent) and events[0].tickers == ["NVDA"]
    assert isinstance(events[1], SourcesEvent) and len(events[1].sources) == 2
    assert any(isinstance(e, DeltaEvent) for e in events)
    result = final(events)
    assert result.answer.strip() == "Nvidia rose 1% [1]."
    assert [c.number for c in result.citations] == [1]


async def test_hallucinated_numbers_never_reach_the_final_answer():
    text = "Nvidia rose 1% [1].\n- Its target is $250 [1]."
    result = final(
        await run(
            service(FakeRetrieval(NVIDIA_CHUNKS[:1]), ScriptedSummarizer(text)), "Nvidia news?"
        )
    )
    assert "$250" not in result.answer
    assert result.removed_sentences == 1


async def test_advice_gets_summary_plus_fixed_disclaimer():
    summarizer = ScriptedSummarizer("Analysts at DBS cut the target to $160 [1].")
    result = final(await run(service(FakeRetrieval([DBS]), summarizer), "Should I buy Nvidia?"))
    assert result.answer.endswith(messages.NOT_ADVICE)
    assert messages.PARTIAL_SOURCES in result.notices
    assert summarizer.requests[0].intent == "advice"


async def test_prediction_gets_disclaimer():
    svc = service(FakeRetrieval(NVIDIA_CHUNKS[:1]), ScriptedSummarizer("Nvidia rose 1% [1]."))
    result = final(await run(svc, "Will Nvidia stock go up?"))
    assert messages.NOT_ADVICE in result.notices


async def test_timing_question_without_company_is_answered_without_llm():
    retrieval, summarizer = FakeRetrieval([]), ScriptedSummarizer()
    result = final(
        await run(service(retrieval, summarizer), "What happened in the market yesterday?")
    )
    assert result.answer.startswith(messages.NO_DATES)
    assert "Nvidia: 6 articles" in result.answer
    assert retrieval.calls == [] and summarizer.requests == []


async def test_live_data_is_answered_with_the_notice_alone():
    retrieval = FakeRetrieval(NVIDIA_CHUNKS[:1])
    summarizer = ScriptedSummarizer("Nvidia's market cap was $3 trillion [1].")
    result = final(await run(service(retrieval, summarizer), "What's Nvidia's current market cap?"))
    assert result.answer.startswith(
        "I don't have live market data, and the articles don't state Nvidia's current market cap"
    )
    # A dated figure beside the notice would read as the current one.
    assert "$3 trillion" not in result.answer and not result.citations
    assert retrieval.calls == [] and summarizer.requests == []


async def test_limited_coverage_is_disclosed_and_llm_told_not_to_pad():
    summarizer = ScriptedSummarizer("Tesla fell alongside other megacaps [1].")
    result = final(await run(service(FakeRetrieval([TESLA_MENTION]), summarizer), "News on Tesla?"))
    assert result.answer.startswith("There's no dedicated news about Tesla in this dataset")
    assert "Do not pad" in summarizer.requests[0].coverage_guidance


async def test_company_with_no_articles_gets_explicit_no_data_without_retrieval():
    retrieval = FakeRetrieval([])
    result = final(await run(service(retrieval, ScriptedSummarizer()), "Any news on Exxon?"))
    assert result.answer == "None of the articles in this dataset mention Exxon."
    assert retrieval.calls == []


async def test_nothing_retrieved_says_so():
    result = final(
        await run(service(FakeRetrieval([]), ScriptedSummarizer()), "Tell me about quantum")
    )
    assert result.answer == messages.NO_GROUNDED_ANSWER


async def test_answer_with_nothing_grounded_falls_back():
    svc = service(FakeRetrieval(NVIDIA_CHUNKS[:1]), ScriptedSummarizer("Nvidia is great."))
    result = final(await run(svc, "Nvidia news?"))
    assert result.answer == messages.NO_GROUNDED_ANSWER


async def test_llm_failure_yields_error_then_graceful_final():
    summarizer = ScriptedSummarizer("partial", error=LlmError("timeout"))
    events = await run(service(FakeRetrieval(NVIDIA_CHUNKS[:1]), summarizer), "Nvidia news?")
    assert any(isinstance(e, ErrorEvent) for e in events)
    assert final(events).answer == messages.ANSWER_UNAVAILABLE


@pytest.mark.parametrize(
    ("question", "intent"),
    [("What's Nvidia's price target?", "price_target"), ("Why did Nvidia jump?", "causal")],
)
async def test_intent_is_passed_to_retrieval_and_summarizer(question, intent):
    retrieval, summarizer = FakeRetrieval(NVIDIA_CHUNKS[:1]), ScriptedSummarizer("x [1].")
    await run(service(retrieval, summarizer), question)
    assert retrieval.calls[0][2] == intent
    assert summarizer.requests[0].intent == intent


async def test_ungrounded_text_is_never_streamed_to_the_client():
    text = "90 + 70 = 160.\n- Nvidia rose 1% [1].\n- Its target is $250 [1]."
    svc = service(FakeRetrieval(NVIDIA_CHUNKS[:1]), ScriptedSummarizer(text))
    events = await run(svc, "Nvidia?")
    streamed = "".join(e.text for e in events if isinstance(e, DeltaEvent))
    assert streamed.strip() == "- Nvidia rose 1% [1]."
    assert "160" not in streamed and "$250" not in streamed


async def test_off_topic_question_streams_nothing():
    svc = service(FakeRetrieval(NVIDIA_CHUNKS[:1]), ScriptedSummarizer("90 + 70 = 160."))
    events = await run(svc, "90 + 70")
    assert not [e for e in events if isinstance(e, DeltaEvent)]
    assert final(events).answer == messages.NO_GROUNDED_ANSWER


async def test_out_of_scope_question_is_declined_before_retrieval():
    retrieval, summarizer, guard = (
        FakeRetrieval(NVIDIA_CHUNKS),
        ScriptedSummarizer(),
        FakeScopeGuard(False),
    )
    events = await run(service(retrieval, summarizer, guard), "Who is the president of the USA?")
    assert final(events).answer == messages.OUT_OF_SCOPE
    assert final(events).citations == []
    assert guard.checked == ["Who is the president of the USA?"]
    assert retrieval.calls == [] and summarizer.requests == []


async def test_clearly_financial_questions_skip_the_scope_guard():
    guard = FakeScopeGuard(False)
    svc = service(
        FakeRetrieval(NVIDIA_CHUNKS[:1]), ScriptedSummarizer("Nvidia rose 1% [1]."), guard
    )
    await run(svc, "What's the news on Nvidia?")
    await run(svc, "How is the stock market doing?")
    assert guard.checked == []


async def test_ambiguous_question_judged_in_scope_is_answered():
    guard = FakeScopeGuard(True)
    svc = service(
        FakeRetrieval(NVIDIA_CHUNKS[:1]), ScriptedSummarizer("Nvidia rose 1% [1]."), guard
    )
    result = final(await run(svc, "Tell me about quantum computing"))
    assert guard.checked and result.citations
