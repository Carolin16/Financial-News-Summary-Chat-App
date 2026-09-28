"""The 12 reference queries, run end to end against the live LLM and the built index.

Run with `pytest -m eval` after indexing. Checks are rule-based and independent of the app's
own verifier: numbers are re-checked against the *original* dataset text of the cited
articles, and each category asserts its answering rule (disclaimers, no padding, ...).
"""

import asyncio
import re
from pathlib import Path

import pytest

from chat_app.api.container import Container
from chat_app.config.settings import get_settings
from chat_app.generation import messages
from chat_app.generation.events import FinalEvent, MetaEvent
from chat_app.generation.grounding import extract_numbers
from chat_app.ingestion.cleaner import TextCleaner, normalize_characters
from chat_app.ingestion.loader import JsonArticleRepository

pytestmark = pytest.mark.eval

REFERENCE_QUERIES = {
    1: "What's the latest news on Intel?",
    2: "What's happening with Apple?",
    3: "Any news on IBM?",
    4: "What do analysts say about Intel?",
    5: "What's Nvidia's price target?",
    6: "Why did Intel stock jump?",
    7: "What's the news on Tesla?",
    8: "What's the news on Google?",
    9: "Should I buy Nvidia?",
    10: "Will Apple stock go up?",
    11: "What's Apple's current market cap?",
    12: "What happened in the market yesterday?",
}

OFF_TOPIC_APPLE_TITLES = ("Occidental", "Suze Orman", "Economic Boycott", "Individual Credit Cards")
SELF_VOICE_FORECAST = re.compile(
    r"\b(I (think|believe|expect|predict)|you should (buy|sell)|I recommend|is a (good|great) buy)",
    re.IGNORECASE,
)


def _skip_reason() -> str | None:
    settings = get_settings()
    if not settings.openai_api_key.get_secret_value():
        return "OPENAI_API_KEY not set"
    local = settings.qdrant_local_path
    if local is not None and not Path(local).exists():
        return "embedded index not built; run chat-index first"
    return None


@pytest.fixture(scope="module")
def results() -> dict[int, tuple[MetaEvent, FinalEvent]]:
    if reason := _skip_reason():
        pytest.skip(reason)

    async def collect() -> dict[int, tuple[MetaEvent, FinalEvent]]:
        container = Container(get_settings())
        service = await container.answer_service()

        async def one(question: str) -> tuple[MetaEvent, FinalEvent]:
            events = [e async for e in service.answer(question)]
            meta = next(e for e in events if isinstance(e, MetaEvent))
            return meta, next(e for e in reversed(events) if isinstance(e, FinalEvent))

        try:
            answers = await asyncio.gather(*(one(q) for q in REFERENCE_QUERIES.values()))
        finally:
            await container.close()
        return dict(zip(REFERENCE_QUERIES, answers, strict=True))

    return asyncio.run(collect())


@pytest.fixture(scope="module")
def article_text_by_link() -> dict[str, str]:
    """Original dataset text (title + cleaned body) keyed by link."""
    cleaner = TextCleaner()
    return {
        raw.link: f"{normalize_characters(raw.title)}\n{cleaner.clean(raw.full_text)}"
        for raw in JsonArticleRepository(get_settings().data_path).load()
    }


def llm_body(final: FinalEvent) -> str:
    """The generated part of the answer, without the app's fixed notices."""
    body = final.answer
    for notice in final.notices:
        body = body.replace(notice, "")
    return body.strip()


def assert_grounded(final: FinalEvent, article_text_by_link: dict[str, str]) -> None:
    """Citations exist and every number in the generated text is in a cited article."""
    body = llm_body(final)
    assert final.citations, "answer has no citations"
    cited_numbers = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    assert cited_numbers <= {c.number for c in final.citations}, "citation to unknown source"
    cited_text = " ".join(article_text_by_link[c.link] for c in final.citations)
    untraceable = extract_numbers(body) - extract_numbers(cited_text)
    assert not untraceable, f"numbers not in cited articles: {untraceable}"


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_generated_answers_are_cited_and_numbers_traceable(results, article_text_by_link, number):
    _, final = results[number]
    assert_grounded(final, article_text_by_link)


def test_q1_intel_news_is_about_intel(results):
    meta, final = results[1]
    assert meta.tickers == ["INTC"]
    assert re.search(r"Broadcom|TSMC|foundry", final.answer)


def test_q2_apple_excludes_off_topic_entries(results):
    _, final = results[2]
    assert not [c.title for c in final.citations if c.title.startswith(OFF_TOPIC_APPLE_TITLES)]


def test_q3_ibm_is_flagged_thin_and_not_padded(results):
    meta, final = results[3]
    assert final.answer.startswith("Coverage of IBM in this news set is limited")
    assert meta.coverage[0].level == "limited"
    for padding in ("Market to Reach", "Market Size", "Data Brokers Market", "Appoints"):
        assert not [c for c in final.citations if padding in c.title], padding


def test_q4_analyst_views_are_attributed_to_firms(results):
    _, final = results[4]
    firms = re.findall(r"Cantor|Evercore|Citic|Raymond James|Lynx|Melius", final.answer)
    assert len(set(firms)) >= 2


def test_q5_price_target_separates_firm_from_consensus(results):
    _, final = results[5]
    sentences = re.split(r"(?<=[.!?])\s+|\n", final.answer)
    firm = [s for s in sentences if "$160" in s]
    consensus = [s for s in sentences if "174.93" in s]
    assert firm and all("DBS" in s for s in firm)
    assert consensus and all(re.search(r"mean|average|consensus", s, re.I) for s in consensus)


def test_q6_causal_uses_the_articles_cause(results):
    _, final = results[6]
    assert re.search(r"Broadcom|TSMC", final.answer)


@pytest.mark.parametrize(("number", "company"), [(7, "Tesla"), (8, "Alphabet (Google)")])
def test_q7_q8_partial_coverage_is_disclosed_not_denied(results, number, company):
    _, final = results[number]
    assert final.answer.startswith(f"Coverage of {company} in this news set is limited")
    assert "None of the articles" not in final.answer


@pytest.mark.parametrize("number", [9, 10])
def test_q9_q10_advice_and_prediction_are_grounded_refusals(results, number):
    _, final = results[number]
    assert final.answer.endswith(messages.NOT_ADVICE)
    assert not SELF_VOICE_FORECAST.search(llm_body(final))


def test_q11_market_cap_is_not_answered_from_memory(results):
    _, final = results[11]
    assert final.answer.startswith(messages.live_data_notice("Apple", "market cap"))
    assert not re.search(r"\$\s?\d[\d.,]*\s*(trillion|billion)\b", llm_body(final), re.I)


def test_q12_yesterday_disclaims_timing_and_offers_coverage(results):
    _, final = results[12]
    assert final.answer.startswith(messages.NO_DATES)
    assert "Intel" in final.answer and not final.citations


OUT_OF_SCOPE_QUESTIONS = ["Who is the president of the USA?", "90 + 70", "Capital of France?"]
BORDERLINE_IN_SCOPE = ["Tell me about quantum computing news", "What is happening with AI?"]


@pytest.fixture(scope="module")
def scope_results() -> dict[str, FinalEvent]:
    if reason := _skip_reason():
        pytest.skip(reason)

    async def collect() -> dict[str, FinalEvent]:
        container = Container(get_settings())
        service = await container.answer_service()

        async def one(question: str) -> FinalEvent:
            events = [e async for e in service.answer(question)]
            return next(e for e in reversed(events) if isinstance(e, FinalEvent))

        questions = OUT_OF_SCOPE_QUESTIONS + BORDERLINE_IN_SCOPE
        try:
            answers = await asyncio.gather(*(one(q) for q in questions))
        finally:
            await container.close()
        return dict(zip(questions, answers, strict=True))

    return asyncio.run(collect())


@pytest.mark.parametrize("question", OUT_OF_SCOPE_QUESTIONS)
def test_out_of_scope_questions_are_declined_without_sources(scope_results, question):
    final = scope_results[question]
    assert final.answer == messages.OUT_OF_SCOPE
    assert not final.citations


@pytest.mark.parametrize("question", BORDERLINE_IN_SCOPE)
def test_borderline_business_questions_are_still_answered(scope_results, question):
    assert scope_results[question].answer != messages.OUT_OF_SCOPE
