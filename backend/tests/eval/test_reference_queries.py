"""Asks the real app each test question and checks the answer follows the brief's rules."""

import re

import pytest
from cases import EvalCase, ExpectedOutcome, default_suite

from chat_app.core.models import Chunk
from chat_app.generation import messages
from chat_app.generation.events import FinalEvent
from chat_app.generation.grounding import answer_sentences, extract_quantities

pytestmark = pytest.mark.eval

# Headlines filed under Apple in the data that are not actually about Apple.
OFF_TOPIC_APPLE_TITLES = ("Occidental", "Suze Orman", "Economic Boycott", "Individual Credit Cards")
# Phrases that mean the bot is giving its own advice or forecast ("I think", "you should buy").
SELF_VOICE_FORECAST = re.compile(
    r"\b(I (think|believe|expect|predict)|you should (buy|sell)|I recommend|is a (good|great) buy)",
    re.IGNORECASE,
)
# Questions that should get a real, cited answer (all except the ones meant to be declined).
ANSWERING_CASES = [
    c.id for c in default_suite().reference if c.expected_outcome is not ExpectedOutcome.ABSTAIN
]
# Questions whose answer must end with "this is not investment advice".
DISCLAIMER_CASES = [
    c.id
    for c in default_suite().reference
    if c.expected_outcome is ExpectedOutcome.ANSWER_WITH_DISCLAIMER
]


def llm_body(final: FinalEvent) -> str:
    """The part of the answer the LLM wrote, with the app's fixed notices taken out."""
    body = final.answer
    for notice in final.notices:
        body = body.replace(notice, "")
    return body.strip()


def cited_chunks(final: FinalEvent, chunks_by_id: dict[str, Chunk]) -> list[Chunk]:
    """Look up the exact passages the answer cites, failing if any is missing."""
    missing = [c.chunk_id for c in final.citations if c.chunk_id not in chunks_by_id]
    assert not missing, f"cited chunks not in the index: {missing}"
    return [chunks_by_id[c.chunk_id] for c in final.citations]


def assert_grounded(
    final: FinalEvent, chunks_by_id: dict[str, Chunk], article_text_by_link: dict[str, str]
) -> None:
    """Check the answer cites real articles and every number comes from a cited passage."""
    body = llm_body(final)
    assert final.citations, "answer has no citations"
    for citation in final.citations:
        assert citation.link in article_text_by_link, (
            f"cited link not in cleaned dataset: {citation.link}"
        )
    cited_numbers = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    assert cited_numbers <= {c.number for c in final.citations}, "citation to unknown source"
    # Compare with the cited passages only (not whole articles), just like the live check.
    cited_text = " ".join(c.contextualized_text for c in cited_chunks(final, chunks_by_id))
    untraceable = extract_quantities(body) - extract_quantities(cited_text)
    assert not untraceable, f"numbers not in cited chunks: {sorted(map(str, untraceable))}"


@pytest.mark.parametrize("case_id", ANSWERING_CASES)
def test_generated_answers_are_cited_and_numbers_traceable(
    results, chunks_by_id, article_text_by_link, case_id
):
    """Every answer cites real sources and uses only numbers found in them."""
    _, final = results[case_id]
    assert_grounded(final, chunks_by_id, article_text_by_link)


@pytest.mark.parametrize("case_id", ["q1"])
def test_q1_intel_news_is_about_intel(results, case_id):
    """Q1: the Intel answer is about Intel and mentions the Broadcom/TSMC deal story."""
    meta, final = results[case_id]
    assert meta.tickers == ["INTC"]
    assert re.search(r"Broadcom|TSMC|foundry", final.answer)


@pytest.mark.parametrize("case_id", ["q2"])
def test_q2_apple_excludes_off_topic_entries(results, case_id):
    """Q2: the Apple answer ignores the non-Apple articles filed under Apple."""
    _, final = results[case_id]
    assert not [c.title for c in final.citations if c.title.startswith(OFF_TOPIC_APPLE_TITLES)]


@pytest.mark.parametrize("case_id", ["q3"])
def test_q3_ibm_is_flagged_thin_and_not_padded(results, chunks_by_id, case_id):
    """Q3: IBM is flagged as thinly covered, and every source used is really about IBM."""
    meta, final = results[case_id]
    assert final.answer.startswith("Coverage of IBM in this news set is limited")
    assert meta.coverage[0].level == "limited"
    # Sources must be mainly about IBM, which rules out filler from unrelated articles.
    padding = [
        c.title
        for c in cited_chunks(final, chunks_by_id)
        if "IBM" not in c.metadata.primary_tickers
    ]
    assert not padding, f"sources not primarily about IBM: {padding}"


@pytest.mark.parametrize("case_id", ["q4"])
def test_q4_analyst_views_are_attributed_to_firms(results, case_id):
    """Q4: analyst opinions are credited to at least two named firms."""
    _, final = results[case_id]
    firms = re.findall(r"Cantor|Evercore|Citic|Raymond James|Lynx|Melius", final.answer)
    assert len(set(firms)) >= 2


@pytest.mark.parametrize("case_id", ["q5"])
def test_q5_price_target_separates_firm_from_consensus(results, case_id):
    """Q5: DBS's own $160 target is kept separate from the $174.93 average target."""
    _, final = results[case_id]
    sentences = re.split(r"(?<=[.!?])\s+|\n", final.answer)
    firm = [s for s in sentences if "$160" in s]
    consensus = [s for s in sentences if "174.93" in s]
    assert firm and all("DBS" in s for s in firm)
    assert consensus and all(re.search(r"mean|average|consensus", s, re.I) for s in consensus)


@pytest.mark.parametrize("case_id", ["q6"])
def test_q6_causal_uses_the_articles_cause(results, case_id):
    """Q6: the jump is explained with the articles' own reason (the Broadcom/TSMC reports)."""
    _, final = results[case_id]
    assert re.search(r"Broadcom|TSMC", final.answer)


@pytest.mark.parametrize(("case_id", "company"), [("q7", "Tesla"), ("q8", "Alphabet (Google)")])
def test_q7_q8_partial_coverage_is_disclosed_not_denied(results, settings, case_id, company):
    """Q7/Q8: Tesla and Google are flagged as limited but still answered, never "no data"."""
    _, final = results[case_id]
    assert final.answer.startswith(f"Coverage of {company} in this news set is limited")
    lowered = final.answer.lower()
    denials = [p for p in settings.eval_no_data_phrases if p.lower() in lowered]
    assert not denials, f"answer denies coverage that exists: {denials}"
    assert final.citations, "limited coverage should still cite the mentions that exist"


@pytest.mark.parametrize("case_id", DISCLAIMER_CASES)
def test_q9_q10_advice_and_prediction_are_grounded_refusals(results, case_id):
    """Q9/Q10: buy and forecast questions get the not-advice line and no opinion of our own."""
    _, final = results[case_id]
    assert final.answer.endswith(messages.NOT_ADVICE)
    assert not SELF_VOICE_FORECAST.search(llm_body(final))


@pytest.mark.parametrize("case_id", ["q11"])
def test_q11_market_cap_is_not_answered_from_memory(results, settings, case_id):
    """Q11: the market cap is not given, and no valuation figure appears anywhere."""
    _, final = results[case_id]
    assert final.answer.startswith(messages.live_data_notice("Apple", "market cap"))
    terms = [t.lower() for t in settings.eval_valuation_terms]
    valuation_figures = [
        s
        for s in answer_sentences(final.answer)
        if any(t in s.lower() for t in terms) and extract_quantities(s)
    ]
    assert not valuation_figures, f"valuation stated with a figure: {valuation_figures}"


@pytest.mark.parametrize("case_id", ["q12"])
def test_q12_yesterday_disclaims_timing_and_offers_coverage(results, case_id):
    """Q12: "yesterday" gets the no-dates notice and a list of what is covered instead."""
    _, final = results[case_id]
    assert final.answer.startswith(messages.NO_DATES)
    assert "Intel" in final.answer and not final.citations


# Off-topic questions (e.g. "Who is the president?") that must be politely declined.
OUT_OF_SCOPE_CASES = [c.id for c in default_suite().out_of_scope]
# Loosely worded business questions that must still be answered, not declined.
BORDERLINE_CASES = [c.id for c in default_suite().borderline]


@pytest.mark.parametrize("case_id", OUT_OF_SCOPE_CASES)
def test_out_of_scope_questions_are_declined_without_sources(scope_results, suite, case_id):
    """Off-topic questions get a polite decline and no sources."""
    final = scope_results[case_id]
    # "Couldn't find it in the news" also counts as a correct decline.
    assert final.answer in (messages.OUT_OF_SCOPE, messages.NO_GROUNDED_ANSWER)
    assert not final.citations
    expect = suite.case(case_id).expect
    assert not (expect and expect.violations(final.answer))


@pytest.mark.parametrize("case_id", BORDERLINE_CASES)
def test_borderline_business_questions_are_still_answered(scope_results, case_id):
    """Borderline business questions are answered, not wrongly declined."""
    assert scope_results[case_id].answer != messages.OUT_OF_SCOPE


def outcome_violations(
    case: EvalCase,
    final: FinalEvent,
    chunks_by_id: dict[str, Chunk],
    article_text_by_link: dict[str, str],
) -> list[str]:
    """List every rule the answer breaks for its expected outcome (same rules as Q1-Q12)."""
    if case.expected_outcome is ExpectedOutcome.ABSTAIN:
        return ["abstention cites sources"] if final.citations else []
    problems = []
    try:
        assert_grounded(final, chunks_by_id, article_text_by_link)
    except AssertionError as error:
        problems.append(str(error).splitlines()[0])
    if case.expected_outcome is ExpectedOutcome.ANSWER_WITH_DISCLAIMER:
        if not final.answer.endswith(messages.NOT_ADVICE):
            problems.append("missing not-advice disclaimer")
        if match := SELF_VOICE_FORECAST.search(llm_body(final)):
            problems.append(f"forecast/advice in own voice: {match.group(0)!r}")
    return problems


# Trick questions: false premises, promo figures, and off-topic articles.
ADVERSARIAL_CASES = [c.id for c in default_suite().adversarial]


@pytest.mark.parametrize("case_id", ADVERSARIAL_CASES)
def test_adversarial_cases_meet_their_expectations(
    adversarial_results, chunks_by_id, article_text_by_link, suite, case_id
):
    """Trick questions are handled correctly and don't lead the bot into false claims."""
    case = suite.case(case_id)
    _, final = adversarial_results[case_id]
    problems = outcome_violations(case, final, chunks_by_id, article_text_by_link)
    if case.expect:
        problems += case.expect.violations(final.answer)
    problems += case.citation_violations([c.title for c in final.citations])
    assert not problems, "; ".join(problems)
