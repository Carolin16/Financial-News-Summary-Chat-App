"""Checks that every cited sentence in an answer says only what its cited passage says.

An LLM judge reads the sentence and the passage it cites, then answers
"supported" or "unsupported" and quotes the passage sentence it relied on. The judge's
"supported" only counts if that quote really appears in the passage, so the judge can't
talk its way into a pass.

Two kinds of test:
- Faithfulness: each sentence is judged several times and the majority wins. The mean
  share of supported sentences across all answers must reach the threshold, and no single
  answer may fall below the lower floor.
- Canaries: hand-written sentences with a known right answer, to prove the judge itself
  works (it rejects true facts the source doesn't state, and accepts one the source does).
"""

import asyncio
from collections.abc import Callable

import pytest

from chat_app.config.settings import Settings
from chat_app.core.models import Chunk
from chat_app.generation.claim_judge import (
    ClaimJudge,
    ClaimJudgment,
    ClaimVerdict,
    FaithfulnessReport,
    FaithfulnessSummary,
    JudgmentReason,
)
from chat_app.generation.events import FinalEvent, MetaEvent
from chat_app.ingestion.sentences import split_sentences

pytestmark = pytest.mark.eval

# The article every canary is judged against. It is about the Broadcom/TSMC reports and never
# mentions Intel's market cap or founders, so claims about those can't be supported by it.
CANARY_TITLE_PREFIX = "Intel stock surges on report of Broadcom, TSMC"
# True in the real world but absent from the canary article. Each entry is
# (sentence to judge, a word that proves the article doesn't state it).
UNSOURCED_CLAIMS = {
    "market cap": ("Intel's market capitalization is about $100 billion [1].", "market cap"),
    "founders": ("Intel was co-founded by Gordon Moore and Robert Noyce [1].", "Noyce"),
}
NEWLINE = "\n"


@pytest.fixture(scope="session")
def faithfulness(
    results: dict[str, tuple[MetaEvent, FinalEvent]],
    chunks_by_id: dict[str, Chunk],
    claim_judge_factory: Callable[[], ClaimJudge],
) -> dict[str, FaithfulnessReport]:
    """Judge every reference answer once and keep the reports for all tests below."""

    async def judge_all() -> dict[str, FaithfulnessReport]:
        judge = claim_judge_factory()

        async def one(final: FinalEvent) -> FaithfulnessReport:
            # Give the judge the exact passage behind each [n], and the answer's fixed
            # notices so they're skipped rather than judged as claims.
            sources = {
                c.number: chunks_by_id[c.chunk_id].contextualized_text
                for c in final.citations
                if c.chunk_id in chunks_by_id
            }
            return await judge.judge_answer(final.answer, sources, notices=final.notices)

        reports = await asyncio.gather(*(one(final) for _, final in results.values()))
        return dict(zip(results, reports, strict=True))

    return asyncio.run(judge_all())


@pytest.fixture(scope="session")
def faithfulness_summary(
    faithfulness: dict[str, FaithfulnessReport], settings: Settings
) -> FaithfulnessSummary:
    """Suite-wide claim support against the configured threshold and floor."""
    return FaithfulnessSummary.from_reports(
        faithfulness, settings.claim_support_threshold, settings.claim_support_floor
    )


def _rejections(report: FaithfulnessReport) -> str:
    """One line per rejected sentence, with its reason and vote agreement."""
    return NEWLINE.join(
        f"  - [{j.reason}, agreement {j.agreement:.2f}] {j.sentence}" for j in report.unsupported
    )


def test_suite_mean_claim_support_meets_threshold(
    faithfulness: dict[str, FaithfulnessReport], faithfulness_summary: FaithfulnessSummary
):
    """The mean share of supported sentences across all answers reaches the threshold."""
    summary = faithfulness_summary
    details = NEWLINE.join(
        f"{case}: {score:.2f}{NEWLINE}{_rejections(faithfulness[case])}"
        for case, score in summary.scores.items()
        if faithfulness[case].unsupported
    )
    assert summary.mean_meets_threshold, (
        f"mean claim support {summary.mean:.2f} < {summary.threshold}{NEWLINE}{details}"
    )


def test_no_case_falls_below_the_floor(
    faithfulness: dict[str, FaithfulnessReport], settings: Settings, case_id
):
    """No single answer's claim support drops below the floor (runs per query)."""
    report = faithfulness[case_id]
    # Answers such as the "yesterday" coverage list cite nothing, so there's nothing to score.
    if report.score is None:
        pytest.skip("no cited content sentences to judge")
    assert report.meets(settings.claim_support_floor), (
        f"{case_id} claim support {report.score:.2f} < floor "
        f"{settings.claim_support_floor}:{NEWLINE}{_rejections(report)}"
    )


def test_print_faithfulness_scorecard(
    faithfulness: dict[str, FaithfulnessReport],
    faithfulness_summary: FaithfulnessSummary,
    settings: Settings,
    capsys,
):
    """Print per-query scores, vote agreement and rejected sentences, even when all pass."""
    summary = faithfulness_summary
    rows = [f"{'case':<6}{'score':>7}{'agree':>7}{'supported':>11}{'judged':>8}{'skipped':>9}"]
    for case_id, report in faithfulness.items():
        score = "n/a" if report.score is None else f"{report.score:.2f}"
        agreement = "n/a" if report.agreement is None else f"{report.agreement:.2f}"
        rows.append(
            f"{case_id:<6}{score:>7}{agreement:>7}{report.supported:>11}"
            f"{len(report.judgments):>8}{len(report.skipped):>9}"
        )
        if report.unsupported:
            rows.append(_rejections(report))
    mean = "n/a" if summary.mean is None else f"{summary.mean:.2f}"
    rows.append(
        f"mean {mean} (threshold {summary.threshold}), floor {summary.floor}, "
        f"below floor: {summary.below_floor or 'none'}, "
        f"votes per sentence: {settings.claim_judge_votes}"
    )
    # Bypass pytest's output capture so the table shows without `-s`.
    with capsys.disabled():
        print(f"{NEWLINE}Claim support:")
        print(NEWLINE.join(rows))


@pytest.fixture(scope="module")
def canary_source(article_text_by_link: dict[str, str]) -> str:
    """The cleaned text of the canary article, found by its title."""
    matches = [t for t in article_text_by_link.values() if t.startswith(CANARY_TITLE_PREFIX)]
    assert matches, f"canary article not found: {CANARY_TITLE_PREFIX!r}"
    return matches[0]


def _judge(factory: Callable[[], ClaimJudge], sentence: str, source: str) -> ClaimJudgment:
    """Judge one sentence against one source with a fresh judge, failing if any call errors."""

    async def run() -> ClaimJudgment:
        return await factory().judge_sentence(sentence, [source])

    judgment = asyncio.run(run())
    # A failed API call is also recorded as "unsupported". Without this check, a broken
    # judge would look like it correctly rejected the canary claims.
    assert all(v.reason is not JudgmentReason.JUDGE_ERROR for v in judgment.votes), (
        "judge call failed"
    )
    return judgment


@pytest.mark.parametrize("claim", list(UNSOURCED_CLAIMS))
def test_canary_true_world_fact_not_in_source_is_unsupported(
    require_api, claim_judge_factory, canary_source, claim
):
    """A true fact the source doesn't state (e.g. Intel's founders) is judged unsupported."""
    sentence, absent_marker = UNSOURCED_CLAIMS[claim]
    # Guard the test itself: if the article did state the fact, "unsupported" would be wrong.
    assert absent_marker.lower() not in canary_source.lower(), "canary source states the fact"
    judgment = _judge(claim_judge_factory, sentence, canary_source)
    assert judgment.verdict is ClaimVerdict.UNSUPPORTED


def test_canary_sentence_from_the_source_is_supported(
    require_api, claim_judge_factory, canary_source
):
    """A sentence copied straight from the source is judged supported.

    This is the control for the canaries above: a judge that rejected everything would pass
    those, but not this one.
    """
    sentence = next(s for s in split_sentences(canary_source) if "Broadcom" in s)
    judgment = _judge(claim_judge_factory, f"{sentence} [1]", canary_source)
    assert judgment.verdict is ClaimVerdict.SUPPORTED, judgment
