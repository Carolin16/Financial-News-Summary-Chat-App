"""Retrieval recall@k: were the gold articles retrieved? Runs retrieval only, no generation.

k defaults to the live retrieval depth (`RETRIEVAL_TOP_K`), overridable with `EVAL_RECALL_K`.
Recall is reported over every gold link, but only links a person has reviewed gate the
suite; cases with unreviewed links are listed as a warning.
"""

import asyncio
import warnings

import pytest
from cases import EvalCase, EvalSuite
from metrics import RecallSummary

from chat_app.api.container import Container
from chat_app.config.settings import Settings
from chat_app.generation.query_analysis import QueryAnalyzer

pytestmark = pytest.mark.eval

NEWLINE = "\n"


@pytest.fixture(scope="session")
def ranked_links(
    require_pipeline: None, suite: EvalSuite, settings: Settings
) -> dict[str, list[str]]:
    """Retrieved passage links in rank order for each reference case, at depth k."""

    async def collect() -> dict[str, list[str]]:
        container = Container(settings)
        retrieval = container.build_retrieval(top_k=settings.recall_k)
        analyzer = QueryAnalyzer(container.registry)

        async def one(case: EvalCase) -> list[str]:
            # Same routing as serving: the analyzer picks the tickers and intent.
            plan = analyzer.analyze(case.query)
            retrieved = await retrieval.fetch(case.query, plan.tickers, plan.intent)
            return [r.chunk.link for r in retrieved]

        try:
            links = await asyncio.gather(*(one(c) for c in suite.reference))
        finally:
            await container.close()
        return {c.id: ranked for c, ranked in zip(suite.reference, links, strict=True)}

    return asyncio.run(collect())


@pytest.fixture(scope="session")
def recall_summary(
    suite: EvalSuite, ranked_links: dict[str, list[str]], settings: Settings
) -> RecallSummary:
    """Recall for every reference case that has gold links."""
    return RecallSummary.compute(suite.reference, ranked_links, settings.recall_k)


def test_print_recall_scorecard(recall_summary: RecallSummary, settings: Settings, capsys):
    """Print recall per case (all links and reviewed only), and warn about unreviewed ones."""
    summary = recall_summary
    rows = [f"{'case':<6}{'all':>6}{'reviewed':>10}{'unreviewed':>12}"]
    for case in summary.cases:
        reviewed = "n/a" if case.recall_reviewed is None else f"{case.recall_reviewed:.2f}"
        rows.append(f"{case.case_id:<6}{case.recall_all:>6.2f}{reviewed:>10}{case.unreviewed:>12}")
        rows.extend(f"  - missed {link}" for link in case.missed)
    mean_all = "n/a" if summary.mean_all is None else f"{summary.mean_all:.2f}"
    mean_reviewed = "n/a" if summary.mean_reviewed is None else f"{summary.mean_reviewed:.2f}"
    rows.append(
        f"k={summary.k}  mean (all) {mean_all}  mean (reviewed) {mean_reviewed}  "
        f"minimum {settings.eval_min_mean_recall}"
    )
    # Bypass pytest's output capture so the table shows without `-s`.
    with capsys.disabled():
        print(f"{NEWLINE}Retrieval recall@{summary.k}:")
        print(NEWLINE.join(rows))
    if summary.unreviewed_cases:
        warnings.warn(
            "gold links still need review (reported, not gated): "
            + ", ".join(summary.unreviewed_cases),
            stacklevel=1,
        )


def test_mean_reviewed_recall_meets_minimum(recall_summary: RecallSummary, settings: Settings):
    """Mean recall over reviewed gold links reaches the configured minimum."""
    summary = recall_summary
    if summary.mean_reviewed is None:
        pytest.skip(
            "no reviewed gold links yet; unreviewed cases: " + ", ".join(summary.unreviewed_cases)
        )
    misses = NEWLINE.join(
        f"{c.case_id}: {c.recall_reviewed:.2f}"
        for c in summary.cases
        if c.recall_reviewed is not None and c.recall_reviewed < 1
    )
    assert summary.meets(settings.eval_min_mean_recall), (
        f"mean reviewed recall@{summary.k} {summary.mean_reviewed:.2f} < "
        f"{settings.eval_min_mean_recall}{NEWLINE}{misses}"
    )
