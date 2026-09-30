"""Recall@k maths and the recall-k setting (no index or LLM; runs by default)."""

import pytest
from cases import EvalCase, ExpectedOutcome, GoldSource
from metrics import RecallSummary, recall_at_k

from chat_app.config.settings import Settings


def case(case_id: str, reviewed: list[str], unreviewed: list[str]) -> EvalCase:
    sources = [GoldSource(link=link, needs_review=False) for link in reviewed]
    sources += [GoldSource(link=link) for link in unreviewed]
    return EvalCase(
        id=case_id,
        query="q?",
        expected_outcome=ExpectedOutcome.ANSWER,
        expected_source_links=sources,
    )


class TestRecallAtK:
    def test_counts_gold_articles_found_in_the_top_k(self):
        assert recall_at_k(["a", "b", "c", "d"], ["a", "d", "z"], k=4) == pytest.approx(2 / 3)

    def test_only_the_first_k_passages_count(self):
        assert recall_at_k(["a", "b", "c", "d"], ["d"], k=3) == 0.0

    def test_several_passages_from_one_article_count_once(self):
        assert recall_at_k(["a", "a", "a", "b"], ["a", "b"], k=4) == 1.0

    def test_no_gold_means_no_score(self):
        assert recall_at_k(["a"], [], k=5) is None


class TestRecallSummary:
    def test_reports_all_links_but_scores_reviewed_separately(self):
        cases = [case("q1", reviewed=["a"], unreviewed=["b"]), case("q2", [], ["c"])]
        ranked = {"q1": ["a", "x"], "q2": ["c"]}
        summary = RecallSummary.compute(cases, ranked, k=2)
        q1, q2 = summary.cases
        assert (q1.recall_all, q1.recall_reviewed, q1.missed) == (0.5, 1.0, ["b"])
        assert (q2.recall_all, q2.recall_reviewed) == (1.0, None)
        assert summary.mean_all == pytest.approx(0.75)
        assert summary.mean_reviewed == 1.0
        assert summary.unreviewed_cases == ["q1", "q2"]

    def test_cases_without_gold_are_skipped(self):
        summary = RecallSummary.compute([case("q12", [], [])], {"q12": ["a"]}, k=5)
        assert summary.cases == [] and summary.mean_all is None

    @pytest.mark.parametrize(("minimum", "passes"), [(0.5, True), (0.9, False)])
    def test_gate_uses_reviewed_links_only(self, minimum, passes):
        cases = [case("q1", reviewed=["a", "b"], unreviewed=["c", "d", "e"])]
        summary = RecallSummary.compute(cases, {"q1": ["a", "c", "d", "e"]}, k=4)
        assert summary.mean_all == pytest.approx(0.8)
        assert summary.mean_reviewed == 0.5
        assert summary.meets(minimum) is passes

    def test_nothing_reviewed_passes_the_gate(self):
        summary = RecallSummary.compute([case("q1", [], ["a"])], {"q1": []}, k=3)
        assert summary.mean_reviewed is None and summary.meets(1.0)


class TestRecallKSetting:
    def test_defaults_to_the_live_retrieval_depth(self):
        assert Settings(retrieval_top_k=6).recall_k == 6

    def test_override_wins(self):
        assert Settings(retrieval_top_k=6, eval_recall_k=12).recall_k == 12

    def test_non_positive_override_is_rejected(self):
        with pytest.raises(ValueError):
            Settings(eval_recall_k=0)
