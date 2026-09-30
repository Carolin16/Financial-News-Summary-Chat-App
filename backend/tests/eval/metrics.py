"""Retrieval recall@k against gold articles, computed before any generation.

Recall bounds everything downstream: if the right article was never retrieved, a poor
answer is a retrieval failure, not a generation one. k counts retrieved passages (what the
LLM is shown), and recall counts gold *articles* found among them.
"""

from collections.abc import Collection, Mapping, Sequence
from statistics import fmean

from cases import EvalCase
from pydantic import BaseModel


def recall_at_k(ranked_links: Sequence[str], gold: Collection[str], k: int) -> float | None:
    """Share of gold articles among the first `k` retrieved passages; None without gold."""
    wanted = set(gold)
    if not wanted:
        return None
    return len(wanted & set(ranked_links[:k])) / len(wanted)


class CaseRecall(BaseModel):
    """Recall for one case, over all gold links and over reviewed ones only."""

    case_id: str
    recall_all: float
    recall_reviewed: float | None
    missed: list[str]
    unreviewed: int


class RecallSummary(BaseModel):
    """Recall across cases. Only reviewed gold links gate; all are reported."""

    k: int
    cases: list[CaseRecall]

    @classmethod
    def compute(
        cls, cases: Sequence[EvalCase], ranked_links: Mapping[str, Sequence[str]], k: int
    ) -> "RecallSummary":
        """Score every case that has gold links; cases without gold are skipped."""
        rows = []
        for case in cases:
            gold = [s.link for s in case.expected_source_links]
            recall = recall_at_k(ranked_links[case.id], gold, k)
            if recall is None:
                continue
            top = set(ranked_links[case.id][:k])
            rows.append(
                CaseRecall(
                    case_id=case.id,
                    recall_all=recall,
                    recall_reviewed=recall_at_k(ranked_links[case.id], case.reviewed_links, k),
                    missed=[link for link in gold if link not in top],
                    unreviewed=len(gold) - len(case.reviewed_links),
                )
            )
        return cls(k=k, cases=rows)

    @property
    def mean_all(self) -> float | None:
        """Mean recall over all gold links, reviewed or not (reported, never gated)."""
        return fmean(c.recall_all for c in self.cases) if self.cases else None

    @property
    def mean_reviewed(self) -> float | None:
        """Mean recall over reviewed gold links; None until some are reviewed."""
        scored = [c.recall_reviewed for c in self.cases if c.recall_reviewed is not None]
        return fmean(scored) if scored else None

    @property
    def unreviewed_cases(self) -> list[str]:
        """Cases whose gold links still need a person to confirm them."""
        return [c.case_id for c in self.cases if c.unreviewed]

    def meets(self, minimum: float) -> bool:
        """True if reviewed recall reaches `minimum` (nothing reviewed yet also passes)."""
        return self.mean_reviewed is None or self.mean_reviewed >= minimum
