"""Context dependence: answers must follow the sources, not the model's memory.

Counterfactuals edit a retrieved passage (e.g. DBS's $160 target becomes $153) and require
the answer to report the edited value. Ablations remove passages and require the answer to
stop stating what only they said. Both run the real pipeline and LLM; only retrieval is
wrapped. Cases and values live in `cases.yaml`.
"""

import asyncio

import pytest
from cases import Ablation, Counterfactual, EvalCase, default_suite
from doubles import (
    PassageTransform,
    TransformingRetriever,
    ablation_transforms,
    counterfactual_transforms,
)

from chat_app.api.container import Container
from chat_app.config.settings import get_settings
from chat_app.core.interfaces import Retriever
from chat_app.generation.events import FinalEvent

pytestmark = pytest.mark.eval

COUNTERFACTUALS = [(c, cf) for c in default_suite().reference for cf in c.counterfactuals]
ABLATIONS = [(c, ab) for c in default_suite().reference for ab in c.ablations]


def _answer_with(query: str, transforms: list[PassageTransform]) -> tuple[FinalEvent, int]:
    """The final answer with transformed retrieval, and how many passages were touched."""
    doubles: list[TransformingRetriever] = []

    def wrap(inner: Retriever) -> Retriever:
        doubles.append(TransformingRetriever(inner, transforms))
        return doubles[-1]

    async def run() -> FinalEvent:
        container = Container(get_settings())
        try:
            service = await container.build_answer_service(retriever_wrapper=wrap)
            events = [e async for e in service.answer(query)]
        finally:
            await container.close()
        return next(e for e in reversed(events) if isinstance(e, FinalEvent))

    final = asyncio.run(run())
    return final, sum(d.touched for d in doubles)


def _failure(problems: list[str], final: FinalEvent) -> str:
    return f"{problems}\n--- answer ---\n{final.answer}"


@pytest.mark.parametrize(
    ("case", "counterfactual"),
    COUNTERFACTUALS,
    ids=[f"{c.id}-{cf.name}" for c, cf in COUNTERFACTUALS],
)
def test_answer_follows_an_edited_source(
    require_pipeline: None, case: EvalCase, counterfactual: Counterfactual
) -> None:
    """After editing a source, the answer reports the edited fact, not the remembered one."""
    final, touched = _answer_with(case.query, counterfactual_transforms(counterfactual))
    assert touched, "no retrieved passage contained the edited span; the case proves nothing"
    problems = counterfactual.expect.violations(final.answer)
    assert not problems, _failure(problems, final)


@pytest.mark.parametrize(
    ("case", "ablation"), ABLATIONS, ids=[f"{c.id}-{ab.name}" for c, ab in ABLATIONS]
)
def test_answer_drops_facts_whose_source_was_removed(
    require_pipeline: None, case: EvalCase, ablation: Ablation
) -> None:
    """After removing a source, the answer doesn't state what only that source said."""
    final, touched = _answer_with(case.query, ablation_transforms(ablation))
    assert touched, "none of the dropped articles were retrieved; the case proves nothing"
    problems = ablation.expect.violations(final.answer)
    assert not problems, _failure(problems, final)
