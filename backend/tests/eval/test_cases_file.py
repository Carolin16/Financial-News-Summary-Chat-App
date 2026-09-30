"""The eval cases file is valid and consistent with the dataset (no LLM; runs by default)."""

import json

import pytest
import yaml
from cases import AnswerExpectation, EvalSuite, ExpectedOutcome, default_suite, load_suite
from pydantic import ValidationError

from chat_app.config.settings import get_settings


@pytest.fixture(scope="module")
def dataset_links() -> set[str]:
    """Every link in the raw dataset."""
    with get_settings().data_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {entry["link"] for entries in raw.values() for entry in entries}


def test_cases_file_loads_with_the_reference_queries():
    """The file parses, and holds the 12 reference queries plus the scope checks."""
    suite = default_suite()
    assert [c.id for c in suite.reference] == [f"q{n}" for n in range(1, 13)]
    assert suite.out_of_scope and suite.borderline


def test_every_gold_link_exists_in_the_dataset(dataset_links):
    """Gold links must name real articles, never invented URLs."""
    unknown = [
        (c.id, s.link)
        for c in default_suite().all_cases
        for s in c.expected_source_links
        if s.link not in dataset_links
    ]
    assert not unknown, unknown


def test_abstain_cases_expect_no_sources():
    """A case that should abstain can't also expect articles to be used."""
    inconsistent = [
        c.id
        for c in default_suite().all_cases
        if c.expected_outcome is ExpectedOutcome.ABSTAIN and c.expected_source_links
    ]
    assert not inconsistent, inconsistent


def test_scope_groups_have_the_expected_outcomes():
    """Out-of-scope questions abstain; borderline business questions are answered."""
    suite = default_suite()
    assert all(c.expected_outcome is ExpectedOutcome.ABSTAIN for c in suite.out_of_scope)
    assert all(c.expected_outcome is ExpectedOutcome.ANSWER for c in suite.borderline)


def _suite_yaml(**overrides: object) -> dict[str, object]:
    case = {"id": "a", "query": "q?", "expected_outcome": "answer", **overrides}
    return {"reference": [case], "out_of_scope": [], "borderline": []}


def test_reviewed_links_exclude_unreviewed_ones():
    """Only confirmed gold links are used for gating."""
    data = _suite_yaml(
        expected_source_links=[
            {"link": "https://x/1", "needs_review": False},
            {"link": "https://x/2"},
        ]
    )
    case = EvalSuite.model_validate(data).reference[0]
    assert case.reviewed_links == ["https://x/1"]


@pytest.mark.parametrize(
    "bad",
    [
        {"expected_outcome": "maybe"},
        {"should_refuse": True},
        {"expected_source_links": [{"link": "https://x/1", "reviewed": True}]},
    ],
)
def test_invalid_cases_are_rejected(bad):
    """Unknown outcomes and unknown fields fail loudly instead of being ignored."""
    with pytest.raises(ValidationError):
        EvalSuite.model_validate(_suite_yaml(**bad))


def test_duplicate_ids_are_rejected(tmp_path):
    """Case ids key the results, so they must be unique."""
    data = _suite_yaml()
    data["borderline"] = [{"id": "a", "query": "other?", "expected_outcome": "answer"}]
    path = tmp_path / "cases.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate case ids"):
        load_suite(path)


def test_adversarial_cases_are_loaded():
    """The adversarial group is part of the suite and of every-case checks."""
    suite = default_suite()
    assert suite.adversarial
    assert {c.id for c in suite.adversarial} <= {c.id for c in suite.all_cases}
    assert suite.group_of(suite.adversarial[0].id) == "adversarial"


@pytest.mark.parametrize(
    ("answer", "ok"),
    [("Only previews exist.", True), ("It was a preview.", True), ("It beat.", False)],
)
def test_include_any_terms_needs_one_match(answer, ok):
    """Any one of the listed wordings satisfies the expectation."""
    expect = AnswerExpectation(include_any_terms=["preview", "previews"])
    assert (not expect.violations(answer)) is ok


def test_citation_violations_match_title_prefixes():
    """Only cited titles starting with an excluded prefix are reported."""
    case = EvalSuite.model_validate(_suite_yaml(excluded_citation_titles=["Demi Moore"])).reference[
        0
    ]
    violations = case.citation_violations(["Demi Moore Puts Tailored Spin", "Netflix earnings"])
    assert violations == ["cited excluded article 'Demi Moore Puts Tailored Spin'"]
    assert EvalSuite.model_validate(_suite_yaml()).reference[0].citation_violations(["x"]) == []
