"""Typed loader for the eval cases file (`cases.yaml`).

Keeping the queries and their expectations in data, not code, lets new cases, gold sources
and required views be added without touching the tests.
"""

import re
from enum import StrEnum
from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chat_app.config.settings import get_settings
from chat_app.generation.grounding import extract_quantities


class ExpectedOutcome(StrEnum):
    """How the bot should respond to a case."""

    ANSWER = "answer"
    # Summarise the news, then end with the not-advice disclaimer (advice/prediction).
    ANSWER_WITH_DISCLAIMER = "answer_with_disclaimer"
    # Refusal or no-data notice and no cited content (live figures, dates, out of scope).
    ABSTAIN = "abstain"


class GoldSource(BaseModel):
    """An article the answer should draw on; unreviewed ones are reported but not gated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    link: str
    needs_review: bool = True
    note: str = ""


def span_pattern(span: str) -> re.Pattern[str]:
    """Match `span` only as a whole token, case-insensitively for words.

    "Ford" must not hit "afford", and "$160" must not hit "$160.91" or "$1600".
    """
    prefix = r"(?<!\w)" if span[0].isalnum() else ""
    suffix = r"(?!\w|\.\d)"
    return re.compile(prefix + re.escape(span) + suffix, re.IGNORECASE)


class Replacement(BaseModel):
    """A span to swap in every retrieved passage (title, header and body)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    old: str = Field(min_length=1)
    new: str


class AnswerExpectation(BaseModel):
    """What an answer must and must not contain after a source is edited or removed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    include_terms: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    # At least one must appear: lets a case accept any of several wordings for one idea.
    include_any_terms: list[str] = Field(default_factory=list)
    # Figures compare by value, so "$153" matches "$153.00" but not "$1,530".
    include_figures: list[str] = Field(default_factory=list)
    exclude_figures: list[str] = Field(default_factory=list)

    def violations(self, answer: str) -> list[str]:
        """Each way `answer` breaks this expectation; empty when it's met."""
        found = extract_quantities(answer)
        problems = [
            f"missing term {t!r}" for t in self.include_terms if not span_pattern(t).search(answer)
        ]
        problems += [
            f"forbidden term {t!r}" for t in self.exclude_terms if span_pattern(t).search(answer)
        ]
        if self.include_any_terms and not any(
            span_pattern(t).search(answer) for t in self.include_any_terms
        ):
            problems.append(f"none of {self.include_any_terms!r}")
        problems += [
            f"missing figure {f}"
            for f in self.include_figures
            if not extract_quantities(f) <= found
        ]
        problems += [
            f"forbidden figure {f}" for f in self.exclude_figures if extract_quantities(f) & found
        ]
        return problems


class Counterfactual(BaseModel):
    """Edit the retrieved sources; a grounded answer must follow the edit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    replacements: list[Replacement] = Field(min_length=1)
    expect: AnswerExpectation


class Ablation(BaseModel):
    """Remove sources; a grounded answer must not state what only they said."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    drop_links: list[str] = Field(min_length=1)
    expect: AnswerExpectation


class EvalCase(BaseModel):
    """One query and what a correct response looks like."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    query: str
    expected_outcome: ExpectedOutcome
    expected_source_links: list[GoldSource] = Field(default_factory=list)
    required_views: list[str] = Field(default_factory=list)
    counterfactuals: list[Counterfactual] = Field(default_factory=list)
    ablations: list[Ablation] = Field(default_factory=list)
    # What the live answer itself must and must not contain.
    expect: AnswerExpectation | None = None
    # Title prefixes of articles the answer must not cite (off-topic entries under a ticker).
    excluded_citation_titles: list[str] = Field(default_factory=list)
    notes: str = ""

    def citation_violations(self, titles: list[str]) -> list[str]:
        """Cited titles that start with one of the excluded prefixes."""
        prefixes = tuple(self.excluded_citation_titles)
        return [
            f"cited excluded article {t!r}" for t in titles if prefixes and t.startswith(prefixes)
        ]

    @property
    def reviewed_links(self) -> list[str]:
        """Gold links a person has confirmed; only these gate retrieval recall."""
        return [s.link for s in self.expected_source_links if not s.needs_review]


class EvalSuite(BaseModel):
    """All cases, grouped by what they test."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: list[EvalCase]
    out_of_scope: list[EvalCase]
    borderline: list[EvalCase]
    # Traps: off-topic entries, false premises, promo numbers, look-alike price targets.
    adversarial: list[EvalCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> "EvalSuite":
        ids = [c.id for c in self.all_cases]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate case ids: {duplicates}")
        return self

    @property
    def all_cases(self) -> list[EvalCase]:
        """Every case in file order."""
        return [*self.reference, *self.out_of_scope, *self.borderline, *self.adversarial]

    def group_of(self, case_id: str) -> str:
        """Which group of the file a case belongs to."""
        groups = {
            "reference": self.reference,
            "out_of_scope": self.out_of_scope,
            "borderline": self.borderline,
            "adversarial": self.adversarial,
        }
        return next(name for name, cases in groups.items() if any(c.id == case_id for c in cases))

    def case(self, case_id: str) -> EvalCase:
        """Any case with this id."""
        return next(c for c in self.all_cases if c.id == case_id)

    def reference_case(self, case_id: str) -> EvalCase:
        """The reference case with this id."""
        return next(c for c in self.reference if c.id == case_id)


def load_suite(path: Path) -> EvalSuite:
    """Parse and validate a cases file."""
    with path.open(encoding="utf-8") as handle:
        return EvalSuite.model_validate(yaml.safe_load(handle))


@cache
def default_suite() -> EvalSuite:
    """The cases file named in settings, loaded once per test run."""
    return load_suite(get_settings().eval_cases_path)
