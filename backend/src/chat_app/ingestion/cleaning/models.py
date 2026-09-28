"""Inputs and outputs of the cleaning pipeline."""

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class CleaningContext:
    """Article facts a step may need besides the text itself."""

    title: str


class StepResult(BaseModel):
    """What one step produced and the signals it observed."""

    text: str
    chars_removed_by_rule: dict[str, int] = Field(default_factory=dict)
    truncation_markers: list[str] = Field(default_factory=list)
    footers_kept: list[str] = Field(default_factory=list)


class CleaningSignals(BaseModel):
    """Observations from cleaning, reported for downstream decisions (e.g. stub detection).

    The cleaner only reports; it never decides whether an article is a stub.
    """

    truncation_markers: list[str] = Field(
        default_factory=list, description="Names of truncation rules that fired."
    )
    chars_removed_by_step: dict[str, int] = Field(
        default_factory=dict,
        description="Net characters removed per step (negative if a step expanded text, "
        "e.g. an ellipsis folded to three dots).",
    )
    chars_removed_by_rule: dict[str, int] = Field(default_factory=dict)
    footers_kept: list[str] = Field(
        default_factory=list,
        description="Footer rules whose tail was kept because it carried article content.",
    )

    @property
    def truncation_marker_found(self) -> bool:
        """True if the text showed signs of being a teaser or paywalled excerpt."""
        return bool(self.truncation_markers)


class CleanedArticle(BaseModel):
    """A source entry with cleaned title and text plus what cleaning observed."""

    title: str
    link: str
    ticker: str
    text: str
    signals: CleaningSignals
