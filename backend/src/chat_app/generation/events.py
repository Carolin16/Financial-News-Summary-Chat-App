"""Events streamed to the client while an answer is produced (sent as SSE)."""

from typing import Literal

from pydantic import BaseModel

from chat_app.retrieval.coverage import CoverageReport


class NumberedCitation(BaseModel):
    """A source as the user sees it; `number` matches the [n] markers in the answer."""

    number: int
    title: str
    link: str
    is_partial: bool


class MetaEvent(BaseModel):
    """How the question was understood; lets the UI explain the answer's framing."""

    type: Literal["meta"] = "meta"
    intent: str
    tickers: list[str]
    coverage: list[CoverageReport]


class SourcesEvent(BaseModel):
    """Candidate sources, sent before generation so links can render early."""

    type: Literal["sources"] = "sources"
    sources: list[NumberedCitation]


class DeltaEvent(BaseModel):
    """A line of the answer that has already passed grounding verification."""

    type: Literal["delta"] = "delta"
    text: str


class FinalEvent(BaseModel):
    """The complete answer with fixed notices; replaces the streamed lines."""

    type: Literal["final"] = "final"
    answer: str
    citations: list[NumberedCitation]
    notices: list[str]
    removed_sentences: int = 0


class ErrorEvent(BaseModel):
    """A failure the client should show; a FinalEvent with a fallback answer follows."""

    type: Literal["error"] = "error"
    message: str


AnswerEvent = MetaEvent | SourcesEvent | DeltaEvent | FinalEvent | ErrorEvent
