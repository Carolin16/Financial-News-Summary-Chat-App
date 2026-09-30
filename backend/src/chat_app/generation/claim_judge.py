"""Checks that each cited sentence says only what its sources say (used by the tests)."""

import asyncio
import logging
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from enum import StrEnum
from statistics import fmean

from pydantic import BaseModel, Field

from chat_app.core.interfaces import AsyncStructuredLlm
from chat_app.core.text_normalization import fold_punctuation
from chat_app.generation import messages
from chat_app.generation.grounding import (
    answer_sentences,
    citation_numbers,
    is_lead_in,
    strip_citations,
)
from chat_app.generation.llm_client import LlmError
from chat_app.generation.prompts import load_prompt

logger = logging.getLogger(__name__)

# Prompt files for the judge, and the line placed between sources when several are cited.
_INSTRUCTIONS_PROMPT = "claim_support"
_INPUT_PROMPT = "claim_support_input"
_SOURCE_SEPARATOR = "\n\n---\n\n"


class ClaimVerdict(StrEnum):
    """Is the sentence backed by its sources or not."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class JudgeResponse(BaseModel):
    """What the judge LLM must return: its verdict and a quote proving it."""

    verdict: ClaimVerdict
    supporting_quote: str = Field(
        description="One sentence copied exactly from the source text; empty if unsupported."
    )


class JudgmentReason(StrEnum):
    """Why a sentence got its final verdict."""

    JUDGE_SUPPORTED = "judge_supported"  # judge said yes and its quote checked out
    JUDGE_UNSUPPORTED = "judge_unsupported"  # judge said the sources don't back it
    QUOTE_NOT_FOUND = "quote_not_found"  # judge said yes, but its quote isn't in the source
    QUOTE_TOO_SHORT = "quote_too_short"  # quote too short to prove anything
    UNKNOWN_CITATION = "unknown_citation"  # cites a source number that doesn't exist
    JUDGE_ERROR = "judge_error"  # the judge call failed


class ClaimVote(BaseModel):
    """One judge call: what the LLM said, and what code decided after checking its quote."""

    proposed: ClaimVerdict | None
    verdict: ClaimVerdict
    supporting_quote: str = ""
    reason: JudgmentReason


class ClaimJudgment(BaseModel):
    """The result for one sentence: all its votes and the majority verdict."""

    sentence: str
    cited: list[int]
    votes: list[ClaimVote] = Field(default_factory=list)
    verdict: ClaimVerdict
    supporting_quote: str = ""
    reason: JudgmentReason

    @property
    def proposed(self) -> ClaimVerdict | None:
        """What the LLM said most often, before code checked the quotes (None if all failed)."""
        proposals = [v.proposed for v in self.votes if v.proposed is not None]
        return Counter(proposals).most_common(1)[0][0] if proposals else None

    @property
    def agreement(self) -> float:
        """How many votes agreed with the final verdict, from 0 to 1."""
        if not self.votes:
            return 1.0
        return sum(v.verdict is self.verdict for v in self.votes) / len(self.votes)


class FaithfulnessReport(BaseModel):
    """How well one answer is backed by its sources."""

    judgments: list[ClaimJudgment]
    skipped: list[str] = Field(
        default_factory=list, description="Intro lines and uncited sentences, not judged."
    )

    @property
    def supported(self) -> int:
        """How many judged sentences are backed by their sources."""
        return sum(j.verdict is ClaimVerdict.SUPPORTED for j in self.judgments)

    @property
    def score(self) -> float | None:
        """Share of judged sentences that are supported (None if there was nothing to judge)."""
        if not self.judgments:
            return None
        return self.supported / len(self.judgments)

    @property
    def agreement(self) -> float | None:
        """How consistent the judge was on average across sentences (None if nothing judged)."""
        if not self.judgments:
            return None
        return fmean(j.agreement for j in self.judgments)

    @property
    def unsupported(self) -> list[ClaimJudgment]:
        """The sentences that are not backed by their sources."""
        return [j for j in self.judgments if j.verdict is ClaimVerdict.UNSUPPORTED]

    def meets(self, threshold: float) -> bool:
        """True if the answer scores at least `threshold` (an answer with no claims passes)."""
        return self.score is None or self.score >= threshold


class FaithfulnessSummary(BaseModel):
    """Scores for the whole test run: the average must pass and no answer may be very bad."""

    scores: dict[str, float]
    mean: float | None
    below_floor: list[str]
    threshold: float
    floor: float

    @classmethod
    def from_reports(
        cls, reports: Mapping[str, FaithfulnessReport], threshold: float, floor: float
    ) -> "FaithfulnessSummary":
        """Combine the per-question reports, leaving out questions with nothing to judge."""
        scores = {case: r.score for case, r in reports.items() if r.score is not None}
        return cls(
            scores=scores,
            mean=fmean(scores.values()) if scores else None,
            below_floor=[case for case, score in scores.items() if score < floor],
            threshold=threshold,
            floor=floor,
        )

    @property
    def mean_meets_threshold(self) -> bool:
        """True if the average score reaches the threshold."""
        return self.mean is None or self.mean >= self.threshold

    @property
    def passes(self) -> bool:
        """True if the average passes and no single answer falls below the floor."""
        return self.mean_meets_threshold and not self.below_floor


class ClaimJudge:
    """Asks an LLM whether each sentence is backed by its sources, then double-checks it in code."""

    def __init__(
        self, llm: AsyncStructuredLlm, min_quote_chars: int, max_concurrency: int, votes: int
    ) -> None:
        """Set the judge LLM, minimum quote length, parallel calls, and votes per sentence (odd)."""
        if votes < 1 or votes % 2 == 0:
            raise ValueError(f"votes must be a positive odd number, got {votes}")
        self._llm = llm
        self._min_quote_chars = min_quote_chars
        self._limit = asyncio.Semaphore(max_concurrency)
        self._votes = votes

    async def judge_answer(
        self, answer: str, sources: Mapping[int, str], notices: Collection[str] = ()
    ) -> FaithfulnessReport:
        """Judge every cited sentence in the answer, skipping app notices and intro lines."""
        for fixed in (*messages.FIXED_MESSAGES, *notices):
            answer = answer.replace(fixed, "")
        to_judge: list[str] = []
        skipped: list[str] = []
        for sentence in answer_sentences(answer):
            if is_lead_in(sentence) or not citation_numbers(sentence):
                skipped.append(sentence)
            else:
                to_judge.append(sentence)
        judgments = await asyncio.gather(*(self._judge_cited(s, sources) for s in to_judge))
        return FaithfulnessReport(judgments=list(judgments), skipped=skipped)

    async def judge_sentence(self, sentence: str, cited_texts: Sequence[str]) -> ClaimJudgment:
        """Ask the judge several times about one sentence and go with the majority."""
        prompt = load_prompt(_INPUT_PROMPT).format(
            sources=_SOURCE_SEPARATOR.join(cited_texts), claim=strip_citations(sentence)
        )
        votes = list(
            await asyncio.gather(*(self._vote(prompt, cited_texts) for _ in range(self._votes)))
        )
        # Supported only if more than half the votes say so.
        supporting = [v for v in votes if v.verdict is ClaimVerdict.SUPPORTED]
        if len(supporting) * 2 > len(votes):
            return ClaimJudgment(
                sentence=sentence,
                cited=citation_numbers(sentence),
                votes=votes,
                verdict=ClaimVerdict.SUPPORTED,
                supporting_quote=supporting[0].supporting_quote,
                reason=JudgmentReason.JUDGE_SUPPORTED,
            )
        # Otherwise report the most common reason for rejecting it.
        rejections = Counter(v.reason for v in votes if v.verdict is ClaimVerdict.UNSUPPORTED)
        return ClaimJudgment(
            sentence=sentence,
            cited=citation_numbers(sentence),
            votes=votes,
            verdict=ClaimVerdict.UNSUPPORTED,
            reason=rejections.most_common(1)[0][0],
        )

    async def _vote(self, prompt: str, cited_texts: Sequence[str]) -> ClaimVote:
        """Ask the judge once, then check its answer in code."""
        try:
            async with self._limit:
                response = await self._llm.aparse(
                    load_prompt(_INSTRUCTIONS_PROMPT), prompt, JudgeResponse
                )
        except LlmError:
            # If the judge fails, we can't confirm support, so count it as unsupported.
            logger.warning("claim judge failed; counting vote as unsupported", exc_info=True)
            return ClaimVote(
                proposed=None, verdict=ClaimVerdict.UNSUPPORTED, reason=JudgmentReason.JUDGE_ERROR
            )
        verdict, reason = self._decide(response, cited_texts)
        return ClaimVote(
            proposed=response.verdict,
            verdict=verdict,
            supporting_quote=response.supporting_quote,
            reason=reason,
        )

    async def _judge_cited(self, sentence: str, sources: Mapping[int, str]) -> ClaimJudgment:
        """Judge a sentence, or fail it straight away if it cites a source that doesn't exist."""
        cited = citation_numbers(sentence)
        if any(n not in sources for n in cited):
            return ClaimJudgment(
                sentence=sentence,
                cited=cited,
                verdict=ClaimVerdict.UNSUPPORTED,
                reason=JudgmentReason.UNKNOWN_CITATION,
            )
        return await self.judge_sentence(sentence, [sources[n] for n in dict.fromkeys(cited)])

    def _decide(
        self, response: JudgeResponse, cited_texts: Sequence[str]
    ) -> tuple[ClaimVerdict, JudgmentReason]:
        """Trust a "supported" verdict only if the quote is long enough and really in a source."""
        if response.verdict is ClaimVerdict.UNSUPPORTED:
            return ClaimVerdict.UNSUPPORTED, JudgmentReason.JUDGE_UNSUPPORTED
        quote = _normalize(response.supporting_quote)
        if len(quote) < self._min_quote_chars:
            return ClaimVerdict.UNSUPPORTED, JudgmentReason.QUOTE_TOO_SHORT
        if not any(quote in _normalize(text) for text in cited_texts):
            return ClaimVerdict.UNSUPPORTED, JudgmentReason.QUOTE_NOT_FOUND
        return ClaimVerdict.SUPPORTED, JudgmentReason.JUDGE_SUPPORTED


def _normalize(text: str) -> str:
    """Tidy punctuation and spaces so the quote and source text can be compared exactly."""
    return " ".join(fold_punctuation(text).split())
