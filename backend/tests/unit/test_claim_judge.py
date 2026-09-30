"""ClaimJudge with a scripted LLM: the judge proposes, code decides (no network)."""

import pytest

from chat_app.config.settings import Settings
from chat_app.generation import messages
from chat_app.generation.claim_judge import (
    ClaimJudge,
    ClaimJudgment,
    ClaimVerdict,
    FaithfulnessReport,
    FaithfulnessSummary,
    JudgeResponse,
    JudgmentReason,
)
from chat_app.generation.llm_client import LlmError

INTEL = (
    "Title: Intel shares jump\n\n"
    "Intel shares rose 16% after reports that Broadcom is weighing a bid for its chip "
    "design unit. TSMC is also said to be studying a stake in Intel's factories."
)
DBS = "Title: DBS cuts target\n\nDBS lowered its Nvidia price target to $160 from $175."
SOURCES = {1: INTEL, 2: DBS}
QUOTE = (
    "Intel shares rose 16% after reports that Broadcom is weighing a bid for its chip design unit."
)
MIN_QUOTE_CHARS = 20


class ScriptedLlm:
    """Returns queued responses (or raises queued errors) and records every prompt."""

    def __init__(self, *responses: JudgeResponse | Exception):
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def aparse(self, instructions, prompt, schema):
        assert schema is JudgeResponse
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def supported(quote: str = QUOTE) -> JudgeResponse:
    return JudgeResponse(verdict=ClaimVerdict.SUPPORTED, supporting_quote=quote)


def judge(*responses: JudgeResponse | Exception) -> tuple[ClaimJudge, ScriptedLlm]:
    llm = ScriptedLlm(*responses)
    return ClaimJudge(llm, min_quote_chars=MIN_QUOTE_CHARS, max_concurrency=2, votes=1), llm


class TestQuoteVerification:
    async def test_supported_with_exact_quote_is_supported(self):
        claim_judge, _ = judge(supported())
        sentence = "Intel rose 16% on a Broadcom report [1]."
        result = await claim_judge.judge_sentence(sentence, [INTEL])
        assert result.verdict is ClaimVerdict.SUPPORTED
        assert result.reason is JudgmentReason.JUDGE_SUPPORTED

    async def test_quote_with_typographic_punctuation_and_line_breaks_still_matches(self):
        text = "Intel said it’s “open to options” for\nthe foundry unit."
        quote = 'Intel said it\'s "open to options" for the foundry unit.'
        claim_judge, _ = judge(supported(quote))
        result = await claim_judge.judge_sentence("Intel is open to options [1].", [text])
        assert result.verdict is ClaimVerdict.SUPPORTED

    @pytest.mark.parametrize(
        ("quote", "reason"),
        [
            ("Intel shares rose 18% after reports that Broadcom is weighing a bid.", "number"),
            ("Intel shares fell 16% after reports that Broadcom is weighing a bid.", "word"),
            ("Intel was co-founded by Gordon Moore and Robert Noyce in 1968.", "absent"),
        ],
    )
    async def test_supported_but_quote_not_in_source_is_unsupported(self, quote, reason):
        claim_judge, _ = judge(supported(quote))
        result = await claim_judge.judge_sentence("Intel rose [1].", [INTEL])
        assert result.proposed is ClaimVerdict.SUPPORTED
        assert result.verdict is ClaimVerdict.UNSUPPORTED, reason
        assert result.reason is JudgmentReason.QUOTE_NOT_FOUND

    @pytest.mark.parametrize("quote", ["", "   ", "Intel shares"])
    async def test_empty_or_trivial_quote_is_unsupported(self, quote):
        claim_judge, _ = judge(supported(quote))
        result = await claim_judge.judge_sentence("Intel rose 16% [1].", [INTEL])
        assert result.verdict is ClaimVerdict.UNSUPPORTED
        assert result.reason is JudgmentReason.QUOTE_TOO_SHORT

    async def test_quote_from_a_source_the_sentence_does_not_cite_is_rejected(self):
        claim_judge, _ = judge(supported("DBS lowered its Nvidia price target to $160 from $175."))
        report = await claim_judge.judge_answer("DBS cut its target to $160 [1].", SOURCES)
        [result] = report.judgments
        assert result.reason is JudgmentReason.QUOTE_NOT_FOUND

    async def test_judge_unsupported_stays_unsupported(self):
        claim_judge, _ = judge(JudgeResponse(verdict=ClaimVerdict.UNSUPPORTED, supporting_quote=""))
        result = await claim_judge.judge_sentence("Intel is worth $100 billion [1].", [INTEL])
        assert result.verdict is ClaimVerdict.UNSUPPORTED
        assert result.reason is JudgmentReason.JUDGE_UNSUPPORTED

    async def test_llm_failure_counts_as_unsupported(self):
        claim_judge, _ = judge(LlmError("timeout"))
        result = await claim_judge.judge_sentence("Intel rose 16% [1].", [INTEL])
        assert result.verdict is ClaimVerdict.UNSUPPORTED
        assert result.reason is JudgmentReason.JUDGE_ERROR
        assert result.proposed is None


class TestJudgeAnswer:
    async def test_notices_lead_ins_and_uncited_sentences_are_not_sent_to_the_judge(self):
        coverage = "Coverage of Tesla in this news set is limited: 1 article focuses on it."
        answer = "\n\n".join(
            [
                coverage,
                "Here's what the articles report:",
                "- Intel rose 16% on a Broadcom report [1].\n- Intel is a storied company.",
                messages.PARTIAL_SOURCES,
                messages.NOT_ADVICE,
            ]
        )
        claim_judge, llm = judge(supported())
        report = await claim_judge.judge_answer(answer, SOURCES, notices=[coverage])
        assert [j.sentence for j in report.judgments] == [
            "Intel rose 16% on a Broadcom report [1]."
        ]
        assert report.skipped == ["Here's what the articles report:", "Intel is a storied company."]
        assert len(llm.prompts) == 1

    async def test_prompt_has_claim_without_markers_and_only_cited_source_text(self):
        claim_judge, llm = judge(supported())
        await claim_judge.judge_answer("Intel rose 16% on a Broadcom report [1][1].", SOURCES)
        [prompt] = llm.prompts
        assert "Intel rose 16% on a Broadcom report." in prompt
        assert "[1]" not in prompt
        assert prompt.count(INTEL) == 1
        assert DBS not in prompt

    async def test_multi_source_sentence_accepts_a_quote_from_either_source(self):
        dbs_quote = "DBS lowered its Nvidia price target to $160 from $175."
        claim_judge, llm = judge(supported(dbs_quote))
        report = await claim_judge.judge_answer("Intel rose while DBS cut Nvidia [1][2].", SOURCES)
        assert report.judgments[0].verdict is ClaimVerdict.SUPPORTED
        assert INTEL in llm.prompts[0] and DBS in llm.prompts[0]

    async def test_citation_to_unknown_source_is_unsupported_without_a_judge_call(self):
        claim_judge, llm = judge()
        report = await claim_judge.judge_answer("Intel rose 16% [7].", SOURCES)
        assert report.judgments[0].reason is JudgmentReason.UNKNOWN_CITATION
        assert llm.prompts == []


def judgment(verdict: ClaimVerdict) -> ClaimJudgment:
    return ClaimJudgment(
        sentence="s [1].",
        cited=[1],
        verdict=verdict,
        reason=JudgmentReason.JUDGE_SUPPORTED,
    )


class TestFaithfulnessScore:
    def test_score_is_supported_over_judged(self):
        verdicts = [ClaimVerdict.SUPPORTED] * 3 + [ClaimVerdict.UNSUPPORTED]
        report = FaithfulnessReport(judgments=[judgment(v) for v in verdicts])
        assert report.score == 0.75
        assert report.supported == 3
        assert len(report.unsupported) == 1

    @pytest.mark.parametrize(("threshold", "passes"), [(0.7, True), (0.75, True), (0.8, False)])
    def test_threshold_comparison(self, threshold, passes):
        verdicts = [ClaimVerdict.SUPPORTED] * 3 + [ClaimVerdict.UNSUPPORTED]
        report = FaithfulnessReport(judgments=[judgment(v) for v in verdicts])
        assert report.meets(threshold) is passes

    def test_answer_with_nothing_to_judge_has_no_score_and_passes(self):
        report = FaithfulnessReport(judgments=[], skipped=["Here's the coverage:"])
        assert report.score is None
        assert report.meets(1.0)


def unsupported() -> JudgeResponse:
    return JudgeResponse(verdict=ClaimVerdict.UNSUPPORTED, supporting_quote="")


def voting_judge(*responses: JudgeResponse | Exception) -> tuple[ClaimJudge, ScriptedLlm]:
    llm = ScriptedLlm(*responses)
    return ClaimJudge(llm, min_quote_chars=MIN_QUOTE_CHARS, max_concurrency=3, votes=3), llm


SENTENCE = "Intel rose 16% on a Broadcom report [1]."
FAKE_QUOTE = "Intel shares rose 18% after reports that Broadcom is weighing a bid."


class TestVoting:
    async def test_all_votes_agree(self):
        claim_judge, llm = voting_judge(supported(), supported(), supported())
        result = await claim_judge.judge_sentence(SENTENCE, [INTEL])
        assert result.verdict is ClaimVerdict.SUPPORTED
        assert result.agreement == 1.0
        assert len(result.votes) == len(llm.prompts) == 3

    async def test_split_vote_majority_supported(self):
        claim_judge, _ = voting_judge(supported(), unsupported(), supported())
        result = await claim_judge.judge_sentence(SENTENCE, [INTEL])
        assert result.verdict is ClaimVerdict.SUPPORTED
        assert result.supporting_quote == QUOTE
        assert result.agreement == pytest.approx(2 / 3)

    async def test_split_vote_majority_unsupported(self):
        claim_judge, _ = voting_judge(unsupported(), supported(), unsupported())
        result = await claim_judge.judge_sentence(SENTENCE, [INTEL])
        assert result.verdict is ClaimVerdict.UNSUPPORTED
        assert result.reason is JudgmentReason.JUDGE_UNSUPPORTED
        assert result.supporting_quote == ""
        assert result.agreement == pytest.approx(2 / 3)

    async def test_each_vote_is_quote_checked_before_counting(self):
        # The judge says "supported" three times, but two quotes are fabricated.
        claim_judge, _ = voting_judge(supported(), supported(FAKE_QUOTE), supported(FAKE_QUOTE))
        result = await claim_judge.judge_sentence(SENTENCE, [INTEL])
        assert result.proposed is ClaimVerdict.SUPPORTED
        assert result.verdict is ClaimVerdict.UNSUPPORTED
        assert result.reason is JudgmentReason.QUOTE_NOT_FOUND

    async def test_failed_calls_count_against_support(self):
        claim_judge, _ = voting_judge(LlmError("x"), supported(), LlmError("y"))
        result = await claim_judge.judge_sentence(SENTENCE, [INTEL])
        assert result.verdict is ClaimVerdict.UNSUPPORTED
        assert result.reason is JudgmentReason.JUDGE_ERROR

    async def test_report_agreement_is_the_mean_over_sentences(self):
        claim_judge, _ = voting_judge(
            supported(), supported(), supported(), supported(), unsupported(), supported()
        )
        report = await claim_judge.judge_answer(f"{SENTENCE}\n{SENTENCE}", SOURCES)
        assert report.agreement == pytest.approx((1 + 2 / 3) / 2)

    @pytest.mark.parametrize("votes", [0, 2, 4])
    def test_even_or_zero_votes_are_rejected_so_ties_cannot_happen(self, votes):
        with pytest.raises(ValueError, match="odd"):
            ClaimJudge(ScriptedLlm(), min_quote_chars=1, max_concurrency=1, votes=votes)


class TestJudgeSettings:
    @pytest.mark.parametrize("votes", [2, 4])
    def test_settings_reject_even_vote_counts(self, votes):
        with pytest.raises(ValueError, match="odd"):
            Settings(claim_judge_votes=votes)

    def test_settings_reject_a_floor_above_the_threshold(self):
        with pytest.raises(ValueError, match="FLOOR"):
            Settings(claim_support_threshold=0.7, claim_support_floor=0.8)


def report_scoring(supported_count: int, judged: int) -> FaithfulnessReport:
    verdicts = [ClaimVerdict.SUPPORTED] * supported_count
    verdicts += [ClaimVerdict.UNSUPPORTED] * (judged - supported_count)
    return FaithfulnessReport(judgments=[judgment(v) for v in verdicts])


class TestFaithfulnessSummary:
    def test_mean_ignores_cases_with_nothing_to_judge(self):
        reports = {
            "q1": report_scoring(4, 4),
            "q2": report_scoring(3, 4),
            "q12": report_scoring(0, 0),
        }
        summary = FaithfulnessSummary.from_reports(reports, threshold=0.8, floor=0.6)
        assert summary.scores == {"q1": 1.0, "q2": 0.75}
        assert summary.mean == pytest.approx(0.875)
        assert summary.passes

    def test_mean_below_threshold_fails(self):
        reports = {"q1": report_scoring(3, 4), "q2": report_scoring(3, 4)}
        summary = FaithfulnessSummary.from_reports(reports, threshold=0.8, floor=0.6)
        assert not summary.mean_meets_threshold and not summary.passes
        assert summary.below_floor == []

    def test_one_case_under_the_floor_fails_even_with_a_good_mean(self):
        reports = {f"q{n}": report_scoring(4, 4) for n in range(1, 10)}
        reports["q10"] = report_scoring(1, 2)
        summary = FaithfulnessSummary.from_reports(reports, threshold=0.8, floor=0.6)
        assert summary.mean_meets_threshold
        assert summary.below_floor == ["q10"]
        assert not summary.passes

    def test_nothing_scored_passes_without_a_mean(self):
        summary = FaithfulnessSummary.from_reports(
            {"q12": report_scoring(0, 0)}, threshold=0.8, floor=0.6
        )
        assert summary.mean is None and summary.passes
