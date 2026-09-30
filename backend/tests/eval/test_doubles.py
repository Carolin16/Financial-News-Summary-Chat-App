"""The context-dependence test double and answer expectations (no LLM; runs by default)."""

import pytest
from cases import AnswerExpectation, span_pattern
from doubles import DropLinks, ReplaceSpan, TransformingRetriever

from chat_app.core.models import ArticleMetadata, Chunk, RetrievedChunk, SearchFilters

DBS_LINK = "https://x/dbs"


def chunk(link: str, title: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{link}-0",
        article_id=link,
        chunk_index=0,
        title=title,
        link=link,
        text=text,
        header=f"Title: {title}",
        is_stub=False,
        metadata=ArticleMetadata(primary_tickers=["NVDA"]),
    )


DBS = chunk(DBS_LINK, "DBS Adjusts NVIDIA Target to $160 From $175", "Mean target is $174.93.")
ARM = chunk("https://x/arm", "Arm edges higher", "Arm rose to $160.91 each.")


class FakeRetriever:
    """Returns fixed chunks and records the arguments it was called with."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.calls: list[tuple[str, SearchFilters, int]] = []

    async def retrieve(self, query, filters, top_k):
        self.calls.append((query, filters, top_k))
        return [RetrievedChunk(chunk=c, score=1.0 - i / 10) for i, c in enumerate(self.chunks)]


class TestReplaceSpan:
    def test_replaces_in_title_header_and_body(self):
        edited = ReplaceSpan("$160", "$153").apply(DBS)
        assert edited.title == "DBS Adjusts NVIDIA Target to $153 From $175"
        assert edited.header == "Title: DBS Adjusts NVIDIA Target to $153 From $175"
        assert "$153" in edited.contextualized_text and "$160" not in edited.contextualized_text

    def test_leaves_longer_figures_alone(self):
        assert ReplaceSpan("$160", "$153").apply(ARM) == ARM

    def test_word_replacement_matches_whole_words_only(self):
        text = chunk("l", "t", "Ford said buyers can afford it; see Stanford.")
        edited = ReplaceSpan("Ford", "Kestrel Motors").apply(text)
        assert edited.text == "Kestrel Motors said buyers can afford it; see Stanford."

    def test_replacement_text_is_literal(self):
        edited = ReplaceSpan("Broadcom", r"Veltrix \1 $0").apply(chunk("l", "t", "Broadcom bid"))
        assert edited.text == r"Veltrix \1 $0 bid"


class TestTransformingRetriever:
    async def test_passes_arguments_through_and_keeps_scores(self):
        inner = FakeRetriever([DBS, ARM])
        filters = SearchFilters(tickers=["NVDA"])
        results = await TransformingRetriever(inner, []).retrieve("q", filters, 5)
        assert inner.calls == [("q", filters, 5)]
        assert [r.chunk for r in results] == [DBS, ARM]
        assert [r.score for r in results] == [1.0, 0.9]

    async def test_drop_links_removes_only_that_article(self):
        double = TransformingRetriever(FakeRetriever([DBS, ARM]), [DropLinks([DBS_LINK])])
        results = await double.retrieve("q", SearchFilters(), 5)
        assert [r.chunk.link for r in results] == [ARM.link]
        assert double.touched == 1

    async def test_transforms_run_in_order_and_count_touched_passages(self):
        transforms = [ReplaceSpan("$160", "$153"), ReplaceSpan("$153", "$150")]
        double = TransformingRetriever(FakeRetriever([DBS, ARM]), transforms)
        [dbs, arm] = await double.retrieve("q", SearchFilters(), 5)
        assert "$150" in dbs.chunk.title
        assert arm.chunk == ARM
        assert double.touched == 1

    async def test_untouched_results_report_zero(self):
        double = TransformingRetriever(FakeRetriever([ARM]), [DropLinks([DBS_LINK])])
        await double.retrieve("q", SearchFilters(), 5)
        assert double.touched == 0


class TestAnswerExpectation:
    def test_met_expectation_has_no_violations(self):
        expect = AnswerExpectation(
            include_terms=["Veltrix"], exclude_terms=["Broadcom"], include_figures=["$153"]
        )
        assert expect.violations("Veltrix Semiconductor set $153.00 [1].") == []

    def test_reports_each_violation(self):
        expect = AnswerExpectation(
            include_terms=["Veltrix"],
            exclude_terms=["Broadcom"],
            include_figures=["$153"],
            exclude_figures=["$160"],
        )
        problems = expect.violations("Broadcom set $160 [1].")
        assert problems == [
            "missing term 'Veltrix'",
            "forbidden term 'Broadcom'",
            "missing figure $153",
            "forbidden figure $160",
        ]

    @pytest.mark.parametrize("answer", ["Buyers can afford it.", "Arm hit $160.91.", "$1,600"])
    def test_whole_token_matching_avoids_false_hits(self, answer):
        expect = AnswerExpectation(exclude_terms=["Ford"], exclude_figures=["$160"])
        assert expect.violations(answer) == []

    def test_term_matching_ignores_case(self):
        assert span_pattern("Broadcom").search("BROADCOM's bid")
