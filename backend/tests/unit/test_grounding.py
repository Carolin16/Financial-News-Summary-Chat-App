import pytest

from chat_app.core.models import ArticleMetadata, Chunk, RetrievedChunk
from chat_app.generation.context import Source, format_sources, number_sources
from chat_app.generation.grounding import RemovalReason, extract_numbers, verify_answer


def source(number: int, text: str, title: str = "Title", is_stub=False, primary=("NVDA",)):
    chunk = Chunk(
        chunk_id=f"c{number}",
        article_id=f"a{number}",
        chunk_index=0,
        title=title,
        link=f"https://x/{number}",
        text=text,
        header=f"Title: {title}",
        is_stub=is_stub,
        metadata=ArticleMetadata(primary_tickers=list(primary)),
    )
    return Source(number=number, retrieved=RetrievedChunk(chunk=chunk, score=1.0))


SOURCES = [
    source(
        1,
        "NVIDIA has a mean price target of $174.93.",
        title="DBS Adjusts NVIDIA Target to $160 From $175",
    ),
    source(2, "Intel shares rose 16% after reports of a Broadcom deal. It returned $30 billion."),
]


class TestExtractNumbers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("$1,024.05 billion", {"1024.05"}),
            ("rose 13% to $22.03", {"13", "22.03"}),
            ("a 7-9% yield", {"7", "9"}),
            ("Q4 results for the H100", set()),
            ("cites [1] and [12]", set()),
        ],
    )
    def test_canonical_forms(self, text, expected):
        assert extract_numbers(text) == expected


class TestVerifyAnswer:
    def test_keeps_cited_sentences_with_traceable_numbers(self):
        answer = "- DBS cut its target to $160 from $175 [1].\n- The mean target is $174.93 [1]."
        report = verify_answer(answer, SOURCES)
        assert report.text == answer
        assert report.cited_numbers == [1]
        assert report.removed == []

    def test_numbers_in_titles_count_as_grounded(self):
        report = verify_answer("DBS set a $160 target [1].", SOURCES)
        assert report.removed == []

    def test_removes_hallucinated_number(self):
        answer = "- Shares rose 16% [2].\n- Analysts expect $200 per share [1]."
        report = verify_answer(answer, SOURCES)
        assert report.text == "- Shares rose 16% [2]."
        [removed] = report.removed
        assert removed.reason is RemovalReason.UNSUPPORTED_NUMBER
        assert removed.detail == "200"

    def test_number_must_be_in_the_cited_source_not_just_any_source(self):
        report = verify_answer("NVIDIA's mean target is $174.93 [2].", SOURCES)
        assert report.text == ""
        assert report.is_empty

    def test_removes_uncited_claims_but_keeps_lead_ins(self):
        answer = (
            "Here is what the articles report:\n- Nvidia is a great company.\n- Intel rose 16% [2]."
        )
        report = verify_answer(answer, SOURCES)
        assert report.text == "Here is what the articles report:\n- Intel rose 16% [2]."
        assert report.removed[0].reason is RemovalReason.UNCITED

    def test_strips_citations_to_nonexistent_sources(self):
        report = verify_answer("Intel rose 16% [2][7].", SOURCES)
        assert report.text == "Intel rose 16% [2]."

    def test_citation_after_the_period_stays_with_its_sentence(self):
        report = verify_answer("Intel shares rose 16%. [2] DBS cut its target. [1]", SOURCES)
        assert report.removed == []
        assert report.cited_numbers == [2, 1]

    def test_multi_sentence_line_prunes_only_the_bad_sentence(self):
        report = verify_answer("Intel rose 16% [2]. It will hit $50 [2].", SOURCES)
        assert report.text == "Intel rose 16% [2]."


class TestFormatSources:
    def test_labels_partial_and_passing_mentions(self):
        sources = number_sources(
            [
                source(0, "Teaser", is_stub=True).retrieved,
                source(0, "Market wrap", primary=()).retrieved,
            ]
        )
        rendered = format_sources(sources, focus_tickers=["NVDA"])
        assert "[1] (PARTIAL)\nTitle: Title" in rendered
        assert "[2] (mentions the company only in passing)" in rendered
