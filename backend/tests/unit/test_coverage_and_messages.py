import pytest

from chat_app.core.models import ArticleMetadata, Chunk
from chat_app.generation import messages
from chat_app.retrieval.coverage import CoverageIndex, CoverageLevel


def chunk(article_id, primary=(), mentioned=(), is_stub=False, index=0):
    return Chunk(
        chunk_id=f"{article_id}-{index}",
        article_id=article_id,
        chunk_index=index,
        title="t",
        link="l",
        text="x",
        header="h",
        is_stub=is_stub,
        metadata=ArticleMetadata(primary_tickers=list(primary), mentioned_tickers=list(mentioned)),
    )


@pytest.fixture
def index() -> CoverageIndex:
    chunks = [
        *(chunk(f"intel{i}", primary=["INTC"]) for i in range(3)),
        chunk("intel0", primary=["INTC"], index=1),  # second chunk of same article
        chunk("intel-stub", primary=["INTC"], is_stub=True),
        chunk("ibm", primary=["IBM"]),
        chunk("wrap", mentioned=["TSLA", "INTC"]),
    ]
    return CoverageIndex(chunks, full_coverage_min_articles=3)


@pytest.mark.parametrize(
    ("ticker", "level", "primary", "mentions"),
    [
        ("INTC", CoverageLevel.FULL, 4, 1),
        ("IBM", CoverageLevel.LIMITED, 1, 0),
        ("TSLA", CoverageLevel.MENTIONS_ONLY, 0, 1),
        ("XOM", CoverageLevel.NONE, 0, 0),
    ],
)
def test_levels_count_unique_articles(index, ticker, level, primary, mentions):
    report = index.assess(ticker)
    assert (report.level, report.primary_articles, report.mention_articles) == (
        level,
        primary,
        mentions,
    )


def test_stubs_do_not_count_toward_full_coverage():
    chunks = [chunk(f"s{i}", primary=["NVDA"], is_stub=True) for i in range(5)]
    assert CoverageIndex(chunks, 3).assess("NVDA").level is CoverageLevel.LIMITED


def test_most_covered_is_restricted_and_ranked(index):
    assert [r.ticker for r in index.most_covered(5, among=["IBM", "INTC", "TSLA"])] == [
        "INTC",
        "IBM",
    ]
    assert index.total_articles == 6


class TestMessages:
    def test_limited_notice_uses_correct_grammar(self, index):
        notice = messages.coverage_notice(index.assess("IBM"), "IBM")
        assert notice == (
            "Coverage of IBM in this news set is limited: 1 article focuses on it and "
            "0 articles mention it in passing."
        )

    def test_full_coverage_has_no_notice_or_guidance(self, index):
        assert messages.coverage_notice(index.assess("INTC"), "Intel") is None
        assert messages.coverage_guidance(index.assess("INTC"), "Intel") == ""

    def test_mentions_only_never_claims_no_data(self, index):
        notice = messages.coverage_notice(index.assess("TSLA"), "Tesla")
        assert "only mentioned in passing in 1 article" in notice
        assert "None of the articles" not in notice

    @pytest.mark.parametrize(
        ("company", "expected"),
        [("Apple", "Apple's current market cap"), (None, "the current market cap")],
    )
    def test_live_data_notice(self, company, expected):
        assert expected in messages.live_data_notice(company, "market cap")
