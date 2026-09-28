import pytest

from chat_app.core.models import ArticleMetadata
from chat_app.ingestion.chunker import STUB_NOTE, Chunker

NAMES = {"INTC": "Intel", "AVGO": "Broadcom"}


def make_chunker(max_tokens: int = 500, overlap: int = 1) -> Chunker:
    return Chunker(
        max_tokens=max_tokens,
        overlap_sentences=overlap,
        encoding_name="cl100k_base",
        company_name=lambda t: NAMES.get(t, t),
    )


@pytest.fixture
def long_text() -> str:
    return " ".join(f"Sentence number {i} says Intel shares moved {i}%." for i in range(120))


def test_short_article_stays_a_single_chunk(make_article):
    chunks = make_chunker().chunk(make_article(text="Intel shares rose 5% on Tuesday."))
    assert len(chunks) == 1
    assert chunks[0].text == "Intel shares rose 5% on Tuesday."


def test_long_article_is_split_within_budget(make_article, long_text):
    chunker = make_chunker(max_tokens=120)
    chunks = chunker.chunk(make_article(text=long_text))
    assert len(chunks) > 1
    assert all(chunker._count(c.contextualized_text) <= 120 for c in chunks)


def test_chunks_break_on_sentence_boundaries(make_article, long_text):
    for chunk in make_chunker(max_tokens=120).chunk(make_article(text=long_text)):
        assert chunk.text.startswith("Sentence number")
        assert chunk.text.endswith("%.")


def test_consecutive_chunks_overlap_by_one_sentence(make_article, long_text):
    first, second, *_ = make_chunker(max_tokens=120).chunk(make_article(text=long_text))
    last_sentence_of_first = first.text.rsplit("Sentence", 1)[-1]
    assert second.text.startswith(f"Sentence{last_sentence_of_first}")


def test_no_content_is_lost(make_article, long_text):
    chunks = make_chunker(max_tokens=120, overlap=0).chunk(make_article(text=long_text))
    assert " ".join(c.text for c in chunks) == long_text


def test_oversized_sentence_falls_back_to_word_windows(make_article):
    run_on = " ".join(["word"] * 400)
    chunker = make_chunker(max_tokens=100)
    chunks = chunker.chunk(make_article(text=run_on))
    assert len(chunks) > 1
    assert all(chunker._count(c.contextualized_text) <= 100 for c in chunks)


def test_header_carries_title_companies_and_stub_note(make_article):
    article = make_article(
        title="Cantor Adjusts Intel Target to $29",
        is_stub=True,
        metadata=ArticleMetadata(primary_tickers=["INTC"], mentioned_tickers=["AVGO"]),
    )
    header = make_chunker().build_header(article)
    assert "Title: Cantor Adjusts Intel Target to $29" in header
    assert "About: Intel (INTC)" in header
    assert "Also mentions: Broadcom (AVGO)" in header
    assert STUB_NOTE in header


def test_chunk_ids_are_deterministic_and_carry_metadata(make_article, long_text):
    article = make_article(text=long_text, metadata=ArticleMetadata(primary_tickers=["INTC"]))
    first_run = make_chunker(max_tokens=120).chunk(article)
    second_run = make_chunker(max_tokens=120).chunk(article)
    assert [c.chunk_id for c in first_run] == [c.chunk_id for c in second_run]
    assert len({c.chunk_id for c in first_run}) == len(first_run)
    assert all(c.metadata.primary_tickers == ["INTC"] for c in first_run)
