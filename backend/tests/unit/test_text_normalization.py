"""The shared punctuation normaliser, used by ingestion and by answer verification."""

from chat_app.core.models import ArticleMetadata, Chunk, RetrievedChunk
from chat_app.core.text_normalization import fold_punctuation
from chat_app.generation.context import Source
from chat_app.generation.grounding import extract_numbers, verify_answer


def test_fold_is_idempotent():
    text = "“It’s − 5…”"
    assert fold_punctuation(fold_punctuation(text)) == fold_punctuation(text) == '"It\'s - 5..."'


def test_verifier_matches_numbers_written_with_typographic_characters():
    chunk = Chunk(
        chunk_id="c",
        article_id="a",
        chunk_index=0,
        title="t",
        link="l",
        text="The fund returned -0.98% versus 2.41% for the index.",
        header="Title: t",
        is_stub=False,
        metadata=ArticleMetadata(),
    )
    sources = [Source(number=1, retrieved=RetrievedChunk(chunk=chunk, score=1.0))]
    answer = "The fund returned −0.98% versus 2.41% [1]."
    assert verify_answer(answer, sources).removed == []
    assert extract_numbers("−0.98%") == {"0.98"}
