from chat_app.ingestion.cleaning.models import CleanedArticle, CleaningSignals
from chat_app.ingestion.stub_detector import StubDetector

LONG_TEXT = " ".join(["word"] * 100)


def cleaned(text: str, truncation_markers: list[str] | None = None) -> CleanedArticle:
    signals = CleaningSignals(truncation_markers=truncation_markers or [])
    return CleanedArticle(title="t", link="l", ticker="X", text=text, signals=signals)


def test_truncation_signal_makes_stub_even_when_long():
    assert StubDetector(min_words=10).is_stub(cleaned(LONG_TEXT, ["continue_reading"]))


def test_short_cleaned_text_is_stub():
    assert StubDetector(min_words=80).is_stub(cleaned("short text"))


def test_full_article_without_signal_is_not_stub():
    assert not StubDetector(min_words=80).is_stub(cleaned(LONG_TEXT))
