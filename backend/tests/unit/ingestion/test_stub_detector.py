import pytest

from chat_app.ingestion.cleaning.models import CleanedArticle, CleaningSignals
from chat_app.ingestion.stub_detector import StubDetector, StubReason

LONG_TEXT = " ".join(["word"] * 100)


def cleaned(text: str, truncation_markers: list[str] | None = None) -> CleanedArticle:
    signals = CleaningSignals(truncation_markers=truncation_markers or [])
    return CleanedArticle(title="t", link="l", ticker="X", text=text, signals=signals)


@pytest.mark.parametrize(
    ("article", "reason"),
    [
        (cleaned(LONG_TEXT, ["continue_reading"]), StubReason.TRUNCATED),
        (cleaned("short", ["paywall_premium"]), StubReason.TRUNCATED),  # truncation wins
        (cleaned("a complete but short recap"), StubReason.TOO_SHORT),
        (cleaned(LONG_TEXT), None),
    ],
)
def test_reason(article, reason):
    detector = StubDetector(min_words=80)
    assert detector.reason(article) == reason
    assert detector.is_stub(article) is (reason is not None)


@pytest.mark.parametrize(("words", "is_stub"), [(79, True), (80, False)])
def test_threshold_is_exclusive(words, is_stub):
    assert StubDetector(min_words=80).is_stub(cleaned(" ".join(["w"] * words))) is is_stub
