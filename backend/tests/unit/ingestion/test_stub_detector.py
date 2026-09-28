from chat_app.ingestion.stub_detector import StubDetector

LONG_TEXT = " ".join(["word"] * 100)


def test_teaser_marker_makes_stub_even_when_long():
    detector = StubDetector(min_words=10)
    assert detector.is_stub(raw_text=f"{LONG_TEXT} Continue Reading", cleaned_text=LONG_TEXT)


def test_paywall_marker_makes_stub():
    raw = "Intel has a mean target of $22.03 PREMIUM Upgrade to read this MT Newswires article"
    assert StubDetector(min_words=5).is_stub(raw_text=raw, cleaned_text=LONG_TEXT)


def test_short_cleaned_text_is_stub():
    assert StubDetector(min_words=80).is_stub(raw_text="short", cleaned_text="short text")


def test_full_article_is_not_stub():
    assert not StubDetector(min_words=80).is_stub(raw_text=LONG_TEXT, cleaned_text=LONG_TEXT)
