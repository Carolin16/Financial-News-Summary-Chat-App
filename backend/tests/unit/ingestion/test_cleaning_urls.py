"""Cleaning step: URL removal."""

import pytest

from chat_app.ingestion.cleaning.steps.pattern_removal import PatternRemovalStep


@pytest.fixture
def step(cleaning_config) -> PatternRemovalStep:
    return PatternRemovalStep("urls", cleaning_config.url_patterns)


def test_removes_url_but_not_trailing_punctuation(step, context, snippets):
    text = step.apply(snippets["url_in_sentence"], context).text
    assert "https://" not in text
    assert "visit   or follow" in text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see www.zacks.com/disclaimer. Next", "see  . Next"),
        ("(https://x.com/a/b) more", "( ) more"),
    ],
)
def test_url_forms(step, context, text, expected):
    assert step.apply(text, context).text == expected


@pytest.mark.parametrize(
    "text", ["Amazon.com (NASDAQ:AMZN)", "Booking.com shares", "Benzinga.com reported", "http, ssl"]
)
def test_bare_domains_and_words_are_company_names_not_urls(step, context, text):
    assert step.apply(text, context).text == text
