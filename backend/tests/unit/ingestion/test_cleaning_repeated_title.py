"""Cleaning step: removal of a title repeated at the start of the body."""

import pytest

from chat_app.ingestion.cleaning.models import CleaningContext
from chat_app.ingestion.cleaning.steps.repeated_title import RepeatedTitleStep


def test_removes_real_repeated_title(snippets):
    title, text = snippets["repeated_title"]["title"], snippets["repeated_title"]["text"]
    result = RepeatedTitleStep().apply(text, CleaningContext(title=title))
    assert result.text.startswith("Apple Inc.'s (NASDAQ:AAPL) key supplier Foxconn")
    assert result.chars_removed_by_rule["repeated_title"] == len(text) - len(result.text)


@pytest.mark.parametrize(
    ("title", "text", "expected"),
    [
        ("Intel Jumps", "intel  jumps After the report.", "After the report."),
        (
            "Intel Jumps",
            "Intel shares jumped after the report.",
            "Intel shares jumped after the report.",
        ),
        ("Intel Jump", "Intel Jumps after the report.", "Intel Jumps after the report."),
        ("", "Body text.", "Body text."),
    ],
)
def test_only_exact_whole_word_prefixes_are_removed(title, text, expected):
    assert RepeatedTitleStep().apply(text, CleaningContext(title=title)).text == expected


def test_curly_quotes_in_title_match_folded_body():
    title = "Apple’s Big Day"
    result = RepeatedTitleStep().apply("Apple's Big Day Shares rose.", CleaningContext(title=title))
    assert result.text == "Shares rose."
