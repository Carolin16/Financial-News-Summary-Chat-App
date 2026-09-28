"""Cleaning step: whitespace and punctuation-spacing tidy-up."""

import pytest

from chat_app.ingestion.cleaning.steps.whitespace import WhitespaceStep


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  Intel   rose \n\t 5% .  ", "Intel rose 5%."),
        ("shares fell . . Next", "shares fell. Next"),
        ("visit ( ) today", "visit today"),
        ("growth ( per year )", "growth (per year)"),
        ("now... and Apple", "now... and Apple"),
    ],
)
def test_tidies_spacing(context, text, expected):
    assert WhitespaceStep().apply(text, context).text == expected


def test_never_glues_digits_into_a_new_number(context):
    # Removing the space before "." would turn "5 .3" into "5.3".
    assert WhitespaceStep().apply("rose 5 .3 points", context).text == "rose 5 .3 points"


@pytest.mark.parametrize("text", ["a . . b", "x ( ) y .", " lots   of   space "])
def test_idempotent(context, text):
    once = WhitespaceStep().apply(text, context).text
    assert WhitespaceStep().apply(once, context).text == once
