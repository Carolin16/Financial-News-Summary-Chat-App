"""Cleaning step: sentence spacing."""

import pytest

from chat_app.ingestion.cleaning.steps.sentence_spacing import SentenceSpacingStep


class TestSentenceSpacing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("How do you compare?The impact", "How do you compare? The impact"),
            ("security.The success", "security. The success"),
        ],
    )
    def test_inserts_missing_space(self, context, text, expected):
        assert SentenceSpacingStep().apply(text, context).text == expected

    @pytest.mark.parametrize(
        "text", ["$4.5 billion", "BRK.B", "Amazon.com", "U.S.A", "e.l.f. Beauty"]
    )
    def test_leaves_numbers_tickers_and_domains(self, context, text):
        assert SentenceSpacingStep().apply(text, context).text == text
