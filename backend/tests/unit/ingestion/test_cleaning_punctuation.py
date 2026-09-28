"""Cleaning step: punctuation."""

import pytest

from chat_app.core.text_normalization import PUNCTUATION_FOLDS, fold_punctuation
from chat_app.ingestion.cleaning.steps.punctuation import PunctuationNormalizationStep


class TestPunctuationNormalization:
    def test_folds_nbsp_and_keeps_ticker(self, context, snippets):
        result = PunctuationNormalizationStep().apply(snippets["nbsp_and_ticker"], context).text
        assert " " not in result
        assert "Apple Inc. (NASDAQ:AAPL) stands" in result

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("curly_quote", "DeepSeek's AI breakthrough"),
            ("en_dash", "Walmart & Target - Many"),
            ("ellipsis", "buy now... and Apple wasn't one of them."),
        ],
    )
    def test_folds_typographic_punctuation(self, context, snippets, key, expected):
        assert expected in PunctuationNormalizationStep().apply(snippets[key], context).text

    def test_minus_sign_folds_to_hyphen(self):
        assert fold_punctuation("returned −0.98%") == "returned -0.98%"

    @pytest.mark.parametrize("text", ["café", "Ömer", "장치", "€9.26", "CN¥25"])
    def test_never_touches_letters_or_currency(self, text):
        assert fold_punctuation(text) == text

    def test_no_fold_produces_a_digit(self):
        assert not any(ch.isdigit() for value in PUNCTUATION_FOLDS.values() for ch in value)
