"""Cleaning step: encoding repair."""

import pytest

from chat_app.ingestion.cleaning.steps.encoding_repair import EncodingRepairStep


class TestEncodingRepair:
    # Synthetic: the dataset has no mojibake, so these reproduce the classic UTF-8-as-cp1252
    # damage the brief describes.
    @pytest.mark.parametrize(
        ("damaged", "repaired"),
        [
            ("Appleâ€™s shares", "Apple’s shares"),
            ("â€œAI boomâ€\u009d", "“AI boom”"),
        ],
    )
    def test_repairs_mojibake(self, context, damaged, repaired):
        assert EncodingRepairStep().apply(damaged, context).text == repaired

    def test_leaves_real_non_ascii_text_alone(self, context, snippets):
        for key in ("currency_euro", "currency_yuan", "credit_getty_parenthetical"):
            text = snippets[key]
            assert EncodingRepairStep().apply(text, context).text == text
