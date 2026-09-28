"""Cleaning step: symbols."""

import pytest

from chat_app.ingestion.cleaning.steps.pattern_removal import PatternRemovalStep


class TestSymbols:
    @pytest.fixture
    def step(self, cleaning_config) -> PatternRemovalStep:
        return PatternRemovalStep("symbols", cleaning_config.symbol_patterns)

    def test_strips_star_ratings_but_keeps_figures(self, context, snippets, step):
        result = step.apply(snippets["star_ratings"], context)
        assert "★" not in result.text
        assert "29.07% 27.57%" in result.text
        assert result.chars_removed_by_rule["symbols"] > 0

    def test_strips_emoji(self, context, snippets, step):
        assert "\U0001f4b0" not in step.apply(snippets["emoji_promo"], context).text

    @pytest.mark.parametrize("text", ["€9.26 billion", "CN¥25.45", "© 2025"])
    def test_keeps_currency_and_copyright_sign(self, context, step, text):
        # "©" is left for the copyright rule, which removes the whole line.
        assert step.apply(text, context).text == text
