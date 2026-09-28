"""Builds a TextCleaner from configuration.

`STEP_BUILDERS` maps step names used in the rules file to constructors. Adding a step is a
new module plus one entry here; the order is chosen in the rules file.
"""

from collections.abc import Callable, Iterable

from chat_app.ingestion.cleaning.config import CleaningConfig
from chat_app.ingestion.cleaning.footer_guard import FooterGuard
from chat_app.ingestion.cleaning.promo_block import PromoBlockScanner
from chat_app.ingestion.cleaning.step import CleaningStep
from chat_app.ingestion.cleaning.steps.encoding_repair import EncodingRepairStep
from chat_app.ingestion.cleaning.steps.noise_blocks import NoiseBlockStep
from chat_app.ingestion.cleaning.steps.pattern_removal import PatternRemovalStep
from chat_app.ingestion.cleaning.steps.punctuation import PunctuationNormalizationStep
from chat_app.ingestion.cleaning.steps.repeated_title import RepeatedTitleStep
from chat_app.ingestion.cleaning.steps.sentence_spacing import SentenceSpacingStep
from chat_app.ingestion.cleaning.steps.whitespace import WhitespaceStep
from chat_app.ingestion.cleaning.text_cleaner import TextCleaner

StepBuilder = Callable[[CleaningConfig, list[str]], CleaningStep]

STEP_BUILDERS: dict[str, StepBuilder] = {
    "encoding_repair": lambda config, terms: EncodingRepairStep(),
    "punctuation": lambda config, terms: PunctuationNormalizationStep(),
    "sentence_spacing": lambda config, terms: SentenceSpacingStep(),
    "symbols": lambda config, terms: PatternRemovalStep("symbols", config.symbol_patterns),
    "noise_blocks": lambda config, terms: NoiseBlockStep(
        config.noise_rules,
        PromoBlockScanner(config.promo_block),
        FooterGuard(config.footer_safety, terms),
        config.keep_ranking_labels,
    ),
    "urls": lambda config, terms: PatternRemovalStep("urls", config.url_patterns),
    "repeated_title": lambda config, terms: RepeatedTitleStep(),
    "whitespace": lambda config, terms: WhitespaceStep(),
}


def build_text_cleaner(config: CleaningConfig, company_terms: Iterable[str]) -> TextCleaner:
    """Instantiate body and title steps in the configured order.

    `company_terms` (tracked company names and tickers) feed the footer safety guard.
    """
    terms = list(company_terms)
    unknown = set(config.step_order + config.title_step_order) - STEP_BUILDERS.keys()
    if unknown:
        raise ValueError(f"unknown cleaning steps in config: {sorted(unknown)}")
    return TextCleaner(
        steps=[STEP_BUILDERS[name](config, terms) for name in config.step_order],
        title_steps=[STEP_BUILDERS[name](config, terms) for name in config.title_step_order],
    )
