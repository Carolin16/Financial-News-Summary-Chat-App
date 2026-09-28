"""Typed cleaning configuration, loaded from TOML so patterns are data, not code."""

import re
import tomllib
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_RULES_PATH = Path(__file__).with_name("cleaning_rules.toml")


class Strategy(StrEnum):
    """How far a noise rule's removal extends from where its pattern matches."""

    SPAN = "span"  # exactly the match
    SENTENCE = "sentence"  # the whole sentence containing the match
    PROMO_BLOCK = "promo_block"  # from the match to the end of the promo (see PromoBlockScanner)
    TO_END = "to_end"  # from the match to the end of the text (publisher footers)


class NoiseRule(BaseModel):
    """One pattern of boilerplate or promotional text."""

    model_config = ConfigDict(frozen=True)

    name: str
    category: str
    pattern: str
    strategy: Strategy = Strategy.SPAN
    ignore_case: bool = False
    truncation: bool = Field(
        default=False, description="Report a truncation signal when this rule fires."
    )
    ranking_label: bool = Field(
        default=False, description="Only applied when ranking labels are configured off."
    )
    allow_foreign_headlines: bool = Field(
        default=False,
        description="Promo blocks: keep scanning through non-English headline runs.",
    )

    @cached_property
    def regex(self) -> re.Pattern[str]:
        """Compiled pattern."""
        return re.compile(self.pattern, re.IGNORECASE if self.ignore_case else 0)


class PromoBlockConfig(BaseModel):
    """Parameters for finding where a promo block ends."""

    max_words: int
    title_case_probe_words: int
    min_words_before_boundary: int
    connectors: list[str]
    sentence_openers: list[str]
    end_markers: list[str]


class FooterSafetyConfig(BaseModel):
    """Guard against stripping a footer tail that still holds article content."""

    min_sentence_words: int
    boilerplate: list[str]

    @cached_property
    def boilerplate_regexes(self) -> list[re.Pattern[str]]:
        """Compiled boilerplate patterns (case-insensitive)."""
        return [re.compile(p, re.IGNORECASE) for p in self.boilerplate]


class CleaningConfig(BaseModel):
    """Everything the cleaning pipeline needs, in one validated object."""

    step_order: list[str]
    title_step_order: list[str]
    keep_ranking_labels: bool = True
    symbol_patterns: list[str]
    url_patterns: list[str]
    promo_block: PromoBlockConfig
    footer_safety: FooterSafetyConfig
    noise_rules: list[NoiseRule]

    @classmethod
    def from_toml(cls, path: Path = DEFAULT_RULES_PATH) -> "CleaningConfig":
        """Load and validate a TOML rules file."""
        with path.open("rb") as handle:
            return cls.model_validate(tomllib.load(handle))
