"""Typed relevance rules (entities, salience, tiers, types, events, tone), loaded from TOML."""

import re
import tomllib
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_RELEVANCE_RULES_PATH = Path(__file__).with_name("relevance_rules.toml")


class EntityConfig(BaseModel):
    """How company mentions are recognised."""

    exchanges: list[str]
    not_tickers: list[str]
    name_suffixes: list[str]
    name_leading_words: list[str]
    name_generic_words: list[str]
    bare_symbol_min_count: int
    learned_name_min_chars: int
    enumeration_min_companies: int


class SalienceConfig(BaseModel):
    """Weights of the salience score; they should sum to at most 1."""

    title_weight: float
    title_subject_weight: float
    lead_weight: float
    exchange_tagged_weight: float
    frequency_weight: float
    share_weight: float
    lead_words: int
    frequency_cap_per_100_words: float
    frequency_min_mentions: int


class TierConfig(BaseModel):
    """Salience needed for a company to be primary, optionally per article type."""

    primary_threshold: float
    primary_threshold_by_type: dict[str, float] = Field(default_factory=dict)
    incidental_article_types: list[str] = Field(default_factory=list)

    def threshold_for(self, article_type: str) -> float:
        """Primary threshold for an article type (the default if not overridden)."""
        return self.primary_threshold_by_type.get(article_type, self.primary_threshold)


class ArticleTypeRule(BaseModel):
    """One article-type label and the cues that identify it."""

    label: str
    patterns: list[str]
    in_lead: bool

    @cached_property
    def regexes(self) -> list[re.Pattern[str]]:
        """Compiled cue patterns."""
        return [re.compile(p) for p in self.patterns]


class ArticleTypeConfig(BaseModel):
    """Ordered article-type rules; the first match wins."""

    lead_chars: int
    rules: list[ArticleTypeRule]


class SentimentConfig(BaseModel):
    """Finance tone lexicon and decision thresholds."""

    positive: list[str]
    negative: list[str]
    negators: list[str]
    negation_window: int
    min_hits: int
    polarity_threshold: float
    mixed_min_each: int


class RelevanceConfig(BaseModel):
    """All relevance rules in one validated object."""

    entities: EntityConfig
    salience: SalienceConfig
    tiers: TierConfig
    article_type: ArticleTypeConfig
    events: dict[str, list[str]]
    sentiment: SentimentConfig

    @classmethod
    def from_toml(cls, path: Path = DEFAULT_RELEVANCE_RULES_PATH) -> "RelevanceConfig":
        """Load and validate a TOML rules file."""
        with path.open("rb") as handle:
            return cls.model_validate(tomllib.load(handle))
