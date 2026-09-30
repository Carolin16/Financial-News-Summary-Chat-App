"""Composition root for offline ingestion: builds concrete stages from settings."""

from chat_app.config.relevance import RelevanceConfig
from chat_app.config.settings import Settings
from chat_app.core.ticker_registry import TickerRegistry
from chat_app.ingestion.article_pipeline import ArticlePipeline
from chat_app.ingestion.cleaning.config import CleaningConfig
from chat_app.ingestion.cleaning.factory import build_text_cleaner
from chat_app.ingestion.deduplicator import Deduplicator
from chat_app.ingestion.loader import JsonArticleRepository
from chat_app.ingestion.relevance.article_type import CueArticleTypeClassifier
from chat_app.ingestion.relevance.enricher import RuleBasedEnricher
from chat_app.ingestion.relevance.entities import RegistryEntityMatcher
from chat_app.ingestion.relevance.events import KeywordEventTagger, LexiconSentimentScorer
from chat_app.ingestion.relevance.salience import WeightedSalienceScorer
from chat_app.ingestion.stub_detector import StubDetector


def build_enricher(settings: Settings) -> RuleBasedEnricher:
    """Build the rule-based relevance tagger from the ticker list and rules file."""
    registry = TickerRegistry.from_json(settings.tickers_path)
    rules = RelevanceConfig.from_toml(settings.relevance_rules_path)
    return RuleBasedEnricher(
        matcher=RegistryEntityMatcher(registry, rules.entities),
        scorer=WeightedSalienceScorer(rules.salience),
        classifier=CueArticleTypeClassifier(rules.article_type),
        event_tagger=KeywordEventTagger(rules.events),
        sentiment_scorer=LexiconSentimentScorer(rules.sentiment),
        tiers=rules.tiers,
        enumeration_min_companies=rules.entities.enumeration_min_companies,
    )


def build_article_pipeline(settings: Settings, enricher: RuleBasedEnricher) -> ArticlePipeline:
    """Assemble the article preparation pipeline from configuration."""
    return ArticlePipeline(
        repository=JsonArticleRepository(settings.data_path),
        cleaner=build_text_cleaner(
            CleaningConfig.from_toml(settings.cleaning_rules_path),
            TickerRegistry.from_json(settings.tickers_path).company_terms(),
        ),
        stub_detector=StubDetector(settings.stub_min_words),
        deduplicator=Deduplicator(
            body_threshold=settings.near_duplicate_threshold,
            title_threshold=settings.title_duplicate_threshold,
            shingle_size=settings.shingle_size,
        ),
        extractor=enricher,
        enrichment_workers=settings.enrichment_workers,
    )
