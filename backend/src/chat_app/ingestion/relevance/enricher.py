"""Composes the relevance strategies into article metadata (a MetadataExtractor).

Rule-based: entity matching -> salience -> tier (threshold depends on article type) ->
events and tone from the sentences about the article's primary companies.
"""

import re

from chat_app.config.relevance import TierConfig
from chat_app.core.models import Article, ArticleMetadata, RelevanceTier
from chat_app.ingestion.relevance.article_type import ArticleTypeClassifier
from chat_app.ingestion.relevance.entities import EntityMatcher, EntityMention
from chat_app.ingestion.relevance.events import EventTagger, SentimentScorer
from chat_app.ingestion.relevance.salience import SalienceScorer
from chat_app.ingestion.sentences import sentence_spans

_WORD = re.compile(r"\S+")


class RuleBasedEnricher:
    """Labels an article's companies, type, events, and tone. Stub status is not an input."""

    def __init__(
        self,
        matcher: EntityMatcher,
        scorer: SalienceScorer,
        classifier: ArticleTypeClassifier,
        event_tagger: EventTagger,
        sentiment_scorer: SentimentScorer,
        tiers: TierConfig,
        enumeration_min_companies: int,
    ) -> None:
        """Every strategy is injected, so each can be swapped or faked in tests."""
        self._matcher = matcher
        self._scorer = scorer
        self._classifier = classifier
        self._events = event_tagger
        self._sentiment = sentiment_scorer
        self._tiers = tiers
        self._enumeration_min = enumeration_min_companies

    def extract(self, article: Article) -> ArticleMetadata:
        """Metadata judged from the article's content, never from its source keys."""
        verdict = self._classifier.classify(article.title, article.text)
        found = self._matcher.find(article.title, article.text)
        title_mentions, body_mentions = found.title, found.body
        word_offsets = [m.start() for m in _WORD.finditer(article.text)]
        results = self._scorer.score(title_mentions, body_mentions, word_offsets)
        sentences = sentence_spans(article.text)
        incidental = self._incidental(title_mentions, body_mentions, sentences)
        threshold = self._tiers.threshold_for(verdict.article_type)
        incidental_by_type = verdict.article_type in self._tiers.incidental_article_types

        tiers: dict[str, RelevanceTier] = {}
        for ticker, result in results.items():
            if ticker in incidental:
                tiers[ticker] = RelevanceTier.INCIDENTAL
                continue
            tier = RelevanceTier.PRIMARY if result.score >= threshold else RelevanceTier.MENTIONED
            if tier is RelevanceTier.MENTIONED and incidental_by_type:
                tier = RelevanceTier.INCIDENTAL
            tiers[ticker] = tier

        salience = {t: r.score for t, r in results.items()}
        primary = _ordered(tiers, salience, RelevanceTier.PRIMARY)
        focus = self._focus_sentences(article.text, sentences, body_mentions, set(primary))
        return ArticleMetadata(
            primary_tickers=primary,
            mentioned_tickers=_ordered(tiers, salience, RelevanceTier.MENTIONED),
            incidental_tickers=_ordered(tiers, salience, RelevanceTier.INCIDENTAL),
            tiers=tiers,
            salience=salience,
            event_types=self._events.tag(article.title, focus),
            sentiment=self._sentiment.score(focus),
            article_type=verdict.article_type,
            article_type_cue=verdict.cue,
        )

    def _incidental(
        self,
        title_mentions: list[EntityMention],
        body_mentions: list[EntityMention],
        sentences: list[tuple[int, int]],
    ) -> set[str]:
        """Companies named only inside long enumerations (and not in the title).

        E.g. a market-size press release listing "IBM, Oracle, SAP, ..." as key players.
        A single passing mention in ordinary prose is still MENTIONED, not incidental.
        """
        enumerations = [
            (start, end)
            for start, end in sentences
            if len({m.ticker for m in body_mentions if start <= m.start < end})
            >= self._enumeration_min
        ]
        in_title = {m.ticker for m in title_mentions}
        positions: dict[str, list[int]] = {}
        for mention in body_mentions:
            positions.setdefault(mention.ticker, []).append(mention.start)
        return {
            ticker
            for ticker, offsets in positions.items()
            if ticker not in in_title
            and all(any(s <= p < e for s, e in enumerations) for p in offsets)
        }

    @staticmethod
    def _focus_sentences(
        text: str,
        sentences: list[tuple[int, int]],
        mentions: list[EntityMention],
        primary: set[str],
    ) -> list[str]:
        """Sentences naming a primary company; all sentences if the article has none."""
        if not primary:
            return [text[s:e] for s, e in sentences]
        positions = [m.start for m in mentions if m.ticker in primary]
        return [text[s:e] for s, e in sentences if any(s <= p < e for p in positions)]


def _ordered(
    tiers: dict[str, RelevanceTier], salience: dict[str, float], tier: RelevanceTier
) -> list[str]:
    """Tickers of one tier, most salient first, then alphabetical (stable output)."""
    return sorted((t for t, v in tiers.items() if v is tier), key=lambda t: (-salience[t], t))
