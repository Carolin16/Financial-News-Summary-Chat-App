"""Scores how much an article is really about each company it mentions, from 0 to 1."""

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from chat_app.config.relevance import SalienceConfig
from chat_app.ingestion.relevance.entities import EntityMention


@dataclass(frozen=True)
class SalienceFeatures:
    """The clues a score was built from, kept so a score can be explained."""

    in_title: bool
    title_subject: bool
    in_lead: bool
    exchange_tagged: bool
    mentions: int
    mentions_per_100_words: float
    share: float
    relative_share: float


@dataclass(frozen=True)
class SalienceResult:
    """One company's score together with the clues behind it."""

    score: float
    features: SalienceFeatures


class SalienceScorer(Protocol):
    """Anything that can score how central each mentioned company is."""

    def score(
        self,
        title_mentions: list[EntityMention],
        body_mentions: list[EntityMention],
        body_word_offsets: list[int],
    ) -> dict[str, SalienceResult]:
        """Return a score per company, given its mentions and where each body word starts."""
        ...


class WeightedSalienceScorer:
    """Scores each company by adding up weighted clues: headline, early mention, frequency."""

    def __init__(self, config: SalienceConfig) -> None:
        """Load the weights and limits from the relevance rules file."""
        self._config = config

    def score(
        self,
        title_mentions: list[EntityMention],
        body_mentions: list[EntityMention],
        body_word_offsets: list[int],
    ) -> dict[str, SalienceResult]:
        """Return a score for every company named in the title or body."""
        config = self._config
        counts = Counter(m.ticker for m in body_mentions)
        total = sum(counts.values())
        leader = max(counts.values(), default=0)
        word_count = max(len(body_word_offsets), 1)
        lead_end = (
            body_word_offsets[config.lead_words]
            if len(body_word_offsets) > config.lead_words
            else float("inf")
        )
        title_order = list(dict.fromkeys(m.ticker for m in title_mentions))
        tagged = {m.ticker for m in body_mentions if m.exchange_tagged}
        first_offset: dict[str, int] = {}
        for mention in body_mentions:
            first_offset.setdefault(mention.ticker, mention.start)

        results = {}
        for ticker in dict.fromkeys([*title_order, *counts]):
            features = SalienceFeatures(
                in_title=ticker in title_order,
                title_subject=title_order[:1] == [ticker],
                in_lead=first_offset.get(ticker, float("inf")) < lead_end,
                exchange_tagged=ticker in tagged,
                mentions=counts[ticker],
                mentions_per_100_words=round(counts[ticker] * 100 / word_count, 3),
                share=round(counts[ticker] / total, 3) if total else 0.0,
                relative_share=round(counts[ticker] / leader, 3) if leader else 0.0,
            )
            results[ticker] = SalienceResult(self._weigh(features), features)
        return results

    def _weigh(self, f: SalienceFeatures) -> float:
        c = self._config
        frequency = (
            min(f.mentions_per_100_words / c.frequency_cap_per_100_words, 1.0)
            if f.mentions >= c.frequency_min_mentions
            else 0.0
        )
        title = c.title_subject_weight if f.title_subject else c.title_weight * f.in_title
        score = (
            title
            + c.lead_weight * f.in_lead
            + c.exchange_tagged_weight * f.exchange_tagged
            + c.frequency_weight * frequency
            + c.share_weight * f.relative_share
        )
        return round(min(score, 1.0), 3)
