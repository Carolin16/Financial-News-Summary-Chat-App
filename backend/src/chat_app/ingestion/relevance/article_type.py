"""Classifies an article's editorial form from title and lead cues (first rule wins)."""

from dataclasses import dataclass
from typing import Protocol

from chat_app.config.relevance import ArticleTypeConfig
from chat_app.core.models import ArticleType


@dataclass(frozen=True)
class TypeVerdict:
    """The type and the exact text that triggered it (None for the default, news)."""

    article_type: ArticleType
    cue: str | None


class ArticleTypeClassifier(Protocol):
    """Decides an article's editorial form."""

    def classify(self, title: str, text: str) -> TypeVerdict:
        """Type of the article with the given title and body."""
        ...


class CueArticleTypeClassifier:
    """Ordered cue rules from configuration; anything unmatched is news."""

    def __init__(self, config: ArticleTypeConfig) -> None:
        """Rules are checked in file order, which is their priority."""
        self._config = config

    def classify(self, title: str, text: str) -> TypeVerdict:
        """Return the first rule whose cue matches the title (or the lead, if allowed)."""
        lead = text[: self._config.lead_chars]
        for rule in self._config.rules:
            places = (title, lead) if rule.in_lead else (title,)
            for place in places:
                for pattern in rule.regexes:
                    if match := pattern.search(place):
                        return TypeVerdict(ArticleType(rule.label), match.group(0))
        return TypeVerdict(ArticleType.NEWS, None)
