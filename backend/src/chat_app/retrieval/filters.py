"""Translates domain `SearchFilters` into a Qdrant filter."""

from qdrant_client import models

from chat_app.core.models import SearchFilters


def build_filter(filters: SearchFilters) -> models.Filter | None:
    """Return a Qdrant filter, or None when no constraint applies."""
    must: list[models.Condition] = []
    if filters.tickers:
        ticker_match = models.MatchAny(any=filters.tickers)
        primary = models.FieldCondition(key="metadata.primary_tickers", match=ticker_match)
        if filters.primary_only:
            must.append(primary)
        else:
            mentioned = models.FieldCondition(key="metadata.mentioned_tickers", match=ticker_match)
            must.append(models.Filter(should=[primary, mentioned]))
    if not filters.include_stubs:
        must.append(models.FieldCondition(key="is_stub", match=models.MatchValue(value=False)))
    if filters.event_types:
        must.append(
            models.FieldCondition(
                key="metadata.event_types",
                match=models.MatchAny(any=[e.value for e in filters.event_types]),
            )
        )
    if filters.sentiment is not None:
        must.append(
            models.FieldCondition(
                key="metadata.sentiment", match=models.MatchValue(value=filters.sentiment.value)
            )
        )
    if filters.article_type is not None:
        must.append(
            models.FieldCondition(
                key="metadata.article_type",
                match=models.MatchValue(value=filters.article_type.value),
            )
        )
    return models.Filter(must=must) if must else None
