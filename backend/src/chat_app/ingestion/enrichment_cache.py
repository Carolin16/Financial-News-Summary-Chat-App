"""Persistent cache for extracted metadata.

Enrichment is the only LLM step in ingestion, so caching it (keyed by article ID and a hash
of the content) makes re-indexing free, deterministic, and runnable without an API key.
"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from chat_app.core.interfaces import MetadataExtractor
from chat_app.core.models import Article, ArticleMetadata


class _CacheEntry(BaseModel):
    content_hash: str
    metadata: ArticleMetadata


class CachedMetadataExtractor:
    """Wraps an extractor; returns cached metadata while an article's content is unchanged."""

    def __init__(self, inner: MetadataExtractor, path: Path) -> None:
        """Load any existing cache from `path`; call `save()` to persist new entries."""
        self._inner = inner
        self._path = path
        self._entries = self._load()
        self.hits = 0
        self.misses = 0

    def extract(self, article: Article) -> ArticleMetadata:
        """Return cached metadata, or extract, remember, and return it."""
        content_hash = _content_hash(article)
        entry = self._entries.get(article.article_id)
        if entry is not None and entry.content_hash == content_hash:
            self.hits += 1
            return entry.metadata
        self.misses += 1
        metadata = self._inner.extract(article)
        self._entries[article.article_id] = _CacheEntry(
            content_hash=content_hash, metadata=metadata
        )
        return metadata

    def save(self) -> None:
        """Write the cache as stable, sorted JSON so diffs stay reviewable."""
        payload = {
            key: entry.model_dump(mode="json") for key, entry in sorted(self._entries.items())
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> dict[str, _CacheEntry]:
        if not self._path.exists():
            return {}
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return {key: _CacheEntry.model_validate(value) for key, value in raw.items()}


def _content_hash(article: Article) -> str:
    return hashlib.sha256(f"{article.title}\n{article.text}".encode()).hexdigest()
