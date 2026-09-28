"""Deterministic identifiers, so re-running indexing overwrites instead of duplicating."""

import hashlib
import uuid

# Fixed namespace: changing it would orphan every point already stored in Qdrant.
_POINT_NAMESPACE = uuid.UUID("6f1c3a52-9d0e-4f5b-8a7e-2c4b1d9e0a31")
_ARTICLE_ID_LENGTH = 16


def article_id_for(link: str) -> str:
    """Stable short ID for an article, derived from its canonical link."""
    return hashlib.sha256(link.strip().encode("utf-8")).hexdigest()[:_ARTICLE_ID_LENGTH]


def chunk_point_id(article_id: str, chunk_index: int) -> str:
    """Stable Qdrant point ID (UUID string) for one chunk of an article."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{article_id}:{chunk_index}"))
