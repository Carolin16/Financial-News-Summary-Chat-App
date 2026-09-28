"""Qdrant client construction: a server URL, or embedded local mode for Docker-free runs."""

from qdrant_client import AsyncQdrantClient, QdrantClient

from chat_app.config.settings import Settings


def make_client(settings: Settings) -> QdrantClient:
    """Sync client for offline indexing."""
    if settings.qdrant_local_path is not None:
        return QdrantClient(path=str(settings.qdrant_local_path))
    return QdrantClient(url=settings.qdrant_url, timeout=int(settings.llm_timeout_seconds))


def make_async_client(settings: Settings) -> AsyncQdrantClient:
    """Async client for the API."""
    if settings.qdrant_local_path is not None:
        return AsyncQdrantClient(path=str(settings.qdrant_local_path))
    return AsyncQdrantClient(url=settings.qdrant_url, timeout=int(settings.llm_timeout_seconds))
