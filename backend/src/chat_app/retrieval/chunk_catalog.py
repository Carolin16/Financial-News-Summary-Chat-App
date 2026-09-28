"""Reads every stored chunk; used to build the coverage index at API startup."""

from qdrant_client import AsyncQdrantClient

from chat_app.core.models import Chunk
from chat_app.retrieval.qdrant_schema import payload_to_chunk

_SCROLL_PAGE = 256


async def load_chunks(client: AsyncQdrantClient, collection: str) -> list[Chunk]:
    """Return all chunks in the collection (empty if it does not exist yet)."""
    if not await client.collection_exists(collection):
        return []
    chunks: list[Chunk] = []
    offset = None
    while True:
        records, offset = await client.scroll(
            collection, offset=offset, limit=_SCROLL_PAGE, with_payload=True, with_vectors=False
        )
        chunks.extend(payload_to_chunk(r.payload or {}) for r in records)
        if offset is None:
            return chunks
