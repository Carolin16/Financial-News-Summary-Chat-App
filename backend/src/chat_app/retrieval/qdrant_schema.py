"""Collection layout shared by the offline indexer and the query-time retriever."""

from qdrant_client import AsyncQdrantClient, QdrantClient, models

from chat_app.core.models import Chunk

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
CONTENT_HASH_FIELD = "content_hash"

# Payload fields with an index, so metadata filters stay fast as the collection grows.
KEYWORD_INDEXES = (
    "article_id",
    "metadata.primary_tickers",
    "metadata.mentioned_tickers",
    "metadata.event_types",
    "metadata.sentiment",
    "metadata.article_type",
)
BOOL_INDEXES = ("is_stub",)


def ensure_collection(client: QdrantClient, name: str, dense_dimensions: int) -> None:
    """Create the hybrid collection and payload indexes if missing (idempotent)."""
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=dense_dimensions, distance=models.Distance.COSINE
            )
        },
        # IDF modifier turns stored term frequencies into BM25 scores at query time.
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    for field in KEYWORD_INDEXES:
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
    for field in BOOL_INDEXES:
        client.create_payload_index(name, field, models.PayloadSchemaType.BOOL)


async def collection_ready(client: AsyncQdrantClient, name: str) -> bool:
    """True if the collection exists and holds at least one point (used by /health)."""
    if not await client.collection_exists(name):
        return False
    return (await client.count(name, exact=False)).count > 0


def chunk_to_payload(chunk: Chunk, content_hash: str) -> dict[str, object]:
    """Serialise a chunk for storage; the hash enables skip-if-unchanged re-indexing."""
    return {**chunk.model_dump(mode="json"), CONTENT_HASH_FIELD: content_hash}


def payload_to_chunk(payload: dict[str, object]) -> Chunk:
    """Rebuild a chunk from a stored payload (extra fields are ignored)."""
    return Chunk.model_validate(payload)
