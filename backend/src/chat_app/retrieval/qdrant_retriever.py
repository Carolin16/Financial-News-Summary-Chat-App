"""Finds the chunks that best match a question, searching by meaning and by keywords at once."""

from qdrant_client import AsyncQdrantClient, models

from chat_app.core.interfaces import EmbeddingProvider
from chat_app.core.models import RetrievedChunk, SearchFilters
from chat_app.retrieval.encoders import Bm25SparseEncoder
from chat_app.retrieval.filters import build_filter
from chat_app.retrieval.qdrant_schema import DENSE_VECTOR, SPARSE_VECTOR, payload_to_chunk


class QdrantHybridRetriever:
    """Runs a meaning search and a keyword search in Qdrant and merges their results."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        embedder: EmbeddingProvider,
        sparse_encoder: Bm25SparseEncoder,
        prefetch_k: int,
    ) -> None:
        """Take the Qdrant connection, both encoders, and how many results each search gathers."""
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._prefetch_k = prefetch_k

    async def retrieve(
        self, query: str, filters: SearchFilters, top_k: int
    ) -> list[RetrievedChunk]:
        """Return the best `top_k` chunks, ranked by how well they do in both searches."""
        # Turn the question into a meaning vector (finds paraphrases, e.g. "jumped" ~ "surged").
        dense_query = await self._embedder.aembed_query(query)
        # And into a keyword vector (catches exact words and numbers, e.g. "DBS", "$160").
        sparse_query = self._sparse.encode_query(query)
        # Limit both searches to the right chunks, e.g. only articles about Intel.
        qdrant_filter = build_filter(filters)
        response = await self._client.query_points(
            collection_name=self._collection,
            # Run both searches, each gathering its own shortlist of candidates.
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using=DENSE_VECTOR,
                    filter=qdrant_filter,
                    limit=self._prefetch_k,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using=SPARSE_VECTOR,
                    filter=qdrant_filter,
                    limit=self._prefetch_k,
                ),
            ],
            # Merge the two shortlists: chunks ranked high in either or both come out on top.
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            # Bring back each chunk's text and tags, not just its ID.
            with_payload=True,
        )
        # Turn Qdrant's results back into the app's own chunk objects.
        return [
            RetrievedChunk(chunk=payload_to_chunk(point.payload or {}), score=point.score)
            for point in response.points
        ]
