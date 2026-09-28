"""Hybrid retrieval: dense (cosine) + BM25 sparse, fused server-side with RRF."""

from qdrant_client import AsyncQdrantClient, models

from chat_app.core.interfaces import EmbeddingProvider
from chat_app.core.models import RetrievedChunk, SearchFilters
from chat_app.retrieval.encoders import Bm25SparseEncoder
from chat_app.retrieval.filters import build_filter
from chat_app.retrieval.qdrant_schema import DENSE_VECTOR, SPARSE_VECTOR, payload_to_chunk


class QdrantHybridRetriever:
    """`Retriever` that runs both searches under the same metadata filter in one request."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        embedder: EmbeddingProvider,
        sparse_encoder: Bm25SparseEncoder,
        prefetch_k: int,
    ) -> None:
        """`prefetch_k` is how many candidates each search contributes to the fusion."""
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._prefetch_k = prefetch_k

    async def retrieve(
        self, query: str, filters: SearchFilters, top_k: int
    ) -> list[RetrievedChunk]:
        """Return up to `top_k` chunks ranked by reciprocal rank fusion."""
        dense_query = await self._embedder.aembed_query(query)
        sparse_query = self._sparse.encode_query(query)
        qdrant_filter = build_filter(filters)
        response = await self._client.query_points(
            collection_name=self._collection,
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
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievedChunk(chunk=payload_to_chunk(point.payload or {}), score=point.score)
            for point in response.points
        ]
