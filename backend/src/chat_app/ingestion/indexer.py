"""Writes chunks to Qdrant: idempotent (deterministic IDs) and incremental (skip unchanged)."""

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from chat_app.core.interfaces import EmbeddingProvider
from chat_app.core.models import Chunk
from chat_app.retrieval.encoders import Bm25SparseEncoder
from chat_app.retrieval.qdrant_schema import (
    CONTENT_HASH_FIELD,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    chunk_to_payload,
    ensure_collection,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexReport:
    """Outcome of one indexing run."""

    written: int
    unchanged: int
    stale_removed: int


class QdrantChunkIndex:
    """`ChunkIndex` that stores dense + BM25 vectors side by side for hybrid search."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedder: EmbeddingProvider,
        sparse_encoder: Bm25SparseEncoder,
    ) -> None:
        """Bind the index to a collection; the collection is created on first use."""
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._sparse = sparse_encoder

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """Write new or changed chunks; return how many were (re)embedded."""
        return self.sync(chunks).written

    def sync(self, chunks: Sequence[Chunk]) -> IndexReport:
        """Make the collection match `chunks`: embed only what changed, drop stale points."""
        ensure_collection(self._client, self._collection, self._embedder.dimensions)
        hashes = {c.chunk_id: _content_hash(c) for c in chunks}
        stored = self._stored_hashes(list(hashes))
        changed = [c for c in chunks if stored.get(c.chunk_id) != hashes[c.chunk_id]]
        if changed:
            self._write(changed, hashes)
        removed = self._remove_stale(chunks)
        report = IndexReport(len(changed), len(chunks) - len(changed), removed)
        logger.info("index sync: %s", report)
        return report

    def _write(self, chunks: Sequence[Chunk], hashes: dict[str, str]) -> None:
        texts = [c.contextualized_text for c in chunks]
        dense = self._embedder.embed(texts)
        sparse = self._sparse.encode_documents(texts)
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={DENSE_VECTOR: dense_vector, SPARSE_VECTOR: sparse_vector},
                payload=chunk_to_payload(chunk, hashes[chunk.chunk_id]),
            )
            for chunk, dense_vector, sparse_vector in zip(chunks, dense, sparse, strict=True)
        ]
        self._client.upsert(self._collection, points=points, wait=True)

    def _stored_hashes(self, ids: list[str]) -> dict[str, str]:
        records = self._client.retrieve(
            self._collection, ids=ids, with_payload=[CONTENT_HASH_FIELD], with_vectors=False
        )
        return {str(r.id): str((r.payload or {}).get(CONTENT_HASH_FIELD)) for r in records}

    def _remove_stale(self, chunks: Sequence[Chunk]) -> int:
        """Delete points whose article left the dataset or now has fewer chunks."""
        keep = {c.chunk_id for c in chunks}
        stale: list[models.ExtendedPointId] = []
        offset: models.ExtendedPointId | None = None
        while True:
            records, offset = self._client.scroll(
                self._collection, offset=offset, with_payload=False, with_vectors=False
            )
            stale.extend(r.id for r in records if str(r.id) not in keep)
            if offset is None:
                break
        if stale:
            self._client.delete(
                self._collection, points_selector=models.PointIdsList(points=stale), wait=True
            )
        return len(stale)


def _content_hash(chunk: Chunk) -> str:
    # Metadata is part of the hash so re-enrichment (e.g. new tickers) refreshes payloads.
    material = chunk.contextualized_text + chunk.metadata.model_dump_json() + str(chunk.is_stub)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
