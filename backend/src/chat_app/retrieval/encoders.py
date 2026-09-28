"""Text encoders: OpenAI dense embeddings and a local BM25 sparse encoder."""

from collections.abc import Sequence

from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI, OpenAI, OpenAIError
from qdrant_client import models

from chat_app.config.settings import Settings


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced after the SDK's retries."""


class OpenAIEmbeddingProvider:
    """`EmbeddingProvider` backed by the OpenAI embeddings endpoint."""

    def __init__(self, settings: Settings) -> None:
        """Create clients with the configured timeout/retry policy and batch size."""
        options = {
            "api_key": settings.openai_api_key.get_secret_value(),
            "base_url": settings.openai_base_url,
            "timeout": settings.llm_timeout_seconds,
            "max_retries": settings.llm_max_retries,
        }
        self._sync = OpenAI(**options)  # type: ignore[arg-type]
        self._async = AsyncOpenAI(**options)  # type: ignore[arg-type]
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions
        self._batch_size = settings.embedding_batch_size

    @property
    def dimensions(self) -> int:
        """Vector length, fixed by configuration so the collection schema matches."""
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches (offline indexing)."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            try:
                response = self._sync.embeddings.create(
                    model=self._model, input=batch, dimensions=self._dimensions
                )
            except OpenAIError as error:
                raise EmbeddingError(str(error)) from error
            vectors.extend(item.embedding for item in response.data)
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query at request time."""
        try:
            response = await self._async.embeddings.create(
                model=self._model, input=[text], dimensions=self._dimensions
            )
        except OpenAIError as error:
            raise EmbeddingError(str(error)) from error
        return response.data[0].embedding


class Bm25SparseEncoder:
    """BM25 term-frequency vectors; Qdrant applies IDF server-side at query time.

    Runs locally (no API call), which is what lets exact tokens such as "DBS" or "$160"
    match even when the dense embedding blurs them.
    """

    def __init__(self, model_name: str) -> None:
        """Load the fastembed BM25 model (tokeniser + stemmer, no neural weights)."""
        self._model = SparseTextEmbedding(model_name=model_name)

    def encode_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        """Encode indexed text."""
        return [
            models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._model.embed(list(texts))
        ]

    def encode_query(self, text: str) -> models.SparseVector:
        """Encode a query; BM25 query vectors weight each term once."""
        embedding = next(iter(self._model.query_embed(text)))
        return models.SparseVector(
            indices=embedding.indices.tolist(), values=embedding.values.tolist()
        )
