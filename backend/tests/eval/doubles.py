"""A `Retriever` double that edits or removes real passages, for context-dependence tests.

If an answer changes when its source is edited, and stops stating a fact when its source is
removed, it came from the corpus. If it doesn't, the model answered from memory.
"""

from collections.abc import Collection, Sequence
from typing import Protocol

from cases import Ablation, Counterfactual, span_pattern

from chat_app.core.interfaces import Retriever
from chat_app.core.models import Chunk, RetrievedChunk, SearchFilters


class PassageTransform(Protocol):
    """Changes one retrieved passage, or returns None to remove it."""

    def apply(self, chunk: Chunk) -> Chunk | None:
        """The transformed chunk, or None to drop it from the results."""
        ...


class ReplaceSpan:
    """Swap a span for another in the title, contextual header and body."""

    def __init__(self, old: str, new: str) -> None:
        """Match `old` as a whole token (see `span_pattern`) and replace it with `new`."""
        self._pattern = span_pattern(old)
        self._new = new

    def apply(self, chunk: Chunk) -> Chunk:
        """The chunk with every occurrence replaced."""
        return chunk.model_copy(
            update={
                "title": self._replace(chunk.title),
                "header": self._replace(chunk.header),
                "text": self._replace(chunk.text),
            }
        )

    def _replace(self, text: str) -> str:
        # A function replacement keeps "$" and "\" in `new` literal.
        return self._pattern.sub(lambda _: self._new, text)


class DropLinks:
    """Remove every passage from the given articles."""

    def __init__(self, links: Collection[str]) -> None:
        """`links` are article URLs as stored on each chunk."""
        self._links = frozenset(links)

    def apply(self, chunk: Chunk) -> Chunk | None:
        """None for passages from a dropped article."""
        return None if chunk.link in self._links else chunk


class TransformingRetriever:
    """Wraps a real `Retriever` and applies transforms to what it returns."""

    def __init__(self, inner: Retriever, transforms: Sequence[PassageTransform]) -> None:
        """Transforms run in order; a None result drops the passage."""
        self._inner = inner
        self._transforms = transforms
        # Passages changed or dropped so far: zero means the case never touched its target.
        self.touched = 0

    async def retrieve(
        self, query: str, filters: SearchFilters, top_k: int
    ) -> list[RetrievedChunk]:
        """The inner results, transformed, with dropped passages removed."""
        results = []
        for retrieved in await self._inner.retrieve(query, filters, top_k):
            chunk: Chunk | None = retrieved.chunk
            for transform in self._transforms:
                if chunk is None:
                    break
                chunk = transform.apply(chunk)
            if chunk != retrieved.chunk:
                self.touched += 1
            if chunk is not None:
                results.append(retrieved.model_copy(update={"chunk": chunk}))
        return results


def counterfactual_transforms(counterfactual: Counterfactual) -> list[PassageTransform]:
    """The span replacements a counterfactual case describes."""
    return [ReplaceSpan(r.old, r.new) for r in counterfactual.replacements]


def ablation_transforms(ablation: Ablation) -> list[PassageTransform]:
    """The link removal an ablation case describes."""
    return [DropLinks(ablation.drop_links)]
