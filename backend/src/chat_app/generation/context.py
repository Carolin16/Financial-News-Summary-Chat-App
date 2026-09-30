"""Numbers the found chunks [1], [2] for the LLM may use."""

from collections.abc import Sequence

from pydantic import BaseModel

from chat_app.core.models import Citation, RetrievedChunk


class Source(BaseModel):
    """A retrieved chunk with the citation number the LLM uses for it."""

    number: int
    retrieved: RetrievedChunk

    @property
    def citation(self) -> Citation:
        """User-facing link for this source."""
        chunk = self.retrieved.chunk
        return Citation(
            article_id=chunk.article_id, title=chunk.title, link=chunk.link, is_stub=chunk.is_stub
        )

    @property
    def text(self) -> str:
        """Everything the LLM sees for this source (header + body)."""
        return self.retrieved.chunk.contextualized_text


def number_sources(retrieved: Sequence[RetrievedChunk]) -> list[Source]:
    """Assign citation numbers 1..n in rank order."""
    return [Source(number=i, retrieved=r) for i, r in enumerate(retrieved, start=1)]


def format_sources(sources: Sequence[Source], focus_tickers: Sequence[str]) -> str:
    """Render sources for the prompt, labelling partial articles and passing mentions."""
    return "\n\n".join(_format_source(s, focus_tickers) for s in sources)


def _format_source(source: Source, focus_tickers: Sequence[str]) -> str:
    """Write one source as "[n] (warnings)" followed by its text, e.g. "[2] (PARTIAL)"."""
    chunk = source.retrieved.chunk
    labels = []
    if chunk.is_stub:
        labels.append("PARTIAL")
    if focus_tickers and not set(focus_tickers) & set(chunk.metadata.primary_tickers):
        labels.append("mentions the company only in passing")
    label = f" ({'; '.join(labels)})" if labels else ""
    return f"[{source.number}]{label}\n{source.text}"
