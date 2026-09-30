"""Splits articles into search-sized pieces, each labelled with its title and companies."""

from collections.abc import Callable

import tiktoken

from chat_app.core.ids import chunk_point_id
from chat_app.core.models import HEADER_SEPARATOR, Article, Chunk
from chat_app.ingestion.sentences import split_sentences

STUB_NOTE = "Note: partial article (paywalled or teaser); treat as low-confidence."


class Chunker:
    """Splits an article into chunks."""

    def __init__(
        self,
        max_tokens: int,
        overlap_sentences: int,
        encoding_name: str,
        company_name: Callable[[str], str],
    ) -> None:
        """Set the chunk size limit, the overlap, and how to show company names."""
        self._max_tokens = max_tokens
        self._overlap = overlap_sentences
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._company_name = company_name

    def chunk(self, article: Article) -> list[Chunk]:
        """Return the article as one chunk if it fits, otherwise as several, in reading order."""
        header = self.build_header(article)
        body_budget = self._max_tokens - self._count(header + HEADER_SEPARATOR)
        if self._count(article.text) <= body_budget:
            bodies = [article.text]
        else:
            bodies = self._pack(split_sentences(article.text), body_budget)
        return [
            Chunk(
                chunk_id=chunk_point_id(article.article_id, index),
                article_id=article.article_id,
                chunk_index=index,
                title=article.title,
                link=article.link,
                text=body,
                header=header,
                is_stub=article.is_stub,
                metadata=article.metadata,
            )
            for index, body in enumerate(bodies)
        ]

    def build_header(self, article: Article) -> str:
        """Build the label (title, companies, stub note) put on top of every chunk."""
        lines = [f"Title: {article.title}"]
        if article.metadata.primary_tickers:
            lines.append(f"About: {self._describe(article.metadata.primary_tickers)}")
        if article.metadata.mentioned_tickers:
            lines.append(f"Also mentions: {self._describe(article.metadata.mentioned_tickers)}")
        if article.is_stub:
            lines.append(STUB_NOTE)
        return "\n".join(lines)

    def _describe(self, tickers: list[str]) -> str:
        """Turn tickers into readable text, e.g. "Intel (INTC), Nvidia (NVDA)"."""
        return ", ".join(f"{self._company_name(t)} ({t})" for t in tickers)

    def _pack(self, sentences: list[str], budget: int) -> list[str]:
        """Fill each chunk with whole sentences until the next one would not fit."""
        units = [piece for s in sentences for piece in self._fit_sentence(s, budget)]
        chunks: list[list[str]] = []
        current: list[str] = []
        for unit in units:
            if current and self._count(" ".join([*current, unit])) > budget:
                chunks.append(current)
                current = self._overlap_tail(current, unit, budget)
            current.append(unit)
        if current:
            chunks.append(current)
        return [" ".join(parts) for parts in chunks]

    def _overlap_tail(self, previous: list[str], next_unit: str, budget: int) -> list[str]:
        """Repeat the last sentence of the previous chunk so facts split across chunks survive."""
        tail = previous[-self._overlap :] if self._overlap else []
        # Skip the overlap if there would be no room left for the next sentence.
        if tail and self._count(" ".join([*tail, next_unit])) > budget:
            return []
        return tail

    def _fit_sentence(self, sentence: str, budget: int) -> list[str]:
        """Break a sentence too long for one chunk into smaller pieces by words."""
        if self._count(sentence) <= budget:
            return [sentence]
        pieces: list[str] = []
        current: list[str] = []
        for word in sentence.split():
            if current and self._count(" ".join([*current, word])) > budget:
                pieces.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _count(self, text: str) -> int:
        """Count tokens the same way the embedding model does."""
        return len(self._encoding.encode(text))
