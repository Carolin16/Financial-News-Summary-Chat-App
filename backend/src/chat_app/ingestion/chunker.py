"""Sentence-aware recursive chunking with contextual headers.

Short articles (the common case) stay whole. Longer ones are packed sentence by sentence up
to the token budget, with a small sentence overlap so a fact split across a boundary is
still retrievable; a single over-long sentence falls back to word-level splitting.
"""

from collections.abc import Callable

import tiktoken

from chat_app.core.ids import chunk_point_id
from chat_app.core.models import HEADER_SEPARATOR, Article, Chunk
from chat_app.ingestion.sentences import split_sentences

STUB_NOTE = "Note: partial article (paywalled or teaser); treat as low-confidence."


class Chunker:
    """Splits an article into `Chunk`s that each carry the article's context and metadata."""

    def __init__(
        self,
        max_tokens: int,
        overlap_sentences: int,
        encoding_name: str,
        company_name: Callable[[str], str],
    ) -> None:
        """Configure the token budget (header included) and how tickers are displayed."""
        self._max_tokens = max_tokens
        self._overlap = overlap_sentences
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._company_name = company_name

    def chunk(self, article: Article) -> list[Chunk]:
        """Return the article's chunks in reading order."""
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
        """Context prepended to every chunk so each is interpretable on its own.

        Titles matter: paywalled stubs often carry their key fact (e.g. a new price
        target) only in the headline.
        """
        lines = [f"Title: {article.title}"]
        if article.metadata.primary_tickers:
            lines.append(f"About: {self._describe(article.metadata.primary_tickers)}")
        if article.metadata.mentioned_tickers:
            lines.append(f"Also mentions: {self._describe(article.metadata.mentioned_tickers)}")
        if article.is_stub:
            lines.append(STUB_NOTE)
        return "\n".join(lines)

    def _describe(self, tickers: list[str]) -> str:
        return ", ".join(f"{self._company_name(t)} ({t})" for t in tickers)

    def _pack(self, sentences: list[str], budget: int) -> list[str]:
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
        tail = previous[-self._overlap :] if self._overlap else []
        # Drop the overlap if it would not leave room for the next unit.
        if tail and self._count(" ".join([*tail, next_unit])) > budget:
            return []
        return tail

    def _fit_sentence(self, sentence: str, budget: int) -> list[str]:
        """Split a sentence that alone exceeds the budget into word windows."""
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
        return len(self._encoding.encode(text))
