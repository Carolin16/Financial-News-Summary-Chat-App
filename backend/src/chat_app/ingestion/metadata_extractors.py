"""Content-based metadata extraction: which companies an article is really about.

Composable strategies behind the `MetadataExtractor` protocol:
- `LlmMetadataExtractor`: the primary judge, reading the article content.
- `HeuristicMetadataExtractor`: offline fallback that counts company mentions.
- `FallbackMetadataExtractor`: tries one strategy and degrades to another on failure.
"""

import logging

from pydantic import ValidationError

from chat_app.core.interfaces import MetadataExtractor, StructuredLlm
from chat_app.core.models import Article, ArticleMetadata
from chat_app.core.ticker_registry import TickerRegistry
from chat_app.generation.llm_client import LlmError
from chat_app.generation.prompts import load_prompt

logger = logging.getLogger(__name__)

_PROMPT_NAME = "metadata_extraction"


class LlmMetadataExtractor:
    """Asks the LLM to label an article, then canonicalises tickers via the registry."""

    def __init__(self, llm: StructuredLlm, registry: TickerRegistry, max_chars: int) -> None:
        """`max_chars` caps the article text sent, bounding cost on very long articles."""
        self._llm = llm
        self._registry = registry
        self._max_chars = max_chars

    def extract(self, article: Article) -> ArticleMetadata:
        """Return LLM-judged metadata for `article`."""
        companies = ", ".join(f"{t} ({self._registry.name_for(t)})" for t in self._registry.tickers)
        instructions = load_prompt(_PROMPT_NAME).format(companies=companies)
        prompt = f"Title: {article.title}\n\n{article.text[: self._max_chars]}"
        metadata = self._llm.parse(instructions, prompt, ArticleMetadata)
        return canonicalize_tickers(metadata, self._registry)


class HeuristicMetadataExtractor:
    """Mention-counting fallback: primary if named in the title or mentioned often."""

    def __init__(self, registry: TickerRegistry, min_primary_mentions: int) -> None:
        """`min_primary_mentions` is how many body mentions make a company primary."""
        self._registry = registry
        self._min_primary_mentions = min_primary_mentions

    def extract(self, article: Article) -> ArticleMetadata:
        """Return metadata inferred from company mentions; tone and type stay neutral."""
        in_title = set(self._registry.find_mentions(article.title))
        counts = self._registry.mention_counts(article.text)
        primary = [
            t
            for t in self._registry.find_mentions(f"{article.title} {article.text}")
            if t in in_title or counts[t] >= self._min_primary_mentions
        ]
        mentioned = [t for t in counts if t not in primary]
        return ArticleMetadata(primary_tickers=primary, mentioned_tickers=mentioned)


class FallbackMetadataExtractor:
    """Uses `primary`, and `fallback` when the primary strategy fails."""

    def __init__(self, primary: MetadataExtractor, fallback: MetadataExtractor) -> None:
        """Compose two strategies; the fallback must not depend on external services."""
        self._primary = primary
        self._fallback = fallback

    def extract(self, article: Article) -> ArticleMetadata:
        """Return primary metadata, degrading gracefully on LLM or validation errors."""
        try:
            return self._primary.extract(article)
        except (LlmError, ValidationError) as error:
            logger.warning("metadata fallback for %s: %s", article.article_id, error)
            return self._fallback.extract(article)


def canonicalize_tickers(metadata: ArticleMetadata, registry: TickerRegistry) -> ArticleMetadata:
    """Map aliases (GOOG, Google) to registry tickers and keep primary/mentioned disjoint."""
    primary = list(dict.fromkeys(registry.canonical(t) for t in metadata.primary_tickers))
    mentioned = [
        t
        for t in dict.fromkeys(registry.canonical(t) for t in metadata.mentioned_tickers)
        if t not in primary
    ]
    return metadata.model_copy(update={"primary_tickers": primary, "mentioned_tickers": mentioned})
