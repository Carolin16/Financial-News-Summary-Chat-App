"""Finds every company an article mentions, and where, so relevance can be scored."""

import re
from dataclasses import dataclass, field
from typing import Protocol

from chat_app.config.relevance import EntityConfig
from chat_app.core.ticker_registry import MentionForm, TickerRegistry


@dataclass(frozen=True)
class EntityMention:
    """One place a company is mentioned, noting if it was written formally, e.g. (NASDAQ:INTC)."""

    ticker: str
    start: int
    end: int
    form: MentionForm
    exchange_tagged: bool = False


@dataclass(frozen=True)
class ArticleMentions:
    """All company mentions in the title and body, plus company names picked up from the text."""

    title: list[EntityMention]
    body: list[EntityMention]
    learned_names: dict[str, str] = field(default_factory=dict)  # name -> ticker


class EntityMatcher(Protocol):
    """Anything that can list the companies an article mentions."""

    def find(self, title: str, text: str) -> ArticleMentions:
        """Return the title and body mentions in reading order."""
        ...


class RegistryEntityMatcher:
    """Finds tracked companies by name, product, or ticker, plus others written formally."""

    def __init__(self, registry: TickerRegistry, config: EntityConfig) -> None:
        """Build the patterns for formal company references from the rules file."""
        self._registry = registry
        self._config = config
        exchanges = "|".join(map(re.escape, config.exchanges))
        name = r"(?P<name>(?:[A-Z][\w.&'-]*\s+){0,4}?[A-Z][\w.&'-]*)"
        # "Walmart (NYSE:WMT)": the exchange tag is reliable on its own; a name is optional.
        self._tagged = re.compile(
            rf"(?:{name}\s*)?(?P<tag>\((?:{exchanges})\s*:\s*(?P<sym>[A-Z]{{1,5}}(?:[.-][A-Z])?)\))"
        )
        # "Cerence (CRNC)" / "Apple (AAPL, Financials)": the symbol must follow a capitalised
        # name, which rules out "deep ultraviolet (DUV)".
        self._paren = re.compile(
            rf"{name}\s*(?P<tag>\((?P<sym>[A-Z]{{2,5}}(?:\.[A-Z])?)(?:,\s*[A-Z][\w ]{{0,20}})?\))"
        )
        # Zacks style, no parentheses: "Cerence CRNC shares", "Walmart Inc. WMT is". Accepted
        # only if the bare symbol recurs (see bare_symbol_min_count), to avoid "Nvidia GPU".
        self._bare = re.compile(
            rf"{name}\s+(?P<tag>(?P<sym>[A-Z]{{2,5}}))(?=\s+[a-z]|['’][s ]|[.,;:])"
        )
        self._not_tickers = {t.upper() for t in config.not_tickers}
        self._suffixes = {s.lower() for s in config.name_suffixes}
        self._leading = {s.lower() for s in config.name_leading_words}
        self._generic = {s.lower() for s in config.name_generic_words}

    def find(self, title: str, text: str) -> ArticleMentions:
        """Find formal and registry mentions, then search for any newly learned names too."""
        body, learned = self._formal_mentions(text)
        body.extend(self._registry_mentions(text))
        title_mentions = self._registry_mentions(title)
        for name, ticker in learned.items():
            for pattern in (
                # Learned names match case-insensitively: "EnPro" in text, "Enpro" in a title.
                re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])", re.IGNORECASE),
                re.compile(rf"(?<![\w$.-])\$?{re.escape(ticker)}\b"),  # bare "CRNC stock"
            ):
                body.extend(_as_mentions(pattern, text, ticker))
                title_mentions.extend(_as_mentions(pattern, title, ticker))
        return ArticleMentions(
            title=_drop_overlaps(title_mentions),
            body=_drop_overlaps(body),
            learned_names=learned,
        )

    def _registry_mentions(self, text: str) -> list[EntityMention]:
        """Mentions of tracked companies (Apple, iPhone, AAPL, ...)."""
        return [
            EntityMention(m.ticker, m.start, m.end, m.form) for m in self._registry.mentions(text)
        ]

    def _formal_mentions(self, text: str) -> tuple[list[EntityMention], dict[str, str]]:
        """Formal references like "Walmart (NYSE:WMT)", and the company names learned from them."""
        mentions: list[EntityMention] = []
        learned: dict[str, str] = {}
        seen: set[tuple[int, int]] = set()
        for pattern in (self._tagged, self._paren, self._bare):
            for match in pattern.finditer(text):
                span = match.span("tag")
                symbol = self._registry.canonical(match.group("sym"))
                words = (match.group("name") or "").split()
                if span in seen or symbol in self._not_tickers or _is_acronym(symbol, words):
                    continue
                if pattern is self._bare and not (
                    self._recurs(symbol, text) or self._has_legal_suffix(words)
                ):
                    continue
                seen.add(span)
                mentions.append(
                    EntityMention(symbol, *span, MentionForm.TICKER, exchange_tagged=True)
                )
                if symbol not in self._registry.tickers:
                    for name in self._company_names(words):
                        learned.setdefault(name, symbol)
        return mentions, learned

    def _recurs(self, symbol: str, text: str) -> bool:
        """True if a bare symbol appears often enough to trust it is a ticker."""
        occurrences = re.findall(rf"(?<![\w$.-])\$?{re.escape(symbol)}\b", text)
        return len(occurrences) >= self._config.bare_symbol_min_count

    def _has_legal_suffix(self, words: list[str]) -> bool:
        """True if the name ends in "Inc.", "Corp." or similar, marking it as a company."""
        return bool(words) and words[-1].lower().strip(",") in self._suffixes

    def _company_names(self, words: list[str]) -> list[str]:
        """Turn "Shares of CyberArk Software Inc." into "CyberArk Software" and "CyberArk"."""
        while words and words[0].lower().strip(".,") in self._leading:
            words = words[1:]
        while words and words[-1].lower().strip(",") in self._suffixes:
            words = words[:-1]
        full = [w.strip(",") for w in words]
        short = list(full)
        while len(short) > 1 and short[-1].lower() in self._generic:
            short = short[:-1]
        names = [" ".join(full), " ".join(short)]
        min_length = self._config.learned_name_min_chars
        return [n for n in dict.fromkeys(names) if len(n) >= min_length]


def _is_acronym(symbol: str, words: list[str]) -> bool:
    """True if the symbol is just the initials of the words before it, e.g. "(NLP)"."""
    parts = [p for w in words for p in re.split(r"[-/]", w) if p]
    if len(parts) < len(symbol):
        return False
    return "".join(p[0] for p in parts[-len(symbol) :]).upper() == symbol


def _as_mentions(pattern: re.Pattern[str], text: str, ticker: str) -> list[EntityMention]:
    """Turn every match of the pattern into a mention of the given ticker."""
    return [
        EntityMention(ticker, m.start(), m.end(), MentionForm.NAME) for m in pattern.finditer(text)
    ]


def _drop_overlaps(mentions: list[EntityMention]) -> list[EntityMention]:
    """Keep one mention where matches overlap, preferring the formal form."""
    kept: list[EntityMention] = []
    for mention in sorted(mentions, key=lambda m: (m.start, -m.end)):
        if kept and mention.start < kept[-1].end:
            if mention.exchange_tagged and not kept[-1].exchange_tagged:
                kept[-1] = mention
            continue
        kept.append(mention)
    return kept
