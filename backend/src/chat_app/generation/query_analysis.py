"""Deterministic query understanding: what is being asked, and about which companies.

Routing decides which answering rules apply (refuse to advise, disclaim timing, ...), so it
is rule-based rather than LLM-based: instant, free, reproducible, and unit-testable. Rules
are data (`INTENT_RULES`), checked in priority order, so adding an intent is a table entry.
"""

import re
from enum import StrEnum

from pydantic import BaseModel, Field

from chat_app.core.ticker_registry import TickerRegistry


class Intent(StrEnum):
    """Question categories, each with its own answering policy."""

    ADVICE = "advice"
    PREDICTION = "prediction"
    LIVE_DATA = "live_data"
    TIMING = "timing"
    PRICE_TARGET = "price_target"
    ANALYST_VIEW = "analyst_view"
    CAUSAL = "causal"
    NEWS = "news"


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Ordered: the first matching rule wins. Refusal-type intents come first so that, e.g.,
# "should I buy Nvidia given its price target?" is treated as advice, not a lookup.
INTENT_RULES: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (
        Intent.ADVICE,
        _rx(
            r"\b(should (i|we)|is it (a )?(good|bad|smart|wise)|worth (buying|investing)"
            r"|(buy|sell|hold|invest in)\b.*\?$|good (buy|investment)|recommend)"
        ),
    ),
    (
        Intent.PREDICTION,
        _rx(
            r"\b(will .+ (go|rise|fall|drop|climb|increase|decrease|crash|rally|recover)"
            r"|going to (go|rise|fall|drop)|forecast|predict|outlook for .+ (stock|shares)"
            r"|where will|expected to (rise|fall))"
        ),
    ),
    (
        Intent.LIVE_DATA,
        _rx(
            r"\b((current|today'?s|live|latest|real[- ]time) (market cap|price|stock price|"
            r"share price|valuation|quote)|market cap(italization)?|trading at (right )?now"
            r"|price right now)\b"
        ),
    ),
    (
        Intent.TIMING,
        _rx(r"\b(yesterday|today|this (morning|afternoon|week)|last (night|week)|tonight)\b"),
    ),
    (Intent.PRICE_TARGET, _rx(r"\b(price target|target price|price objective|pt)\b")),
    (
        Intent.ANALYST_VIEW,
        _rx(r"\b(analysts?|rating|upgrade|downgrade|wall street (says|thinks)|consensus)\b"),
    ),
    (
        Intent.CAUSAL,
        _rx(
            r"\b(why|what caused|reason|behind the)\b.*\b(jump|surge|soar|rise|rose|rall|"
            r"fall|fell|drop|plunge|sink|sank|slide|slid|move|moved|up|down|spike)"
        ),
    ),
)


# Which live figure a LIVE_DATA question asks for, used to word the "unavailable" notice.
LIVE_METRICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("market cap", _rx(r"market cap|capitali[sz]ation")),
    ("valuation", _rx(r"valuation")),
)
DEFAULT_LIVE_METRIC = "share price"


# Vocabulary that marks a question as clearly about financial news. Questions matching
# neither this nor a known company go to the scope guard instead of straight to retrieval.
FINANCE_VOCABULARY = _rx(
    r"\b(stocks?|shares?|equit(y|ies)|market|nasdaq|dow|s&p|index|earnings|revenue|profit"
    r"|sales|guidance|analysts?|ratings?|upgrade|downgrade|price target|valuation|dividend"
    r"|buyback|ipo|mergers?|acquisitions?|deals?|investors?|investing|invest|funds?|etf|portfolio"
    r"|hedge|economy|economic|inflation|interest rates?|fed|tariffs?|sector|industry"
    r"|compan(y|ies)|ceo|layoffs?|chipmakers?|semiconductors?|trading|traders?|wall street"
    r"|bull(ish)?|bear(ish)?|rally|sell-?off|news)\b"
)


class QueryPlan(BaseModel):
    """The analyzer's verdict for one question."""

    intent: Intent
    tickers: list[str]
    live_metric: str | None = None
    clearly_in_scope: bool = Field(
        default=True,
        description="False when neither a company nor finance vocabulary was found.",
    )


class QueryAnalyzer:
    """Classifies intent with ordered rules and resolves companies via the registry."""

    def __init__(
        self,
        registry: TickerRegistry,
        rules: tuple[tuple[Intent, re.Pattern[str]], ...] = INTENT_RULES,
    ) -> None:
        """Rules are injectable so deployments can extend or reorder them."""
        self._registry = registry
        self._rules = rules

    def analyze(self, question: str) -> QueryPlan:
        """Return the question's intent (NEWS if nothing more specific) and tickers."""
        text = question.strip()
        intent = next((i for i, pattern in self._rules if pattern.search(text)), Intent.NEWS)
        tickers = self._registry.find_mentions(text)
        return QueryPlan(
            intent=intent,
            tickers=tickers,
            live_metric=_live_metric(text) if intent is Intent.LIVE_DATA else None,
            clearly_in_scope=bool(tickers) or bool(FINANCE_VOCABULARY.search(text)),
        )


def _live_metric(text: str) -> str:
    return next(
        (name for name, pattern in LIVE_METRICS if pattern.search(text)), DEFAULT_LIVE_METRIC
    )
