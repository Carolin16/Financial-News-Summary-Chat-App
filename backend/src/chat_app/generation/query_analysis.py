"""Reads a question to work out what is being asked and which companies it names."""

import re
from enum import StrEnum

from pydantic import BaseModel, Field

from chat_app.core.ticker_registry import TickerRegistry


class Intent(StrEnum):
    """The types of question we recognise, each handled in its own way."""

    ADVICE = "advice"  # "Should I buy Nvidia?"
    PREDICTION = "prediction"  # "Will Apple stock go up?"
    LIVE_DATA = "live_data"  # "What's Apple's current market cap?"
    TIMING = "timing"  # "What happened yesterday?"
    PRICE_TARGET = "price_target"  # "What's Nvidia's price target?"
    ANALYST_VIEW = "analyst_view"  # "What do analysts say about Intel?"
    CAUSAL = "causal"  # "Why did Intel stock jump?"
    NEWS = "news"  # anything else, e.g. "What's the latest on Intel?"


def _rx(pattern: str) -> re.Pattern[str]:
    """Build a pattern that ignores upper and lower case."""
    return re.compile(pattern, re.IGNORECASE)


# Checked top to bottom and the first match wins, so advice and forecasts are caught first.
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
            r"|where will|expected to (rise|fall)"
            # "Which stocks will be the biggest winners?" is also a forecast.
            r"|will .+ (outperform|be the (best|biggest|top)|winners?))"
        ),
    ),
    # Live only with "current" or "right now": a plain "market cap" may be in an article.
    (
        Intent.LIVE_DATA,
        _rx(
            r"\b((current|today'?s|live|latest|real[- ]time) (market cap(italization)?|price"
            r"|stock price|share price|valuation|quote)|trading at (right )?now"
            r"|(price|market cap|valuation) right now)\b"
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


# Which live figure was asked for, so the "no live data" note can name it.
LIVE_METRICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("market cap", _rx(r"market cap|capitali[sz]ation")),
    ("valuation", _rx(r"valuation")),
)
DEFAULT_LIVE_METRIC = "share price"


# Finance words. A question with none of these and no company gets an extra topic check.
FINANCE_VOCABULARY = _rx(
    r"\b(stocks?|shares?|equit(y|ies)|market|nasdaq|dow|s&p|index|earnings|revenue|profit"
    r"|sales|guidance|analysts?|ratings?|upgrade|downgrade|price target|valuation|dividend"
    r"|buyback|ipo|mergers?|acquisitions?|deals?|investors?|investing|invest|funds?|etf|portfolio"
    r"|hedge|economy|economic|inflation|interest rates?|fed|tariffs?|sector|industry"
    r"|compan(y|ies)|ceo|layoffs?|chipmakers?|semiconductors?|trading|traders?|wall street"
    r"|bull(ish)?|bear(ish)?|rally|sell-?off|news)\b"
)


class QueryPlan(BaseModel):
    """What we worked out about one question."""

    intent: Intent
    tickers: list[str]
    live_metric: str | None = None
    clearly_in_scope: bool = Field(
        default=True,
        description="False when neither a company nor finance vocabulary was found.",
    )


class QueryAnalyzer:
    """Sorts a question into a type and finds the companies it mentions."""

    def __init__(
        self,
        registry: TickerRegistry,
        rules: tuple[tuple[Intent, re.Pattern[str]], ...] = INTENT_RULES,
    ) -> None:
        """Take the list of known companies and the question-type rules."""
        self._registry = registry
        self._rules = rules

    def analyze(self, question: str) -> QueryPlan:
        """Return the question type (plain news if no rule matches) and the companies named."""
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
    """Name the live figure asked for, e.g. "market cap", or "share price" if unclear."""
    return next(
        (name for name, pattern in LIVE_METRICS if pattern.search(text)), DEFAULT_LIVE_METRIC
    )
