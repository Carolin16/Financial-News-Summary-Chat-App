import pytest

from chat_app.config.settings import Settings
from chat_app.core.ticker_registry import TickerRegistry
from chat_app.generation.query_analysis import Intent, QueryAnalyzer


@pytest.fixture(scope="module")
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer(TickerRegistry.from_json(Settings().tickers_path))


@pytest.mark.parametrize(
    ("question", "intent", "tickers"),
    [
        # The 12 reference queries.
        ("What's the latest news on Intel?", Intent.NEWS, ["INTC"]),
        ("What's happening with Apple?", Intent.NEWS, ["AAPL"]),
        ("Any news on IBM?", Intent.NEWS, ["IBM"]),
        ("What do analysts say about Intel?", Intent.ANALYST_VIEW, ["INTC"]),
        ("What's Nvidia's price target?", Intent.PRICE_TARGET, ["NVDA"]),
        ("Why did Intel stock jump?", Intent.CAUSAL, ["INTC"]),
        ("What's the news on Tesla?", Intent.NEWS, ["TSLA"]),
        ("What's the news on Google?", Intent.NEWS, ["GOOGL"]),
        ("Should I buy Nvidia?", Intent.ADVICE, ["NVDA"]),
        ("Will Apple stock go up?", Intent.PREDICTION, ["AAPL"]),
        ("What's Apple's current market cap?", Intent.LIVE_DATA, ["AAPL"]),
        ("What happened in the market yesterday?", Intent.TIMING, []),
    ],
)
def test_reference_queries_are_routed_correctly(analyzer, question, intent, tickers):
    plan = analyzer.analyze(question)
    assert plan.intent is intent
    assert plan.tickers == tickers


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Should I buy Nvidia given its price target?", Intent.ADVICE),
        ("Is Intel a good investment?", Intent.ADVICE),
        ("Will Nvidia stock rise after earnings?", Intent.PREDICTION),
        ("What is the outlook for Intel shares?", Intent.PREDICTION),
        ("What is Microsoft's stock price right now?", Intent.LIVE_DATA),
        ("Did anyone downgrade Intel?", Intent.ANALYST_VIEW),
        ("Why did Netflix shares fall?", Intent.CAUSAL),
        ("Tell me about Amazon's union vote", Intent.NEWS),
    ],
)
def test_paraphrases_and_priority(analyzer, question, intent):
    assert analyzer.analyze(question).intent is intent


def test_multiple_companies_are_all_resolved(analyzer):
    assert set(analyzer.analyze("Compare Intel and AMD news").tickers) == {"INTC", "AMD"}


@pytest.mark.parametrize(
    "question",
    [
        "Who is the president of the USA?",
        "90 + 70",
        "What is the capital of France?",
        "Tell me about quantum computing",
    ],
)
def test_questions_without_company_or_finance_terms_need_scope_check(analyzer, question):
    assert not analyzer.analyze(question).clearly_in_scope


@pytest.mark.parametrize(
    "question",
    [
        "What happened in the market yesterday?",
        "Who is Apple's CEO?",
        "How are tariffs affecting chip stocks?",
        "Any big deals this week?",
    ],
)
def test_company_or_finance_terms_are_clearly_in_scope(analyzer, question):
    assert analyzer.analyze(question).clearly_in_scope
