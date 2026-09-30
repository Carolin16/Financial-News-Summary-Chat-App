"""Shared eval fixtures: the cases file, live answers, stored chunks, cleaned dataset.

Session-scoped so the live queries run once for every eval module. Tests skip, rather
than error, when the API key or the vector store is missing. Every case's answer and the
outcome of each test that checked it are written to `runs/<timestamp>.json`.
"""

import asyncio
import json
import socket
from collections.abc import Callable, Generator
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from cases import EvalCase, EvalSuite, default_suite

from chat_app.api.container import Container
from chat_app.config.settings import Settings, get_settings
from chat_app.core.models import Chunk
from chat_app.core.ticker_registry import TickerRegistry
from chat_app.generation.claim_judge import ClaimJudge
from chat_app.generation.events import FinalEvent, MetaEvent
from chat_app.generation.llm_client import OpenAILlmClient
from chat_app.ingestion.cleaning.config import CleaningConfig
from chat_app.ingestion.cleaning.factory import build_text_cleaner
from chat_app.ingestion.loader import JsonArticleRepository
from chat_app.retrieval.chunk_catalog import load_chunks

# Only decides skip-vs-run; a reachable store answers a TCP connect well within this.
_REACHABILITY_TIMEOUT_SECONDS = 2.0

# Each eval session writes one JSON file here (gitignored) so failures can be inspected later.
_RUNS_DIR = Path(__file__).parent / "runs"
_RUN_FILE_TIMESTAMP = "%Y%m%d-%H%M%S"
_LOGGED_CITATION_FIELDS = {"number", "title", "link", "is_partial"}
_RUN_FILE_KEY = pytest.StashKey[Path]()

# Per case id: query, answer, citations and the outcome of every test that checked it.
_run_log: dict[str, dict[str, Any]] = {}


def _run_entry(case_id: str) -> dict[str, Any]:
    """The log entry for a case, created on first use."""
    if case_id not in _run_log:
        suite = default_suite()
        _run_log[case_id] = {
            "id": case_id,
            "group": suite.group_of(case_id),
            "query": suite.case(case_id).query,
            "answer": None,
            "citations": [],
            "checks": [],
        }
    return _run_log[case_id]


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Record each case-level test's outcome and, on failure, why."""
    report = yield
    if not isinstance(item, pytest.Function) or not hasattr(item, "callspec"):
        return report
    case_id = item.callspec.params.get("case_id")
    # Setup only matters when it stops the test (a skip or a fixture error).
    relevant = report.when == "call" or (report.when == "setup" and not report.passed)
    if isinstance(case_id, str) and relevant:
        reason = ""
        if report.skipped and isinstance(report.longrepr, tuple):
            reason = report.longrepr[2]
        elif call.excinfo is not None:
            reason = call.excinfo.exconly()
        _run_entry(case_id)["checks"].append(
            {"test": item.originalname, "outcome": report.outcome, "reason": reason}
        )
    return report


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Write this session's case results to a timestamped JSON file (eval runs only)."""
    if not _run_log:
        return
    for entry in _run_log.values():
        entry["passed"] = not any(c["outcome"] == "failed" for c in entry["checks"])
    _RUNS_DIR.mkdir(exist_ok=True)
    path = _RUNS_DIR / f"{datetime.now().strftime(_RUN_FILE_TIMESTAMP)}.json"
    path.write_text(json.dumps(list(_run_log.values()), indent=2), encoding="utf-8")
    session.config.stash[_RUN_FILE_KEY] = path


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Point at the results file so it's easy to find after a run."""
    if path := terminalreporter.config.stash.get(_RUN_FILE_KEY, None):
        terminalreporter.write_line(f"eval results written to {path}")


def _api_skip_reason() -> str | None:
    """Why tests that call the LLM can't run, or None."""
    if not get_settings().openai_api_key.get_secret_value():
        return "OPENAI_API_KEY not set"
    return None


def _pipeline_skip_reason() -> str | None:
    """Why end-to-end tests (LLM + built index) can't run, or None."""
    if reason := _api_skip_reason():
        return reason
    settings = get_settings()
    if settings.qdrant_local_path is not None:
        if not settings.qdrant_local_path.exists():
            return "embedded index not built; run chat-index first"
        return None
    url = urlparse(settings.qdrant_url)
    try:
        with socket.create_connection(
            (url.hostname or "localhost", url.port or 80), timeout=_REACHABILITY_TIMEOUT_SECONDS
        ):
            return None
    except OSError:
        return f"Qdrant not reachable at {settings.qdrant_url}"


def _make_claim_judge(settings: Settings) -> ClaimJudge:
    """A deterministic judge built from the evaluation settings."""
    llm = OpenAILlmClient(
        settings, model=settings.claim_judge_model, temperature=settings.claim_judge_temperature
    )
    return ClaimJudge(
        llm,
        min_quote_chars=settings.claim_judge_min_quote_chars,
        max_concurrency=settings.claim_judge_max_concurrency,
        votes=settings.claim_judge_votes,
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Run any test that takes `case_id` once per reference case in the cases file."""
    # Tests that choose their own subset of cases (e.g. only q7/q8) keep that choice.
    explicit = any(
        "case_id" in marker.args[0] for marker in metafunc.definition.iter_markers("parametrize")
    )
    if "case_id" in metafunc.fixturenames and not explicit:
        ids = [c.id for c in default_suite().reference]
        metafunc.parametrize("case_id", ids, ids=ids)


@pytest.fixture(scope="session")
def settings() -> Settings:
    """App settings, including the evaluation thresholds."""
    return get_settings()


@pytest.fixture(scope="session")
def suite() -> EvalSuite:
    """Every eval case and its expectations, from the cases file."""
    return default_suite()


@pytest.fixture(scope="session")
def require_api() -> None:
    """Skip unless the LLM can be called."""
    if reason := _api_skip_reason():
        pytest.skip(reason)


@pytest.fixture(scope="session")
def require_pipeline() -> None:
    """Skip unless the LLM and the built index are both available."""
    if reason := _pipeline_skip_reason():
        pytest.skip(reason)


@pytest.fixture(scope="session")
def claim_judge_factory(settings: Settings) -> Callable[[], ClaimJudge]:
    """Builds a fresh judge; call it inside the event loop that will use it."""
    return lambda: _make_claim_judge(settings)


def _answer_cases(cases: list[EvalCase]) -> dict[str, tuple[MetaEvent, FinalEvent]]:
    """Meta and final events for each case, keyed by case id, run concurrently; logs each."""

    async def collect() -> dict[str, tuple[MetaEvent, FinalEvent]]:
        container = Container(get_settings())
        service = await container.answer_service()

        async def one(case: EvalCase) -> tuple[MetaEvent, FinalEvent]:
            events = [e async for e in service.answer(case.query)]
            meta = next(e for e in events if isinstance(e, MetaEvent))
            return meta, next(e for e in reversed(events) if isinstance(e, FinalEvent))

        try:
            answers = await asyncio.gather(*(one(c) for c in cases))
        finally:
            await container.close()
        return {c.id: answer for c, answer in zip(cases, answers, strict=True)}

    answered = asyncio.run(collect())
    for case_id, (_, final) in answered.items():
        entry = _run_entry(case_id)
        entry["answer"] = final.answer
        entry["citations"] = [
            c.model_dump(include=_LOGGED_CITATION_FIELDS) for c in final.citations
        ]
    return answered


@pytest.fixture(scope="session")
def results(require_pipeline: None, suite: EvalSuite) -> dict[str, tuple[MetaEvent, FinalEvent]]:
    """Meta and final events for each reference case, keyed by case id, run concurrently once."""
    return _answer_cases(suite.reference)


@pytest.fixture(scope="session")
def scope_results(require_pipeline: None, suite: EvalSuite) -> dict[str, FinalEvent]:
    """Final answers for the out-of-scope and borderline cases, keyed by case id."""
    answered = _answer_cases([*suite.out_of_scope, *suite.borderline])
    return {case_id: final for case_id, (_, final) in answered.items()}


@pytest.fixture(scope="session")
def adversarial_results(
    require_pipeline: None, suite: EvalSuite
) -> dict[str, tuple[MetaEvent, FinalEvent]]:
    """Meta and final events for each adversarial case, keyed by case id."""
    return _answer_cases(suite.adversarial)


@pytest.fixture(scope="session")
def chunks_by_id(require_pipeline: None) -> dict[str, Chunk]:
    """Every indexed chunk, so citations can be checked against the exact passage cited."""

    async def collect() -> list[Chunk]:
        container = Container(get_settings())
        try:
            return await load_chunks(container.qdrant, container.settings.qdrant_collection)
        finally:
            await container.close()

    return {c.chunk_id: c for c in asyncio.run(collect())}


@pytest.fixture(scope="session")
def article_text_by_link(settings: Settings) -> dict[str, str]:
    """Cleaned dataset text (title + body) keyed by link."""
    cleaner = build_text_cleaner(
        CleaningConfig.from_toml(settings.cleaning_rules_path),
        TickerRegistry.from_json(settings.tickers_path).company_terms(),
    )
    cleaned = (cleaner.clean(raw) for raw in JsonArticleRepository(settings.data_path).load())
    return {c.link: f"{c.title}\n{c.text}" for c in cleaned}
