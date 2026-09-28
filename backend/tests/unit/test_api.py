"""HTTP contract: input validation, SSE framing, and health, with a fake container."""

import json

import pytest
from fastapi.testclient import TestClient

from chat_app.api.container import VectorStoreUnavailableError
from chat_app.api.main import create_app
from chat_app.api.routes.chat import SEARCH_UNAVAILABLE
from chat_app.config.settings import Settings
from chat_app.generation.events import FinalEvent, MetaEvent


class FakeService:
    def __init__(self):
        self.questions: list[str] = []

    async def answer(self, question):
        self.questions.append(question)
        yield MetaEvent(intent="news", tickers=["INTC"], coverage=[])
        yield FinalEvent(answer="Intel rose [1].", citations=[], notices=[])


class FakeQdrant:
    def __init__(self, up=True, ready=True):
        self.up, self.ready = up, ready

    async def collection_exists(self, name):
        if not self.up:
            raise ConnectionError("qdrant down")
        return self.ready

    async def count(self, name, exact=False):
        return type("Count", (), {"count": 5})()


class FakeContainer:
    def __init__(self, qdrant=None):
        self.settings = Settings(max_query_chars=50)
        self.qdrant = qdrant or FakeQdrant()
        self.service = FakeService()

    async def answer_service(self):
        return self.service

    async def close(self):
        pass


@pytest.fixture
def container() -> FakeContainer:
    return FakeContainer()


@pytest.fixture
def client(container, monkeypatch):
    monkeypatch.setenv("MAX_QUERY_CHARS", "50")
    from chat_app.config.settings import get_settings

    get_settings.cache_clear()
    with TestClient(create_app(container=container)) as test_client:
        yield test_client
    get_settings.cache_clear()


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\r\n\r\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_chat_streams_typed_sse_events(client, container):
    response = client.post("/chat", json={"question": "  Latest   on Intel? "})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["meta", "final"]
    assert events[1][1]["answer"] == "Intel rose [1]."
    assert container.service.questions == ["Latest on Intel?"]


@pytest.mark.parametrize(
    "payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 51}]
)
def test_chat_rejects_invalid_questions(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_health_ok_when_index_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qdrant": "up", "index_ready": True}


def test_health_reports_not_ready_before_indexing(monkeypatch):
    container = FakeContainer(qdrant=FakeQdrant(ready=False))
    with TestClient(create_app(container=container)) as client:
        assert client.get("/health").json()["index_ready"] is False


def test_health_503_when_qdrant_down():
    container = FakeContainer(qdrant=FakeQdrant(up=False))
    with TestClient(create_app(container=container)) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["qdrant"] == "down"


def test_chat_returns_503_with_readable_message_when_store_unreachable():
    container = FakeContainer()

    async def unavailable():
        raise VectorStoreUnavailableError("All connection attempts failed")

    container.answer_service = unavailable
    with TestClient(create_app(container=container)) as client:
        response = client.post("/chat", json={"question": "Intel news?"})
    assert response.status_code == 503
    assert response.json() == {"detail": SEARCH_UNAVAILABLE}
