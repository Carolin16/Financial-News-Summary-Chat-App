import json

from chat_app.core.ids import article_id_for, chunk_point_id
from chat_app.ingestion.loader import JsonArticleRepository


def test_loader_keeps_every_entry_including_duplicates(tmp_path):
    entry = {"title": "T", "link": "https://x/1", "ticker": "AAPL", "full_text": "Body"}
    path = tmp_path / "news.json"
    path.write_text(json.dumps({"AAPL": [entry], "MSFT": [{**entry, "ticker": "MSFT"}]}))

    articles = JsonArticleRepository(path).load()

    assert [a.ticker for a in articles] == ["AAPL", "MSFT"]
    assert {a.link for a in articles} == {"https://x/1"}


def test_loader_falls_back_to_key_when_ticker_missing(tmp_path):
    path = tmp_path / "news.json"
    path.write_text(json.dumps({"IBM": [{"title": "T", "link": "l", "full_text": "b"}]}))
    [article] = JsonArticleRepository(path).load()
    assert article.ticker == "IBM"


def test_ids_are_deterministic_and_distinct():
    article_id = article_id_for("https://x/1")
    assert article_id == article_id_for(" https://x/1 ")
    assert article_id != article_id_for("https://x/2")
    assert chunk_point_id(article_id, 0) == chunk_point_id(article_id, 0)
    assert chunk_point_id(article_id, 0) != chunk_point_id(article_id, 1)
