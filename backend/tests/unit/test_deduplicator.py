import pytest

from chat_app.ingestion.deduplicator import Deduplicator

BODY = " ".join(f"word{i}" for i in range(60))


@pytest.fixture
def dedup() -> Deduplicator:
    return Deduplicator(body_threshold=0.6, title_threshold=0.8, shingle_size=5)


def test_same_link_under_several_tickers_is_merged(dedup, make_article):
    articles = [
        make_article(link="https://x/1", source_tickers=["AAPL"]),
        make_article(link="https://x/1", source_tickers=["MSFT"]),
        make_article(link="https://x/1", source_tickers=["AAPL"]),
    ]
    [merged] = dedup.deduplicate(articles)
    assert merged.source_tickers == ["AAPL", "MSFT"]


def test_updated_version_replaces_original(dedup, make_article):
    original = make_article(
        title="Market Chatter: Amazon Shuts Down Inspire", link="https://x/1", is_stub=True
    )
    updated = make_article(
        title="Update: Market Chatter: Amazon Shuts Down Inspire",
        link="https://x/2",
        is_stub=True,
        source_tickers=["AMZN"],
    )
    [kept] = dedup.deduplicate([original, updated])
    assert kept.link == "https://x/2"
    assert set(kept.source_tickers) == {"INTC", "AMZN"}


def test_templated_bodies_with_different_headlines_are_kept(dedup, make_article):
    # Analyst-note teasers share boilerplate bodies but report different firms' targets.
    cantor = make_article(title="Cantor Adjusts Intel Target to $29", link="https://x/1", text=BODY)
    citic = make_article(title="Citic Downgrades Intel, Target $24", link="https://x/2", text=BODY)
    assert len(dedup.deduplicate([cantor, citic])) == 2


def test_same_headline_with_different_bodies_is_kept(dedup, make_article):
    first = make_article(title="Sector Update: Tech", link="https://x/1", text=BODY)
    other_body = " ".join(f"other{i}" for i in range(60))
    second = make_article(title="Sector Update: Tech", link="https://x/2", text=other_body)
    assert len(dedup.deduplicate([first, second])) == 2


def test_longer_text_wins_when_neither_is_marked_updated(dedup, make_article):
    short = make_article(title="Apple news", link="https://x/1", text=BODY)
    longer = make_article(title="Apple news", link="https://x/2", text=BODY + " extra words here")
    [kept] = dedup.deduplicate([short, longer])
    assert kept.link == "https://x/2"


def test_preserves_first_seen_order(dedup, make_article):
    articles = [make_article(title=f"Story {n}", link=f"https://x/{n}") for n in "abc"]
    assert [a.link for a in dedup.deduplicate(articles)] == [
        "https://x/a",
        "https://x/b",
        "https://x/c",
    ]
