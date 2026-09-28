import pytest

from chat_app.ingestion.deduplicator import Deduplicator, preferred_version

BODY = " ".join(f"word{i}" for i in range(60))


@pytest.fixture
def dedup() -> Deduplicator:
    return Deduplicator(body_threshold=0.6, title_threshold=0.8, shingle_size=5)


class TestExactDuplicates:
    def test_same_link_under_several_keys_is_merged_with_provenance(self, dedup, make_article):
        articles = [
            make_article(link="https://x/1", source_keys=["AAPL"]),
            make_article(link="https://x/1", source_keys=["MSFT"]),
            make_article(link="https://x/1", source_keys=["AAPL"]),
        ]
        [merged] = dedup.deduplicate(articles)
        assert merged.source_keys == ["AAPL", "MSFT"]
        assert merged.merged_links == []  # same link: nothing absorbed from elsewhere

    def test_most_complete_copy_is_kept(self, dedup, make_article):
        # Copies of one link can differ (e.g. a carousel item the cleaner missed).
        short = make_article(link="https://x/1", text="Stocks fell.", source_keys=["MSFT"])
        full = make_article(
            link="https://x/1", text="Stocks fell as rates rose.", source_keys=["NVDA"]
        )
        [kept] = dedup.deduplicate([short, full])
        assert kept.text == "Stocks fell as rates rose."
        assert kept.source_keys == ["NVDA", "MSFT"]

    def test_equal_copies_keep_the_first_seen(self, dedup, make_article):
        first = make_article(link="https://x/1", text="Quote 706.66.", source_keys=["AAPL"])
        second = make_article(link="https://x/1", text="Quote 706.96.", source_keys=["MSFT"])
        [kept] = dedup.deduplicate([first, second])
        assert kept.text == "Quote 706.66."


class TestNearDuplicates:
    def test_update_with_less_content_loses_to_original(self, dedup, make_article):
        # Real case: the "Update:" teaser is cut off after 5 words of story, the original after 16.
        original = make_article(
            title="Market Chatter: Amazon Shuts Down Inspire Shopping Feed Inside Its App",
            link="https://x/original",
            text="Amazon.com (AMZN) is closing down its shopping feed, called Inspire, inside "
            "its mobile app, The Info",
            is_stub=True,
            source_keys=["AMZN"],
        )
        update = make_article(
            title="Update: Market Chatter: Amazon Shuts Down Inspire Shopping Feed Inside Its App",
            link="https://x/update",
            text="Amazon.com (AMZN) is closing down i",
            is_stub=True,
            source_keys=["AMZN"],
        )
        [kept] = dedup.deduplicate([original, update])
        assert kept.link == "https://x/original"
        assert kept.merged_links == ["https://x/update"]

    def test_update_wins_a_tie(self, make_article):
        original = make_article(title="Intel news", link="https://x/1", text="a b c")
        update = make_article(title="Update: Intel news", link="https://x/2", text="d e f")
        assert preferred_version(original, update) == (update, original)

    def test_templated_bodies_with_different_headlines_are_kept(self, dedup, make_article):
        # Analyst-note teasers share boilerplate bodies but report different firms' targets.
        cantor = make_article(
            title="Cantor Adjusts Intel Target to $29", link="https://x/1", text=BODY
        )
        citic = make_article(
            title="Citic Downgrades Intel, Target $24", link="https://x/2", text=BODY
        )
        assert len(dedup.deduplicate([cantor, citic])) == 2

    def test_same_headline_with_different_bodies_is_kept(self, dedup, make_article):
        first = make_article(title="Sector Update: Tech", link="https://x/1", text=BODY)
        other_body = " ".join(f"other{i}" for i in range(60))
        second = make_article(title="Sector Update: Tech", link="https://x/2", text=other_body)
        assert len(dedup.deduplicate([first, second])) == 2

    def test_same_headline_and_body_is_merged(self, dedup, make_article):
        first = make_article(title="Apple news", link="https://x/1", text=BODY)
        second = make_article(title="Apple news", link="https://x/2", text=BODY + " extra words")
        [kept] = dedup.deduplicate([first, second])
        assert kept.link == "https://x/2"
        assert kept.merged_links == ["https://x/1"]


def test_preserves_first_seen_order(dedup, make_article):
    articles = [make_article(title=f"Story {n}", link=f"https://x/{n}") for n in "abc"]
    assert [a.link for a in dedup.deduplicate(articles)] == [
        "https://x/a",
        "https://x/b",
        "https://x/c",
    ]
