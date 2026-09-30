"""Fixed user-facing wording added by code, never left to the LLM.

Disclaimers and coverage caveats are compliance statements, so they must appear every time
and read the same way; generating them deterministically guarantees both.
"""

from chat_app.retrieval.coverage import CoverageLevel, CoverageReport

NOT_ADVICE = (
    "This is a summary of news coverage, not investment advice. I can't tell you whether to "
    "buy or sell, or predict where a stock will go."
)

NO_DATES = (
    "The articles in this dataset don't carry reliable publication dates, so I can't tell "
    "you what happened on a specific day such as yesterday."
)


NO_GROUNDED_ANSWER = "I couldn't find information in the news articles that answers this question."

OUT_OF_SCOPE = (
    "I can only answer questions about the financial news in this dataset, such as "
    "companies, stocks, analyst views, and markets. Try asking about a company like Intel, "
    "Apple, or Nvidia."
)

ANSWER_UNAVAILABLE = "Sorry, I couldn't generate an answer right now. Please try again in a moment."

PARTIAL_SOURCES = (
    "Some of the sources are partial articles (paywalled or teasers), so treat those "
    "details as lower-confidence."
)

# Every constant message above; they are app wording, never claims to verify.
FIXED_MESSAGES: tuple[str, ...] = (
    NOT_ADVICE,
    NO_DATES,
    NO_GROUNDED_ANSWER,
    OUT_OF_SCOPE,
    ANSWER_UNAVAILABLE,
    PARTIAL_SOURCES,
)


def live_data_notice(company: str | None, metric: str) -> str:
    """Explains that a live figure (price, market cap) cannot be provided."""
    subject = f"{company}'s current {metric}" if company else f"the current {metric}"
    return (
        f"I don't have live market data, and the articles don't state {subject}, "
        "so I can't give you that figure."
    )


def coverage_notice(report: CoverageReport, company: str) -> str | None:
    """Caveat for companies the dataset covers thinly; None when coverage is full."""
    if report.level is CoverageLevel.LIMITED:
        return (
            f"Coverage of {company} in this news set is limited: "
            f"{_articles(report.primary_articles, 'focuses', 'focus')} on it and "
            f"{_articles(report.mention_articles, 'mentions', 'mention')} it in passing."
        )
    if report.level is CoverageLevel.MENTIONS_ONLY:
        return (
            f"There's no dedicated news about {company} in this dataset; it's only mentioned "
            f"in passing in {_articles(report.mention_articles)}."
        )
    if report.level is CoverageLevel.NONE:
        return f"None of the articles in this dataset mention {company}."
    return None


def coverage_guidance(report: CoverageReport, company: str) -> str:
    """Instructions for the LLM matching the company's coverage level."""
    if report.level is CoverageLevel.FULL:
        return ""
    return (
        f"Coverage of {company} is thin. Summarise only what the sources actually say about "
        f"{company}; ignore their content about other companies. Do not pad the answer with "
        f"loosely related material, and it is fine for the answer to be short."
    )


def _articles(count: int, singular_verb: str = "", plural_verb: str = "") -> str:
    """Count phrase with agreeing verb, e.g. '1 article focuses' / '4 articles focus'."""
    phrase = f"{count} article {singular_verb}" if count == 1 else f"{count} articles {plural_verb}"
    return phrase.strip()
