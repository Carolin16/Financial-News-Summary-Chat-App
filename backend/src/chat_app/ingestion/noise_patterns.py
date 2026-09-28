"""Publisher boilerplate and promotional patterns found in the news feed.

Kept as data, separate from the cleaning logic, so supporting a new publisher means adding
a pattern here rather than editing the cleaner (OCP). Patterns were derived by profiling
`stock_news.json`; each group notes why it is removed.
"""

import re

# Truncation/teaser markers: signal that the full article is behind a link or paywall.
# The stub detector looks for these in the *raw* text, so they are exported separately.
TEASER_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bContinue Reading\b"),
    re.compile(r"PREMIUM\s+Upgrade to read this .*?article", re.IGNORECASE),
)

# Inline UI chrome and paywall text removed wherever it appears.
INLINE_NOISE: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"PREMIUM\s+Upgrade to read this .*?Already have a subscription\?\s*Sign in",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\bContinue Reading\b"),
    re.compile(r"\bView Comments\s*\|?"),
    re.compile(r"\bStory Continues\b"),
    re.compile(r"Try Now\s*>>"),
    re.compile(r"See today's best-performing stocks on TipRanks\s*>>"),
    re.compile(r"[★☆]+"),  # rating glyphs from screener tables carry no meaning as text
)

# Sentences that open with a cross-promotion label. The whole sentence is dropped because
# the label is followed by other articles' headlines, which would otherwise be retrieved
# as if they were facts in this article.
PROMO_SENTENCE_PREFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(READ NEXT|READ ALSO|Read More(?: on [A-Z.]+)?|Up Next)\s*:", re.IGNORECASE),
    re.compile(r"^\s*(Trending|Don't Miss|Recommended|Related)\s*:", re.IGNORECASE),
)

# Sentences that are self-promotion or disclaimers. Dropped so their figures (e.g. "our
# newsletter has returned 275%") never leak into answers as if they were news.
PROMO_SENTENCE_CONTENT: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bnewsletter", re.IGNORECASE),
    re.compile(r"\bclick (here|now)\b", re.IGNORECASE),
    re.compile(r"\bsign up\b.*\b(for|to)\b", re.IGNORECASE),
    re.compile(r"\bfree (daily )?(email|subscription|trial)\b", re.IGNORECASE),
    re.compile(r"\btarget yield with monthly dividends\b", re.IGNORECASE),
    re.compile(r"\bpre-IPO shares\b", re.IGNORECASE),
    re.compile(r"\bsee more details here\b", re.IGNORECASE),
    re.compile(r"^\s*Disclosure\s*:", re.IGNORECASE),
    re.compile(r"\boriginally (published|appeared) (at|on|in)\b", re.IGNORECASE),
    re.compile(r"\bfollow (him|her|them|us) on\b", re.IGNORECASE),
    re.compile(r"\bEmail [A-Z][a-z]+ [A-Z][a-z]+ at\b"),
    re.compile(r"\bfirst on TheFly\b", re.IGNORECASE),
    re.compile(r"\bunlock the (profitable )?stock recommendations\b", re.IGNORECASE),
    re.compile(r"\bGUARANTEED\b"),
)

# Common UTF-8-read-as-cp1252 sequences, mapped back to the character they encoded; used when
# a whole-text round trip is not possible. ASCII folding happens later, in one place.
MOJIBAKE_REPLACEMENTS: dict[str, str] = {
    "â€™": "’",  # right single quote
    "â€˜": "‘",  # left single quote
    "â€œ": "“",  # left double quote
    "â€": "”",  # right double quote
    "â€“": "–",  # en dash
    "â€”": "—",  # em dash
    "â€¦": "…",  # ellipsis
    "â€¢": "•",  # bullet
    "Â": "",  # stray prefix before non-breaking spaces
}
MOJIBAKE_MARKERS: tuple[str, ...] = ("â€", "Â", "Ã")

# Typographic characters folded to ASCII so exact-token (BM25) matching and the patterns
# above behave the same whichever quote style a publisher used.
TYPOGRAPHIC_REPLACEMENTS: dict[str, str] = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "•": "-",
    "·": "-",
}
