# Financial News Chat

Ask about recent financial news from `data/stock_news.json` and get short, **grounded**
summaries: every point links to its source article, every number is checked against the
cited source, and questions the data can't answer (advice, forecasts, live prices, "what
happened yesterday") get honest, grounded responses instead of guesses.

## Quick start

### Docker (Qdrant + API + UI)

```bash
cp .env.example .env                                   # add OPENAI_API_KEY
docker compose up --build -d                           # qdrant -> backend -> ui
docker compose run --rm backend chat-index             # offline indexing (idempotent)
open http://localhost:3000
```

### Local, without Docker (embedded Qdrant)

```bash
cd backend
python -m venv .venv && .venv/Scripts/activate        # or source .venv/bin/activate
pip install -e ".[dev]"
export QDRANT_LOCAL_PATH=../.qdrant                    # embedded Qdrant; no server needed
chat-index                                             # build the index
uvicorn chat_app.api.main:app --port 8000

cd ../ui && npm install && npm run dev                 # http://localhost:5173
```

### Tests and checks

```bash
cd backend
pytest                          # unit + integration (no network, no API key)
pytest -m eval                  # the 12 reference queries against the live LLM
ruff check . && black --check . && mypy src
cd ../ui && npm test && npm run lint
```

## Architecture

```
            OFFLINE (chat-index)                                 ONLINE (FastAPI, SSE)
 stock_news.json                                      question
   -> load -> clean -> flag stubs -> dedupe              -> QueryAnalyzer (rules): intent + tickers
                                                         -> scope gate (rules; LLM only if unclear)
   -> enrich (LLM, cached) -> chunk + header             -> CoverageIndex: full / limited / none
   -> embed (dense) + BM25 (sparse) -> Qdrant ------->   -> CompanyFirstRetrieval -> Qdrant hybrid (RRF)
                                                         -> LlmSummarizer (streamed draft)
                                                         -> verify_answer (citations + numbers)
                                                         -> fixed notices -> final event -> React UI
```

Layers map to packages under `backend/src/chat_app/`: `ingestion` → `retrieval` →
`generation` → `api`; `core` holds domain models and interfaces (`ArticleRepository`,
`MetadataExtractor`, `EmbeddingProvider`, `Retriever`, `Summarizer`, `StructuredLlm`,
`StreamingLlm`), and composition roots (`ingestion/factory.py`, `api/container.py`) are
the only places that name concrete classes. The UI (`ui/`) has no business logic: it
renders the server's events.

## Key design decisions

**Relevance is judged from content.** An LLM labels each article's `primary_tickers`
(what it's about) vs `mentioned_tickers` (named in passing), plus event types, sentiment,
and article type. Results are cached in `data/enrichment_cache.json` (keyed by content
hash), so re-indexing and Docker builds make no LLM calls; a mention-counting heuristic is
the fallback if the LLM fails. On the dataset this correctly handles every trap in the
brief: the Occidental/Buffett, Suze Orman, and Demi Moore entries have no Apple/Netflix
primary; "Intel Breakup…" filed under AAPL is primary INTC.

**Cleaning was driven by profiling the data.** Findings that shaped the code:
- Paywalled MT Newswires teasers share boilerplate, so body similarity alone falsely
  "deduplicates" different analyst notes (Cantor vs Citic score 1.0). Near-duplicates
  therefore require a matching headline too (ignoring an `Update:` prefix): 138 entries →
  118 unique links → **117 articles**, with the Amazon "Update:" re-publication merged.
- Stubs keep their key fact only in the title ("DBS … Price Target to $160 From $175"), so
  titles go into every chunk's contextual header and count as citable source text.
- **No mojibake exists in this file** (0 occurrences of `â€`, `Â`, `Ã`), although the brief
  expects it; what looks garbled in a non-UTF-8 console is curly quotes and non-breaking
  spaces. The encoding-repair step (ftfy) stays in the pipeline defensively for other feeds.
- Cleaning is a pipeline of small steps behind a `CleaningStep` interface (encoding repair →
  punctuation → sentence spacing → symbols → noise blocks → URLs → repeated title →
  whitespace), with every pattern in `ingestion/cleaning/cleaning_rules.toml`. It removes
  13.8% of the text: UI chrome, cross-promotion, sponsored lines, publisher footers,
  disclaimers, image credits, and contact details.
- Promo blocks are removed whole, not just their marker. Headline lists often run into real
  text with no punctuation ("…Analyst Ratings The Chinese startup may be…"), so the end is
  detected from the switch from Title-Case headlines to sentence-case prose.
- Footers are stripped to the end only after a guard confirms the tail holds no article
  content (company names, tickers, or numbers in full sentences outside known boilerplate).
- Promotional sentences ("our newsletter … has returned 275%") are removed, keyed on the
  promo wording, never on "returned N%", which appears in four genuine figures.
- Invariants are tested over the whole dataset: cleaning is idempotent, never creates a
  number, and keeps every exchange ticker and € / ¥ figure in article content.
- The same punctuation normaliser (`core/text_normalization.py`) is applied to LLM output
  before numbers are verified, so both sides compare identical characters.
- **24 of 118 source entries** carry a truncation signal (paywall or "Continue Reading");
  after deduplication 24 articles are stubs, including one video blurb that is only 77 words
  once its "Related Videos" carousel is removed. Stubs are labelled `PARTIAL` in prompts.

**Deterministic routing, LLM only for writing.** Intent classification is ordered regex
rules (`generation/query_analysis.py`): instant, free, reproducible, and unit-tested
against all 12 reference queries. Compliance wording (not-advice statement, no-dates
notice, "no live data", coverage caveats) is added by code, so it can't be omitted.

**Out-of-scope questions are declined before retrieval.** Grounding proves a claim is
supported, not that the question belongs here: "Who is the president?" is answerable
from articles that mention him. A question naming a company or using finance vocabulary
is in scope immediately; anything else gets one small structured LLM check
(`scope_guard.py`). Out-of-scope questions receive a fixed message with no retrieval and
no sources. The check fails open on LLM errors, because answers are still verified.

**Coverage is computed, not guessed.** `CoverageIndex` counts non-stub primary articles per
company. With a threshold of 5, the six well-covered companies are FULL while IBM (4),
Google (3), and Tesla (1) are LIMITED — exactly the brief's thin/partial cases. Limited
coverage gets a stated caveat and a prompt instruction not to pad; "no articles" is only
ever said when there are zero mentions.

**Hybrid search.** Dense (`text-embedding-3-small`, cosine) and BM25 sparse vectors live
side by side in Qdrant and are fused server-side with RRF under the same metadata filter.
Retrieval is company-first: primary-subject chunks, topped up with passing mentions only
if slots remain. BM25 is what ranks the "DBS … $160" stub first for a query like
"DBS $160 target" (integration-tested).

**Nothing unverified reaches the user.** The LLM's output is buffered and each line is
checked by `verify_answer` as soon as it is complete; only lines that pass are streamed.
Any sentence that cites nothing, or quotes a number absent from the sources it cites, is
withheld, so an off-topic answer (e.g. "90 + 70 = 160") never appears even briefly. The
`final` event carries the complete answer with the fixed notices.

## Answering rules: where each is enforced

| Rule | Enforcement |
|---|---|
| Grounded, cited | Prompt rule + verifier drops uncited sentences + unknown `[n]` stripped |
| Numbers exact | Verifier: every number must appear in a *cited* source (title included) |
| Firm vs consensus target | `intent_price_target.md`; eval asserts `$160` ↔ DBS and `174.93` ↔ mean |
| "Why did it move?" | `intent_causal.md`: only causes the sources state, attributed |
| Limited coverage (Tesla, Google, IBM) | `CoverageIndex` → fixed caveat + no-padding guidance |
| Advice / prediction | Summary of attributed views + fixed not-advice statement |
| Timing ("yesterday") | No LLM: fixed no-dates notice + coverage summary |
| Not in data (market cap) | Fixed "no live data" notice; only as-of-article facts follow |
| Out of scope (politics, maths, trivia) | Scope gate → fixed message, no retrieval, no sources |

## Testing

- **Unit** (`tests/unit`): every ingestion step (cleaner, stub detector, deduplicator,
  loader/IDs, enrichment + cache + fallback, chunker), ticker registry, query routing for
  all 12 reference queries plus paraphrases, filters, coverage, grounding verifier,
  answer-service policies (with fake retriever/summarizer), API contract (SSE framing,
  validation, health). Includes a regression test on the real dataset (117 articles,
  23 stubs, no promo text).
- **Integration** (`tests/integration`): indexer + hybrid retriever on real Qdrant code
  paths (embedded mode): idempotent/incremental re-indexing, stale-point removal, filters,
  exact-token ranking.
- **Eval** (`tests/eval`, `-m eval`): the 12 reference queries end to end with the live
  LLM. Numbers are re-checked against the *original dataset* text of cited articles,
  independently of the app's verifier, plus a rule assertion per category.

Latest results: **304 passed** (unit + integration), **27/27 eval checks passed**
(12 reference queries + out-of-scope and borderline questions), re-run on the index rebuilt
after the cleaning rewrite (2026-09-28: 117 articles, 272 chunks, 21 stale points removed);
`ruff`, `mypy --strict`, ESLint, and 9 UI tests clean.

### Known issues

- **Relevance labels are not stable across LLM runs for multi-company articles.** Running the
  extractor three times on identical text gave different primary tickers for e.g. "The race
  for AI will only get 'faster and cheaper'" (MSFT/AMZN/GOOGL/IBM/ORCL primary in 1 of 3 runs).
  Re-enriching after the cleaning change flipped 9 articles' primary/stub labels this way; the
  cleaning itself was ruled out (same label distribution on old and new text). Consequence:
  Microsoft sits exactly on the FULL-coverage threshold and dropped from FULL to LIMITED.
  Fix under consideration: majority vote over several extraction runs, cached.

## Production readiness

- Offline indexing is separate from serving; deterministic point IDs (UUID5 of article +
  chunk index) and content hashes make it idempotent and incremental (unchanged chunks
  aren't re-embedded; removed ones are deleted).
- OpenAI calls have timeouts and exponential-backoff retries (SDK); failures degrade to a
  fallback message (answers) or the heuristic extractor (enrichment).
- One JSON log line per query: query, intent, tickers, retrieved chunk IDs, latency,
  token usage, sentences removed by the verifier.
- `/health` reports Qdrant reachability and index readiness; inputs validated by Pydantic.
- All tunables come from environment variables (`.env.example`).

## Trade-offs and future work

- Rule-based intent routing is precise for known phrasings but can miss unusual ones;
  an LLM classifier with the rules as a guard-rail would widen coverage.
- The number verifier checks presence, not meaning (a real figure attached to the wrong
  claim would pass); an entailment check per sentence would close that gap.
- Near-duplicate detection is pairwise (O(n²)); MinHash/LSH would be needed at scale.
- The coverage index is built at API startup; with continuous ingestion it should be
  maintained incrementally or stored alongside the index.
- Black is configured, but local formatting used `ruff format` (Black-compatible) because
  Black refuses to run on Python 3.12.5; CI on a current 3.12.x runs Black itself.
