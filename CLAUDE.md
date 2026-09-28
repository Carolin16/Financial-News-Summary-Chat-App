# CLAUDE.md

Guidance for Claude Code when working in this repository. Source of truth: `requirments.docx` (plain text despite the extension).

## Project overview

A simple chat app where a user asks about recent financial news from `data/stock_news.json` and gets grounded summary answers.
The UI should be simple and functional; styling is **not** evaluated.
Evaluated on: **code quality**, **creativity of approach**, and **completeness of testing**.

## Tech stack (fixed decisions — don't substitute)

| Concern | Choice |
|---|---|
| LLM | GPT 5.6 Terra |
| Embeddings | `text-embedding-3-small` |
| Vector DB | Qdrant |
| Search | Hybrid: dense vectors (cosine) + sparse BM25 vectors stored side by side, fused server-side with Reciprocal Rank Fusion (RRF). Gives exact-token matching ("DBS", "$160") without a separate search engine. |
| Metadata filters | `primary_tickers`, `mentioned_tickers`, `is_stub`, `event_types`, `sentiment`, `article_type` |
| Chunking | Sentence-aware recursive chunking with contextual headers; articles ≤ 500 tokens stay a single chunk |
| Backend | FastAPI (async, SSE streaming); Pydantic schemas shared with metadata models |
| Frontend | React (JavaScript) |

## Dataset: `data/stock_news.json`

- Shape: `{ "<TICKER>": [ { title, link, ticker, full_text }, ... ] }`
- 138 entries: AAPL, MSFT, AMZN, NFLX, NVDA, INTC (20 each), IBM (18).
- **118 unique articles**; 20 links repeat across tickers, and some articles appear twice in updated form.
- Text contains mojibake (`â€™`, `Â`, etc.) — normalise during cleaning.

### Ingestion must

1. **Deduplicate** by link (and near-duplicate content for updated versions). Counting duplicates would overstate coverage.
2. **Judge relevance from content, never from the ticker key.** Off-topic entries are excluded or flagged, e.g. under AAPL: Occidental/Buffett, Suze Orman, the retail boycott; some "Apple" entries are really about Intel; a Netflix entry about a celebrity's outfit. Derive `primary_tickers` vs `mentioned_tickers` from the text.
3. **Flag thin content** (~23 entries: paywalls, teasers, "Continue Reading") as `is_stub`; answers based on them are low-confidence.
4. **Strip promotional noise** ("our newsletter has returned 275%…", "READ NEXT", "Don't Miss", "Trending", "View Comments") so it doesn't distort tone or leak numbers.

## Answering rules (non-negotiable)

- **Grounded only:** every summary point is traceable to (and links to) a source article; nothing comes from outside the file or from model memory.
- **Numbers:** no number appears unless quoted exactly from a source article. Guard against hallucinated figures.
- **Price targets:** distinguish a single firm's target from a consensus figure.
- **"Why did it move?":** state the cause the articles give; do not invent other reasons.
- **Limited coverage (Tesla, Google):** say coverage is limited and summarise only the actual mentions (or say there is no dedicated news). Never claim "no data" when mentions exist, and never give a full summary from own knowledge.
- **Thin coverage (IBM):** don't pad with unrelated press releases.
- **Advice / prediction:** respond with a summary of what the news and analysts say plus a clear statement that the bot doesn't give investment advice. Never forecast in the bot's own voice. (Brokerage context — informational, not advice.)
- **Timing:** don't claim timing that can't be verified. For "What happened yesterday?", explain the articles lack reliable dates and offer a coverage summary instead.
- **No answer in data** (e.g. current market cap): say so; don't answer from memory.

## Reference queries (acceptance / eval suite)

| # | Category | Query |
|---|---|---|
| 1 | Basic | What's the latest news on Intel? |
| 2 | Basic | What's happening with Apple? |
| 3 | Basic (thin) | Any news on IBM? |
| 4 | Analyst ratings | What do analysts say about Intel? |
| 5 | Price target | What's Nvidia's price target? |
| 6 | Causal | Why did Intel stock jump? |
| 7 | Partial coverage | What's the news on Tesla? |
| 8 | Partial coverage | What's the news on Google? |
| 9 | Advice (grounded refusal) | Should I buy Nvidia? |
| 10 | Prediction (grounded refusal) | Will Apple stock go up? |
| 11 | Not in data | What's Apple's current market cap? |
| 12 | Not in data / timing | What happened in the market yesterday? |

Tests should cover each of these against the answering rules above (citations present, numbers traceable, refusals correct), plus unit tests for each ingestion step.

## Architecture & code quality

- **Layers:** ingestion → retrieval/processing → generation → API → UI. The UI contains **no business logic**.
- **Folder layout:**
  ```
  backend/                 Python service (own pyproject.toml + Dockerfile)
    src/chat_app/          importable package: `from chat_app.retrieval import ...`
      core/                domain models (Pydantic) + interfaces (ArticleRepository, Retriever, EmbeddingProvider, Summarizer)
      config/              settings loaded from env
      ingestion/           load → clean → dedupe → enrich metadata → chunk → index (offline)
      retrieval/           Qdrant hybrid search, filters, relevance/coverage checks
      generation/          prompt assembly, LLM client, grounding/citation checks
        prompts/           prompt templates
      api/                 FastAPI app, SSE, health check
        routes/            endpoint modules
    tests/
      unit/  integration/  eval/ (12 reference queries)  fixtures/
    scripts/               CLI entry points (e.g. run offline indexing)
  ui/                      React app, presentation only (own package.json + Dockerfile)
    public/
    src/components/  src/hooks/  src/services/ (API client)
  data/                    source dataset (stock_news.json), shared with backend
  docker-compose.yml       Qdrant + backend + ui (to be added)
  ```
- **SRP:** JSON loader, text cleaner, relevance/metadata enricher, chunker, indexer, retriever, summarizer, and presentation are separate modules.
- **DIP:** core logic depends on interfaces (e.g. `ArticleRepository`, `Retriever`, `EmbeddingProvider`, `Summarizer`), not concrete JSON files, Qdrant, or a specific LLM client.
- **OCP:** adding a ticker, data source, or analysis strategy must not require modifying existing code.
- Meaningful, consistent names; small focused functions; type hints throughout; docstrings on public functions; comments explain *why*, not *what*.
- Formatting/linting: Black + Ruff (Python), ESLint + Prettier (JS).
- **No magic numbers or strings.** File paths, API keys, model names, top-k, chunk sizes, and thresholds come from config / environment variables; keep `.env.example` up to date.

## Production readiness

- Indexing is an **offline step**, separate from query-time serving; it is **idempotent and incremental** (deterministic point IDs, e.g. hash of link + chunk index — re-running never duplicates vectors).
- LLM and embedding calls have timeouts, retries with backoff, and graceful fallbacks.
- Structured logging per query: query text, retrieved chunk IDs, latency, token usage.
- Health-check endpoint (`/health`).
- Runs reproducibly via Docker (Compose: Qdrant + API + UI) or a single command.
- Validate inputs at the API boundary (Pydantic).

## Commands

_Full commands to be filled in once code exists (ingest, serve API, run UI, test, lint)._

Fixed by the build files:
- API entry point: `chat_app.api.main:app` (backend/Dockerfile runs it with uvicorn on port 8000; `/health` is its healthcheck).
- Backend dev install: `pip install -e ".[dev]"` from `backend/`. Lint/format/type-check: `ruff check`, `black`, `mypy` (config in `backend/pyproject.toml`).
- UI (Vite + React 19): `npm run dev | build | test | lint | format` from `ui/`.
- In Docker, the UI's nginx proxies `/api/*` → `http://backend:8000/` with buffering off for SSE, so the frontend calls the API via `/api/...` and the compose service must be named `backend`.
- `data/` sits outside the backend build context; compose mounts it read-only at `/data`.
- `docker compose up --build` runs qdrant (v1.19.1, persistent `qdrant_storage` volume) → backend → ui, each waiting on the previous one's healthcheck. Default ports: UI 3000, API 8000, Qdrant 6333 (override with `UI_PORT`, `API_PORT`, `QDRANT_PORT`).
- Compose sets `QDRANT_URL` and `DATA_PATH` for the backend; backend settings must read these names. Secrets (e.g. the LLM API key) come from an optional root `.env`, and `.env.example` must list them.
- Offline indexing runs as a one-off in the backend image: `docker compose run --rm backend <index command>` (command to be defined).
