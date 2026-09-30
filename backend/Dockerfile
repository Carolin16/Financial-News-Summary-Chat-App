FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken

WORKDIR /app

# Built from the repo root (compose and Railway), so the dataset outside backend/ is reachable.
COPY backend/pyproject.toml ./
COPY backend/src ./src
RUN pip install .

# Bake the BM25 model and tokenizer into the image: no downloads at startup or query time.
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')" \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY backend/scripts ./scripts

# Bake the dataset in: platforms that build each service alone (e.g. Railway) can't mount it,
# and offline indexing (`chat-index`) reads it from here.
COPY data/stock_news.json /data/stock_news.json
ENV DATA_PATH=/data/stock_news.json

RUN useradd --create-home --uid 1000 app && chown -R app /app/.cache
USER app

EXPOSE 8000

# slim image has no curl, so probe /health with the stdlib
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "chat_app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
