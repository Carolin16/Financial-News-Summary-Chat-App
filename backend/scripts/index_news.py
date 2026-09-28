"""Run offline indexing: `python scripts/index_news.py` (same as the `chat-index` command)."""

from chat_app.ingestion.cli import main

if __name__ == "__main__":
    main()
