"""JSON logging: one machine-readable line per event, with `extra=` fields included."""

import json
import logging
import sys
from datetime import UTC, datetime

# Attributes every LogRecord has; anything else was passed via `extra=` and is logged.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}

# Third-party loggers that log every HTTP request at INFO; too noisy for per-query logs.
_QUIET_LOGGERS = ("httpx", "httpx2", "httpcore", "openai", "urllib3", "huggingface_hub")


class JsonFormatter(logging.Formatter):
    """Formats records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the record, its `extra` fields, and any exception."""
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str) -> None:
    """Route all logging to stdout as JSON at `level`."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
