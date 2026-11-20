"""Structured JSON-lines logging shared by every module.

Message text is only emitted at DEBUG by the chat pipeline; structured fields
(ids, intents, scores, latencies, token counts) are safe at any level.  Known
secret-looking fields are redacted by the formatter.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from core.config import get_settings

_RESERVED: set[str] = set(
    logging.LogRecord("", logging.NOTSET, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}

_SENSITIVE_KEYS: set[str] = {
    "api_key",
    "groq_api_key",
    "jwt_secret",
    "secret",
    "password",
    "authorization",
    "token",
}


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = "***" if key.lower() in _SENSITIVE_KEYS else value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Install a single JSON handler on the root logger (idempotent)."""
    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)

    # Model downloads emit one INFO line per HTTP request; keep the log useful.
    for noisy in ("httpx2", "httpx", "httpcore2", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


configure_logging()
