"""Structured logging.

The application logs as JSON lines to stderr (one object per event) so that
long-running ingestion can be parsed by `jq`, grep'd, and shipped to disk
without a log framework. A human-readable formatter is available for the CLI's
`--log-format human` mode and for local debugging.

Logging is configured once per process by :func:`setup_logging`; it is idempotent
so tests that import multiple modules do not stack handlers.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Forward structured fields set via extra={"extra": {...}}.
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key not in payload:
                    payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _HumanFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)s %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict) and extra:
            base += " " + json.dumps(extra, default=str)
        return base


def setup_logging(level: int | str = "INFO", fmt: str = "json") -> None:
    """Configure root logging. Idempotent.

    Args:
        level: logging level name or int.
        fmt: "json" (default, machine-parseable) or "human" (readable).
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level if isinstance(level, int) else getattr(logging, str(level)))
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if fmt == "json" else _HumanFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level if isinstance(level, int) else getattr(logging, str(level), logging.INFO))
    # Third-party noise we would rather not drown ingestion output in.
    logging.getLogger("multipart").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    """Emit a structured event. ``fields`` are merged into the log payload.

    This is the preferred way to log from pipeline code so that events are
    greppable by key (e.g. ``job_id``, ``stage``, ``document_id``).
    """
    logger.log(level, msg, extra={"extra": fields})
