"""Structured logging.

The application logs as JSON lines to stderr (one object per event) so that
long-running ingestion can be parsed by `jq`, grep'd, and shipped to disk
without a log framework. A human-readable formatter is available for the CLI's
`--log-format human` mode and for local debugging.

Logging is configured once per process by :func:`setup_logging`; it is idempotent
so tests that import multiple modules do not stack handlers.

Per-job verbose logs: the worker runs the short log alongside a DEBUG-level
capture to one file per job (``<state_root>/job_logs/<job_id>.attempt<N>.log``).
A failed job keeps its file for debugging; a successful or deferred job has its
file flushed (deleted). :class:`JobLogCapture` does the capture,
:func:`prune_job_logs` caps how many failure logs accumulate.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
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
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(level: int | str = "INFO", fmt: str = "json") -> None:
    """Configure root logging. Idempotent.

    Args:
        level: logging level name or int.
        fmt: "json" (default, machine-parseable) or "human" (readable).
    """
    global _CONFIGURED
    resolved = level if isinstance(level, int) else getattr(logging, str(level), logging.INFO)
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(resolved)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if fmt == "json" else _HumanFormatter())
    # The handler carries the level itself, not just via the root logger:
    # JobLogCapture temporarily lowers the ROOT level to DEBUG for per-job
    # files, and this gate keeps the short log short during that window.
    handler.setLevel(resolved)
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)
    # Third-party noise we would rather not drown ingestion output in.
    logging.getLogger("multipart").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    """Emit a structured event. ``fields`` are merged into the log payload.

    This is the preferred way to log from pipeline code so that events are
    greppable by key (e.g. ``job_id``, ``stage``, ``document_id``).

    ``exc_info=True`` is pulled out of *fields* and passed to the logging
    machinery so the traceback lands in ``record.exc_info`` (both formatters
    rely on it); without this it would be silently merged into the payload.
    """
    exc_info = fields.pop("exc_info", None)
    logger.log(level, msg, extra={"extra": fields}, exc_info=exc_info)


class JobLogCapture:
    """Capture DEBUG-level logging to a per-job file while active.

    The ROOT logger's level gates record *creation*, so a DEBUG FileHandler
    alone would capture nothing while the root sits at INFO. The capture
    therefore lowers the root level to DEBUG for its window and adds a
    DEBUG-level FileHandler (human formatter), then restores both on exit.
    Other handlers keep their own levels, so the short stderr log stays
    short for the whole job.

    The file is opened fresh (``mode="w"``) and written to under its final
    name from the first line, so a process killed mid-job still leaves the
    log behind: a mid-job crash is exactly the case the log exists for.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handler: logging.FileHandler | None = None
        self._prev_root_level: int | None = None

    def __enter__(self) -> JobLogCapture:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handler = logging.FileHandler(self._path, mode="w", encoding="utf-8")
        self._handler.setLevel(logging.DEBUG)
        self._handler.setFormatter(_HumanFormatter())
        root = logging.getLogger()
        self._prev_root_level = root.level
        root.setLevel(logging.DEBUG)
        root.addHandler(self._handler)
        return self

    def __exit__(self, *exc_info: object) -> None:
        assert self._handler is not None and self._prev_root_level is not None
        root = logging.getLogger()
        root.removeHandler(self._handler)
        self._handler.close()
        root.setLevel(self._prev_root_level)
        self._handler = None
        self._prev_root_level = None


def prune_job_logs(job_logs_dir: Path, limit: int = 500) -> int:
    """Keep at most *limit* newest job log files; delete the rest, oldest first.

    Failure logs are kept forever by design, so without a cap a long campaign
    would accumulate one file per failed job. Best-effort: I/O errors on a
    file never propagate; the queue must not be blocked by log housekeeping.
    Returns the number of files removed.
    """
    try:
        files = [p for p in job_logs_dir.iterdir() if p.is_file()]
    except OSError:
        return 0
    if len(files) <= limit:
        return 0
    files.sort(key=lambda p: p.stat().st_mtime)
    removed = 0
    for p in files[: len(files) - limit]:
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed
