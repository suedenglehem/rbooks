"""Scheduled discovery (M7 slice 6, PRD lines 15/83/181).

The ``scan`` command is a single discovery pass. ``run_discovery`` makes
discovery *scheduled*: one scan pass immediately, then a fixed interval
between passes until a stop signal, so new and changed books are picked
up automatically while a (possibly multi-week) ingestion runs.

Each pass is the same streaming discovery as ``scan`` (PRD §8A: no tree
in memory, configured paths ignored, magic-byte validation): new books
are registered and enqueued, changed books (size+mtime fast check →
re-hash) may get a new revision, and files that disappeared since the
last pass are *reported* (``missing``) — never deleted (their alias rows
stay, and the coverage report flags them as orphaned). A scheduled pass
therefore only ever adds or revises; it never removes.

Stop semantics mirror ``run_worker`` (PRD §7): the stop flag is honored
between passes and during the sleep, so a pass never starts
interrupted and an in-flight pass runs to completion (a read-only walk
with idempotent per-file upserts — even an interrupted pass is safe to
repeat).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from .config import Config
from .db import Database
from .jobs import Jobs
from .log import get_logger, log_event
from .scan import ScanReport, scan_roots

log = get_logger("discover")

__all__ = ["DEFAULT_DISCOVER_INTERVAL", "run_discovery"]

# One full pass over the operator's 34k-file library takes ~5-7 min of
# I/O (measured 2026-09-21); an hourly cadence keeps discovery overhead
# under ~10% while new books land within the hour.
DEFAULT_DISCOVER_INTERVAL = 3600.0


def _sleep_interruptible(delay: float, stop_event: Callable[[], bool] | None) -> None:
    if stop_event is None:
        time.sleep(delay)
        return
    end = time.monotonic() + delay
    while True:
        if stop_event():
            return
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.05, remaining))


def run_discovery(
    db: Database,
    cfg: Config,
    jobs: Jobs,
    *,
    interval_seconds: float = DEFAULT_DISCOVER_INTERVAL,
    stop_event: Callable[[], bool] | None = None,
    on_pass: Callable[[list[ScanReport]], None] | None = None,
) -> int:
    """Periodically re-scan all source roots (one pass, then sleep, repeat).

    *on_pass*, when given, receives each pass's per-root reports in
    order (the module logger always gets a summary line). Returns the
    number of passes executed.
    """
    passes = 0
    while True:
        if stop_event is not None and stop_event():
            break
        reports = scan_roots(db, cfg, jobs)
        passes += 1
        log_event(
            log,
            logging.INFO,
            "discovery pass complete",
            passes=passes,
            discovered=sum(r.discovered for r in reports),
            unchanged=sum(r.unchanged for r in reports),
            new_documents=sum(r.new_documents for r in reports),
            new_revisions=sum(r.new_revisions for r in reports),
            jobs=sum(r.jobs_enqueued for r in reports),
        )
        if on_pass is not None:
            on_pass(reports)
        _sleep_interruptible(interval_seconds, stop_event)
    return passes
