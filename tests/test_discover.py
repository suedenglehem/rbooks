"""Scheduled discovery (M7 slice 6): a stoppable loop over the streaming scan.

Properties under test: the stop flag is honored before a pass starts and
during the between-pass sleep; a pass never starts interrupted; unchanged
books are not re-hashed or re-enqueued (the fast path carries over from
``scan``); new books dropped between passes are picked up; and a file that
disappears is *reported* (``missing``), never deleted from the catalog.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from fixtures import make_pdf
from library_rag.config import Config
from library_rag.db import Database
from library_rag.discover import run_discovery
from library_rag.identity import normalize_path
from library_rag.jobs import Jobs
from library_rag.scan import ScanReport

_INTERVAL = 0.01  # fast enough for tests; the production default is hourly


@pytest.fixture
def src(base_config: Config) -> Path:
    """The configured source root, created (it does not pre-exist)."""
    root = base_config.paths.source_roots[0]
    root.mkdir(parents=True)
    return root


def _stop_after(
    n: int,
) -> tuple[Callable[[], bool], Callable[[list[ScanReport]], None], list[list[ScanReport]]]:
    """Stop control that ends the run right after *n* passes.

    Returns ``(stop_event, on_pass, seen)``: wire ``on_pass`` into
    ``run_discovery`` and the flag flips once *n* passes have reported, so
    the next top-of-loop check breaks the run. ``seen`` collects each pass's
    reports in order.
    """
    flag: dict[str, bool] = {"stop": False}
    seen: list[list[ScanReport]] = []

    def on_pass(reports: list[ScanReport]) -> None:
        seen.append(reports)
        if len(seen) >= n:
            flag["stop"] = True

    def stop() -> bool:
        return flag["stop"]

    return stop, on_pass, seen


def test_stops_immediately_without_a_pass(state_db: Database, base_config: Config, src: Path) -> None:
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)

    passes = run_discovery(state_db, base_config, jobs, interval_seconds=_INTERVAL, stop_event=lambda: True)

    assert passes == 0
    assert state_db.query_one("SELECT path FROM scan_state") is None
    assert jobs.counts() == {}


def test_one_pass_registers_new_book(state_db: Database, base_config: Config, src: Path) -> None:
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)
    stop, on_pass, seen = _stop_after(1)

    passes = run_discovery(
        state_db, base_config, jobs, interval_seconds=_INTERVAL, stop_event=stop, on_pass=on_pass
    )

    assert passes == 1
    report = seen[0][0]
    assert report.discovered == 1
    assert report.new_documents == 1
    assert report.jobs_enqueued == 1
    assert jobs.counts() == {"pending": 1}


def test_second_pass_unchanged_no_rehash(
    state_db: Database, base_config: Config, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)
    stop, on_pass, _ = _stop_after(1)
    run_discovery(state_db, base_config, jobs, interval_seconds=_INTERVAL, stop_event=stop, on_pass=on_pass)

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("stream_hash must not run for an unchanged file")

    monkeypatch.setattr("library_rag.scan.stream_hash", boom)
    stop2, on_pass2, seen2 = _stop_after(1)
    passes = run_discovery(
        state_db, base_config, jobs, interval_seconds=_INTERVAL, stop_event=stop2, on_pass=on_pass2
    )

    assert passes == 1
    report = seen2[0][0]
    assert report.unchanged == 1
    assert report.new_documents == 0
    assert report.jobs_enqueued == 0
    assert jobs.counts() == {"pending": 1}  # no double enqueue


def test_new_book_between_passes(state_db: Database, base_config: Config, src: Path) -> None:
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)
    state: dict[str, bool] = {"stop": False}
    seen: list[list[ScanReport]] = []

    def on_pass(reports: list[ScanReport]) -> None:
        seen.append(reports)
        if len(seen) == 1:
            make_pdf(src / "B.pdf", ["another page " * 5])  # dropped in mid-run
        if len(seen) >= 2:
            state["stop"] = True

    passes = run_discovery(
        state_db,
        base_config,
        jobs,
        interval_seconds=_INTERVAL,
        stop_event=lambda: state["stop"],
        on_pass=on_pass,
    )

    assert passes == 2
    first, second = seen[0][0], seen[1][0]
    assert first.new_documents == 1
    assert second.discovered == 2
    assert second.unchanged == 1  # A is not reprocessed
    assert second.new_documents == 1  # B was picked up without a full reprocess
    assert second.jobs_enqueued == 1
    assert jobs.counts() == {"pending": 2}


def test_sleep_between_passes_is_interruptible(state_db: Database, base_config: Config, src: Path) -> None:
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)
    stop_at = time.monotonic() + 0.15

    def stop() -> bool:
        return time.monotonic() >= stop_at

    start = time.monotonic()
    passes = run_discovery(state_db, base_config, jobs, interval_seconds=10.0, stop_event=stop)
    elapsed = time.monotonic() - start

    assert passes == 1  # the in-flight pass finished...
    assert elapsed < 1.0  # ...and the 10 s sleep was cut short, not run out


def test_missing_file_reported_not_deleted(state_db: Database, base_config: Config, src: Path) -> None:
    a = src / "A.pdf"
    make_pdf(a, ["a page of text " * 5])
    jobs = Jobs(state_db)
    stop, on_pass, _ = _stop_after(1)
    run_discovery(state_db, base_config, jobs, interval_seconds=_INTERVAL, stop_event=stop, on_pass=on_pass)

    a.unlink()
    stop2, on_pass2, seen2 = _stop_after(1)
    passes = run_discovery(
        state_db, base_config, jobs, interval_seconds=_INTERVAL, stop_event=stop2, on_pass=on_pass2
    )

    assert passes == 1
    assert normalize_path(a) in seen2[0][0].missing
    row = state_db.query_one("SELECT COUNT(*) AS n FROM path_aliases")
    assert row is not None and row["n"] == 1  # the alias row stays; coverage flags it
