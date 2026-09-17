"""Database-commit crash hooks: before-commit rolls back, after-commit is durable."""

from __future__ import annotations

import pytest

from library_rag import crash
from library_rag.catalog import Format, register_source, source_file_count
from library_rag.crash import CrashPhase, SimulatedCrash
from library_rag.db import Database

H = "33" * 32


def _die() -> None:
    raise SimulatedCrash()


def test_crash_before_commit_rolls_back(state_db: Database) -> None:
    crash.set_hook(CrashPhase.BEFORE_DB_COMMIT, _die)
    with pytest.raises(SimulatedCrash):
        register_source(state_db, "/books/X.pdf", H, 10, Format.PDF)
    # The commit never landed, so the catalog is exactly as it was: empty.
    assert source_file_count(state_db) == 0
    assert state_db.query_one("SELECT 1 AS x FROM documents") is None
    assert state_db.query_one("SELECT 1 AS x FROM path_aliases") is None


def test_crash_after_commit_is_durable(state_db: Database) -> None:
    crash.set_hook(CrashPhase.AFTER_DB_COMMIT, _die)
    with pytest.raises(SimulatedCrash):
        register_source(state_db, "/books/X.pdf", H, 10, Format.PDF)
    # The commit already happened before the after-hook fired: the row is durable.
    assert source_file_count(state_db) == 1
    assert state_db.query_one("SELECT 1 AS x FROM documents") is not None


def test_crash_before_commit_on_job_claim_rolls_back(state_db: Database) -> None:
    from library_rag.identity import make_task_key
    from library_rag.jobs import Jobs

    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("extract", "d", "v"), "extract",
                 input_id="d", input_version="v", now=1000.0)
    # Pause inside the claim transaction: the claim write must roll back.
    crash.set_hook(CrashPhase.BEFORE_DB_COMMIT, _die)
    with pytest.raises(SimulatedCrash):
        jobs.claim("worker-1", ttl=10.0, now=1000.0)
    assert jobs.counts() == {"pending": 1}  # still pending, never became running
