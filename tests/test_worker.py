"""Worker loop (PRD §7): claim durable jobs, run the extract/ocr/chunk/embed/
publish stages, classify failures (permanent vs. transient), and heartbeat the
lease between units. As of M4, a full drain of one document completes four
jobs: extract, chunk, embed, and publish.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest

from fixtures import make_pdf
from library_rag.config import Config
from library_rag.db import Database
from library_rag.embeddings import (
    FakeEmbedder,
    checkpoint_path,
    embedding_sha,
    write_checkpoint,
)
from library_rag.identity import make_task_key
from library_rag.indexing import FakeQdrant, RealQdrantOps
from library_rag.jobs import BOOK_BUDGET_PARKED, Claimed, Jobs
from library_rag.log import JobLogCapture, log_event, prune_job_logs
from library_rag.scan import STAGE_EXTRACT, scan_roots
from library_rag.worker import run_worker


@pytest.fixture
def src(base_config: Config) -> Path:
    root = base_config.paths.source_roots[0]
    root.mkdir(parents=True)
    return root


def _last_job(db: Database) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM jobs ORDER BY job_id DESC")
    assert row is not None
    return dict(row)


def _job(db: Database, stage: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM jobs WHERE stage = ?", (stage,))
    assert row is not None
    return dict(row)


def test_worker_end_to_end(state_db: Database, base_config: Config, src: Path) -> None:
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)
    report = scan_roots(state_db, base_config, jobs)[0]
    assert report.new_documents == 1

    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    assert run_worker(state_db, base_config, once=True, poll_delay=0, qdrant=q, embedder=emb) == 4
    assert jobs.counts() == {"succeeded": 4}

    run = state_db.query_one(
        "SELECT state, unit_count, chunk_fingerprint FROM extraction_runs"
    )
    assert run is not None
    assert run["state"] == "succeeded"
    assert int(run["unit_count"]) == 1
    extract_manifest = json.loads(_job(state_db, "extract")["output_manifest"])
    assert set(extract_manifest) == {"run_id", "units"}
    assert extract_manifest["units"] == 1
    chunk_manifest = json.loads(_job(state_db, "chunk")["output_manifest"])
    assert set(chunk_manifest) == {"run_id", "chunks", "fingerprint"}
    assert chunk_manifest["chunks"] == 1
    assert run["chunk_fingerprint"] == chunk_manifest["fingerprint"]
    n_chunks = state_db.query_one("SELECT COUNT(*) AS n FROM chunks")
    assert n_chunks is not None and int(n_chunks["n"]) == 1


def test_worker_corrupt_permanent_failed(state_db: Database, base_config: Config, src: Path) -> None:
    # Magic bytes say PDF; PyMuPDF cannot open it => permanent "corrupt".
    (src / "bad.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)

    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1

    assert jobs.counts() == {"permanent_failed": 1}
    job = _last_job(state_db)
    assert job["error_category"] == "corrupt"


def test_worker_heartbeats_between_units(
    state_db: Database, base_config: Config, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 25 pages => a heartbeat at pages 10 and 20 (indices 9 and 19).
    make_pdf(src / "big.pdf", [f"page {i} " + "word " * 10 for i in range(25)])
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)

    beats: list[int] = []

    def fake_heartbeat(self: Jobs, job: Claimed, ttl: float, now: float | None = None) -> None:
        beats.append(job.job_id)

    monkeypatch.setattr(Jobs, "heartbeat", fake_heartbeat)
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    # The chunk stage is pure in-memory work and never heartbeats; embed
    # heartbeats once per batch (one chunk => one batch) and publish heartbeats
    # once, so: 2 extract + 1 embed + 1 publish.
    assert run_worker(state_db, base_config, once=True, poll_delay=0, qdrant=q, embedder=emb) == 4

    assert len(beats) == 4
    assert jobs.counts() == {"succeeded": 4}


def test_worker_unknown_stage_permanent(state_db: Database, base_config: Config) -> None:
    jobs = Jobs(state_db)
    jobs.enqueue(
        make_task_key("frobnicate", "x", "v1"), "frobnicate", input_id="x", input_version="v1"
    )

    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1

    assert jobs.counts() == {"permanent_failed": 1}
    assert _last_job(state_db)["error_category"] == "unknown_stage"


def test_worker_missing_revision_permanent(state_db: Database, base_config: Config) -> None:
    jobs = Jobs(state_db)
    jobs.enqueue(
        make_task_key("extract", "no-such-rev", "0" * 64),
        "extract",
        input_id="no-such-rev",
        input_version="0" * 64,
    )

    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1

    assert jobs.counts() == {"permanent_failed": 1}
    job = _last_job(state_db)
    assert job["error_category"] == "missing_source"


def test_worker_single_local_qdrant_client_per_run(
    state_db: Database,
    base_config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression for the drain-worker death: a local (file-based) Qdrant
    # client holds an exclusive lock on the storage folder for its whole
    # lifetime, so a *second* client in the same process is refused by its
    # own first lock ("Storage folder ... is already accessed by another
    # instance of Qdrant client"). The worker must therefore open at most
    # ONE client per run, shared by the reconcile pass and every publish
    # job, and close it before returning.
    qdrant_dir = tmp_path / "qdrant"
    base_config.services.qdrant_path = str(qdrant_dir)
    # publish computes the embedding key even for an empty generation, so
    # the embedder must be configured (fake is the test idiom).
    base_config.embedding.fake = True

    # One active revision with a succeeded, zero-unit extraction run: both
    # seeded publish jobs reach the client without needing any artifacts,
    # and the zero-chunk publish verifies cleanly (expected 0 points, got 0).
    ts = 1_000_000.0
    state_db.execute(
        "INSERT INTO documents (doc_id, anchor_sha256, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("doc-1", "a" * 64, ts, ts),
    )
    state_db.execute(
        """
        INSERT INTO source_revisions
            (rev_id, doc_id, sha256, size_bytes, format, archive_relpath,
             first_path, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
        """,
        ("rev-1", "doc-1", "b" * 64, 10, "pdf", "rev-1.pdf", "/books/rev-1.pdf", ts),
    )
    state_db.execute(
        """
        INSERT INTO extraction_runs
            (run_id, rev_id, doc_id, parser_version, settings_sha, unit_count,
             state, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 0, 'succeeded', ?, ?)
        """,
        ("run-1", "rev-1", "doc-1", "pymupdf-1.0", "c" * 64, ts, ts),
    )

    # Publication loads vectors from the embedding checkpoints even for an
    # empty generation, so seed the consistent zero-batch the embed stage
    # would have written: one row plus its (empty) checkpoint artifact.
    emb_sha = embedding_sha(base_config)
    ckpt = checkpoint_path(base_config.paths.artifact_root, "run-1", emb_sha, 0)
    vector_sha = write_checkpoint(
        ckpt, [], base_config.embedding.dimensions, {"chunk_ids": []}
    )
    state_db.execute(
        """
        INSERT INTO embedding_batches
            (batch_id, run_id, model_revision, embedding_sha, batch_index,
             chunk_ids, vector_sha256, artifact_relpath, created_at)
        VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)
        """,
        (
            "emb-1",
            "run-1",
            "fake-v1",
            emb_sha,
            "[]",
            vector_sha,
            str(ckpt.relative_to(base_config.paths.artifact_root)),
            ts,
        ),
    )

    # Count every real client the worker constructs during the run.
    created: list[RealQdrantOps] = []
    real_ctor = RealQdrantOps

    def counting_ctor(cfg: Config, *args: Any, **kwargs: Any) -> RealQdrantOps:
        ops = real_ctor(cfg, *args, **kwargs)
        created.append(ops)
        return ops

    monkeypatch.setattr("library_rag.worker.RealQdrantOps", counting_ctor)

    jobs = Jobs(state_db)
    for version in ("v1", "v2"):
        jobs.enqueue(
            make_task_key("publish", "rev-1", version),
            "publish",
            input_id="rev-1",
            input_version=version,
        )

    # No qdrant argument: the worker must open its own client. Reconcile is
    # off so the seeded zero-fingerprint run is not re-queued as chunk work.
    completed = run_worker(
        state_db, base_config, once=True, poll_delay=0, reconcile_on_start=False
    )

    # The invariant the fix establishes: exactly one client for the whole
    # run, shared by both publish jobs. (The buggy code constructed one per
    # publish job; the second local client crashed on the first one's lock.)
    assert len(created) == 1
    assert completed == 2
    assert Jobs(state_db).counts() == {"succeeded": 2}

    # And the worker closed its client: the storage lock is free again, so a
    # later client in the same process can open the store.
    lock_path = qdrant_dir / ".lock"
    with open(lock_path, "rb") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


# --- per-job verbose logs ---------------------------------------------------
# While a job runs, DEBUG records are captured to
# <state_root>/job_logs/<job_id>.attempt<N>.log alongside the short stderr
# log. The file is KEPT when the job ended in a real failure and FLUSHED
# otherwise (success, plain deferral, or lost lease).


def _job_logs_dir(base_config: Config) -> Path:
    return base_config.paths.state_root / "job_logs"


def test_job_logs_flushed_on_success(
    state_db: Database, base_config: Config, src: Path
) -> None:
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)

    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    assert run_worker(state_db, base_config, once=True, poll_delay=0, qdrant=q, embedder=emb) == 4
    assert jobs.counts() == {"succeeded": 4}

    # Every job succeeded, so every per-job log was flushed: no files left.
    d = _job_logs_dir(base_config)
    if d.exists():
        assert list(d.iterdir()) == []


def test_job_logs_kept_on_permanent_failure(
    state_db: Database, base_config: Config, src: Path
) -> None:
    # Magic bytes say PDF; PyMuPDF cannot open it => permanent "corrupt".
    (src / "bad.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)

    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1
    assert jobs.counts() == {"permanent_failed": 1}

    d = _job_logs_dir(base_config)
    files = sorted(d.iterdir())
    assert len(files) == 1
    assert files[0].name.endswith(".attempt1.log")
    # The captured file carries the DEBUG trail: the start event and the
    # classified failure (category + detail serialized in the payload).
    text = files[0].read_text(encoding="utf-8")
    assert "extract: start" in text
    assert "extract: failed" in text
    assert "corrupt" in text


def test_job_log_capture_level_window(tmp_path: Path) -> None:
    root = logging.getLogger()
    prev_level = root.level
    n_handlers = len(root.handlers)
    logger = logging.getLogger("library_rag.worker")
    path = tmp_path / "job.log"

    log_event(logger, logging.DEBUG, "outside-before", marker="before")
    with JobLogCapture(path):
        assert root.level == logging.DEBUG
        assert len(root.handlers) == n_handlers + 1
        log_event(logger, logging.DEBUG, "captured", marker="inside")
    # The window is restored: level and handler count back to what they were.
    assert root.level == prev_level
    assert len(root.handlers) == n_handlers

    text = path.read_text(encoding="utf-8")
    assert "captured" in text
    assert "inside" in text  # structured fields land in the line too
    assert "before" not in text  # nothing captured before the window opened

    log_event(logger, logging.DEBUG, "outside-after", marker="after")
    assert "after" not in path.read_text(encoding="utf-8")


def test_prune_job_logs_keeps_newest(tmp_path: Path) -> None:
    d = tmp_path / "job_logs"
    d.mkdir()
    base = 1_000_000.0
    for i in range(505):
        p = d / f"{i}.log"
        p.write_text(f"log {i}\n")
        ts = base + i  # i=0 oldest ... i=504 newest
        os.utime(p, (ts, ts))

    removed = prune_job_logs(d, limit=500)
    assert removed == 5
    names = {p.name for p in d.iterdir()}
    assert len(names) == 500
    for i in range(5):  # the five oldest are gone
        assert f"{i}.log" not in names
    assert "5.log" in names
    assert "504.log" in names


def test_job_log_kept_on_retryable_flushed_after_success(
    state_db: Database, base_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A transient failure (category set) keeps its log; the retry that then
    # succeeds flushes its own, leaving the failure's log for debugging.
    jobs = Jobs(state_db)
    jobs.enqueue(
        make_task_key("extract", "rev-x", "v1"),
        "extract",
        input_id="rev-x",
        input_version="v1",
    )

    calls = {"n": 0}

    def fake_extract(
        db: Database, cfg: Config, jobs: Jobs, job: Claimed, lease_ttl: float
    ) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            # A real backoff keeps attempt 2 out of the first drain window.
            jobs.fail(job, "worker_error", "boom", transient=True, backoff=300.0)
        else:
            jobs.succeed(job, "{}")
        return True

    monkeypatch.setattr("library_rag.worker._run_extract", fake_extract)

    # Attempt 1: retryable_failed WITH a category => the log is kept.
    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1
    job_row = _last_job(state_db)
    assert job_row["state"] == "retryable_failed"
    assert job_row["error_category"] == "worker_error"
    d = _job_logs_dir(base_config)
    files = sorted(p.name for p in d.iterdir())
    assert files == [f"{job_row['job_id']}.attempt1.log"]

    # The operator requeues the retryable job; attempt 2 succeeds => flushed.
    assert jobs.retry() == 1
    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1
    assert Jobs(state_db).counts() == {"succeeded": 1}
    files = sorted(p.name for p in d.iterdir())
    assert files == [f"{job_row['job_id']}.attempt1.log"]


def test_job_log_flushed_on_deferral(state_db: Database, base_config: Config) -> None:
    # A publish job whose revision has no settled extraction run defers.
    # Deferral is not a failure: error_category is cleared, so the log is
    # flushed even though the job sits in retryable_failed.
    ts = 1_000_000.0
    state_db.execute(
        "INSERT INTO documents (doc_id, anchor_sha256, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("doc-1", "a" * 64, ts, ts),
    )
    state_db.execute(
        """
        INSERT INTO source_revisions
            (rev_id, doc_id, sha256, size_bytes, format, archive_relpath,
             first_path, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
        """,
        ("rev-1", "doc-1", "b" * 64, 10, "pdf", "rev-1.pdf", "/books/rev-1.pdf", ts),
    )
    jobs = Jobs(state_db)
    jobs.enqueue(
        make_task_key("publish", "rev-1", "v1"),
        "publish",
        input_id="rev-1",
        input_version="v1",
    )

    q = FakeQdrant(base_config.embedding.dimensions)
    assert run_worker(
        state_db, base_config, once=True, poll_delay=0, qdrant=q, reconcile_on_start=False
    ) == 1

    job = _job(state_db, "publish")
    assert job["state"] == "retryable_failed"
    assert job["error_category"] is None
    d = _job_logs_dir(base_config)
    if d.exists():
        assert list(d.iterdir()) == []


# --- bounded runs (--max-books) -------------------------------------------------


def test_max_books_bounded_run(state_db: Database, base_config: Config, src: Path) -> None:
    # Three books, budget of two: A and B run end-to-end (extract + chunk +
    # embed + publish each). In live stats mode publishing B commits a NEW
    # corpus-stats epoch, so the fan-out re-publishes A under it — that
    # convergence job is part of finishing B. C's extract is parked without
    # consuming an attempt and must not hold the run open.
    for name in ("A.pdf", "B.pdf", "C.pdf"):
        make_pdf(src / name, [f"{name} page of text " * 5])
    jobs = Jobs(state_db)
    report = scan_roots(state_db, base_config, jobs)[0]
    assert report.new_documents == 3

    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    completed = run_worker(
        state_db, base_config, poll_delay=0, max_books=2, qdrant=q, embedder=emb
    )

    assert completed == 9  # 4 + 4 + A's epoch fan-out republish after B publishes
    assert jobs.counts() == {"succeeded": 9, "retryable_failed": 1}

    parked = state_db.query_one(
        "SELECT * FROM jobs WHERE error_detail = ?", (BOOK_BUDGET_PARKED,)
    )
    assert parked is not None
    assert parked["stage"] == STAGE_EXTRACT
    assert int(parked["attempts"]) == 0  # release restored the claim's increment


def test_max_books_waits_out_deferral_gap(
    state_db: Database, base_config: Config, src: Path
) -> None:
    # The drain-exit waits out short deferral gaps: an in-budget extract that
    # is not yet due keeps the run open until it becomes claimable and then
    # finishes end-to-end. A plain *once*-style exit on the first empty claim
    # would leave the book unstarted.
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)[0]

    state_db.execute(
        "UPDATE jobs SET state = 'retryable_failed', next_attempt_at = ? WHERE stage = ?",
        (time.time() + 0.3, STAGE_EXTRACT),
    )

    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    assert run_worker(
        state_db, base_config, poll_delay=0, max_books=1, qdrant=q, embedder=emb
    ) == 4
    assert jobs.counts() == {"succeeded": 4}


def test_max_books_ignores_parked_budget_jobs(state_db: Database, base_config: Config) -> None:
    # A budget-parked extract from a previous bounded run must not hold the new
    # run open: it is excluded from the drain-exit wait (each release re-arms
    # one — waiting on them would loop forever). With nothing else in the queue
    # the run exits immediately, having processed nothing.
    jobs = Jobs(state_db)
    jobs.enqueue(
        make_task_key(STAGE_EXTRACT, "rev-parked", "v1"),
        STAGE_EXTRACT,
        input_id="rev-parked",
        input_version="v1",
    )
    state_db.execute(
        "UPDATE jobs SET state = 'retryable_failed', next_attempt_at = ?, error_detail = ?"
        " WHERE stage = ?",
        (time.time() + 30.0, BOOK_BUDGET_PARKED, STAGE_EXTRACT),
    )

    started = time.monotonic()
    assert run_worker(state_db, base_config, poll_delay=0, max_books=1) == 0
    assert time.monotonic() - started < 2.0
