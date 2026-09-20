"""Worker loop (PRD §7): claim durable jobs, run the extract/ocr/chunk/embed/
publish stages, classify failures (permanent vs. transient), and heartbeat the
lease between units. As of M4, a full drain of one document completes four
jobs: extract, chunk, embed, and publish.
"""

from __future__ import annotations

import fcntl
import json
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
from library_rag.jobs import Claimed, Jobs
from library_rag.scan import scan_roots
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
