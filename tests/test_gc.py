"""M7 GC with reference checks (PRD §14).

GC must never touch an object a catalog row still references, must skip
recently-modified files (the scan copies archive bytes *before* committing
the revision row), must refuse while a worker is running, and must default
to a dry-run report. Real embedded (local) Qdrant storage is used, as in the
backup tests: no server, no Docker, no GPU.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest

from fixtures import ingest_and_publish, make_pdf
from library_rag.config import Config
from library_rag.db import Database
from library_rag.embeddings import FakeEmbedder
from library_rag.gc import GcError, run_gc
from library_rag.indexing import RealQdrantOps

_PAGES: list[str | None] = [
    "Chapter one introduces the lighthouse keeper.",
    "Chapter two follows the keeper's daughter to sea.",
    "Chapter three ends with the storm and the lamp.",
]

NOW = 1_700_000_000.0  # fixed "current time" for the aging helper


@pytest.fixture
def local_library(
    state_db: Database, base_config: Config
) -> tuple[Database, Config]:
    """One real book (3 pages) scanned and published into embedded Qdrant.

    The Qdrant client is closed before yielding: GC refuses while a job is
    running, and the publish job row only clears once the worker exits.
    """
    base_config.embedding.fake = True
    base_config.services.qdrant_path = str(base_config.paths.qdrant_root)
    q = RealQdrantOps(base_config)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    src = base_config.paths.source_roots[0] / "book.pdf"
    make_pdf(src, _PAGES)
    try:
        ingest_and_publish(state_db, base_config, src, qdrant=q, embedder=emb)
    finally:
        q.close()
    return state_db, base_config


def _age(path: Path) -> None:
    """Make *path* look one hour old (past the 600 s default grace)."""
    os.utime(path, (NOW - 3600, NOW - 3600))


def _age_tree(root: Path) -> None:
    if root.is_dir():
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                _age(Path(dirpath) / name)


def _orphan_sha() -> str:
    return hashlib.sha256(b"orphan archive bytes").hexdigest()


def _active_archive_relpath(db: Database) -> str:
    row = db.query_one("SELECT archive_relpath FROM source_revisions WHERE is_active = 1")
    assert row is not None
    return str(row["archive_relpath"])


# --- 1. Clean library: every referenced object is protected --------------------


def test_clean_library_has_no_garbage(
    local_library: tuple[Database, Config],
) -> None:
    db, cfg = local_library
    # Age every real object so nothing is protected by the grace window: the
    # only thing that must protect them is the catalog reference.
    _age_tree(cfg.paths.archive_root)
    _age_tree(cfg.paths.artifact_root)
    _age_tree(cfg.paths.state_root / "job_logs")

    report = run_gc(db, cfg, now=NOW)
    assert report.candidates == []
    assert report.executed is False

    report = run_gc(db, cfg, execute=True, now=NOW)
    assert report.candidates == []
    assert report.deleted == 0
    assert (cfg.paths.archive_root / _active_archive_relpath(db)).is_file()
    # Referenced checkpoints and their .json sidecars survived.
    for row in db.query("SELECT artifact_relpath FROM embedding_batches"):
        p = cfg.paths.artifact_root / row["artifact_relpath"]
        assert p.is_file(), row["artifact_relpath"]


# --- 2. Orphan archive objects --------------------------------------------------


def test_dry_run_reports_but_keeps(
    local_library: tuple[Database, Config],
) -> None:
    db, cfg = local_library
    sha = _orphan_sha()
    p = cfg.paths.archive_root / sha[:2] / sha
    p.parent.mkdir(parents=True)
    p.write_bytes(b"orphan archive bytes")
    _age(p)

    report = run_gc(db, cfg, now=NOW)
    assert [
        (c.kind, c.relpath, c.reason, c.size_bytes) for c in report.candidates
    ] == [("archive", f"{sha[:2]}/{sha}", "no source_revisions reference", len(p.read_bytes()))]
    assert p.is_file()  # dry-run deletes nothing


def test_execute_deletes_orphans_and_empty_dirs(
    local_library: tuple[Database, Config],
) -> None:
    db, cfg = local_library
    sha = _orphan_sha()
    p = cfg.paths.archive_root / sha[:2] / sha
    p.parent.mkdir(parents=True)
    p.write_bytes(b"orphan archive bytes")
    _age(p)

    report = run_gc(db, cfg, execute=True, now=NOW)
    assert report.deleted == 1
    assert report.bytes_reclaimed == len(b"orphan archive bytes")
    assert not p.is_file()
    assert not (cfg.paths.archive_root / sha[:2]).is_dir()  # emptied dir removed
    # The live book's object is untouched.
    assert (cfg.paths.archive_root / _active_archive_relpath(db)).is_file()


def test_malformed_archive_path_is_candidate(
    local_library: tuple[Database, Config],
) -> None:
    db, cfg = local_library
    junk = cfg.paths.archive_root / "not-an-object.txt"
    junk.write_text("x", encoding="utf-8")
    _age(junk)

    report = run_gc(db, cfg, now=NOW)
    assert [(c.kind, c.reason) for c in report.candidates] == [
        ("archive", "malformed archive path")
    ]


# --- 3. Orphan artifacts ---------------------------------------------------------


def test_orphan_extract_artifact(
    local_library: tuple[Database, Config],
) -> None:
    db, cfg = local_library
    p = cfg.paths.artifact_root / "extract" / "deadbeef" / "u000001.json.gz"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\x1f\x8b orphan unit")
    _age(p)

    report = run_gc(db, cfg, now=NOW)
    assert [(c.kind, c.reason) for c in report.candidates] == [
        ("artifact", "no catalog reference")
    ]
    report = run_gc(db, cfg, execute=True, now=NOW)
    assert report.deleted == 1
    assert not p.is_file()
    assert not (cfg.paths.artifact_root / "extract" / "deadbeef").is_dir()
    # The book's real extract units are untouched.
    units = list(db.query("SELECT artifact_relpath FROM source_units"))
    assert units
    assert all((cfg.paths.artifact_root / u["artifact_relpath"]).is_file() for u in units)


def test_orphan_embedding_checkpoint(
    local_library: tuple[Database, Config],
) -> None:
    db, cfg = local_library
    base = cfg.paths.artifact_root / "embeddings" / "run-x" / ("y" * 64)
    base.mkdir(parents=True)
    bin_p = base / "batch_00000.bin"
    json_p = base / "batch_00000.json"
    bin_p.write_bytes(b"\x00\x01 orphan batch")
    json_p.write_text("{}", encoding="utf-8")
    _age(bin_p)
    _age(json_p)

    report = run_gc(db, cfg, now=NOW)
    rels = sorted(c.relpath for c in report.candidates)
    # An orphaned .bin does not protect its sidecar: both are collectible.
    assert rels == [f"embeddings/run-x/{'y' * 64}/batch_00000.bin",
                   f"embeddings/run-x/{'y' * 64}/batch_00000.json"]
    report = run_gc(db, cfg, execute=True, now=NOW)
    assert report.deleted == 2
    assert not bin_p.is_file() and not json_p.is_file()
    assert not (cfg.paths.artifact_root / "embeddings" / "run-x").is_dir()


# --- 4. Stale job logs ------------------------------------------------------------


def test_stale_job_log(
    local_library: tuple[Database, Config],
) -> None:
    db, cfg = local_library
    logs = cfg.paths.state_root / "job_logs"
    live = db.query_one("SELECT MIN(job_id) AS n FROM jobs")
    assert live is not None
    stale = logs / "999999.attempt1.log"
    live_log = logs / f"{live['n']}.attempt1.log"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("kept failure log for a deleted job", encoding="utf-8")
    live_log.write_text("live job log", encoding="utf-8")
    _age(stale)
    _age(live_log)

    report = run_gc(db, cfg, now=NOW)
    assert [(c.kind, c.relpath, c.reason) for c in report.candidates] == [
        ("job_log", "999999.attempt1.log", "job row gone")
    ]
    report = run_gc(db, cfg, execute=True, now=NOW)
    assert report.deleted == 1
    assert not stale.is_file()
    assert live_log.is_file()  # the live job's log is protected by its row
    # Unrecognized filenames are never collected.
    stray = logs / "notes.txt"
    stray.write_text("operator scratch", encoding="utf-8")
    _age(stray)
    assert run_gc(db, cfg, now=NOW).candidates == []


# --- 5. Grace period and refusal ---------------------------------------------------


def test_grace_skips_recent_files(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    sha = _orphan_sha()
    p = cfg.paths.archive_root / sha[:2] / sha
    p.parent.mkdir(parents=True)
    p.write_bytes(b"orphan archive bytes")
    # Fresh (un-aged): inside the default 600 s grace.
    assert run_gc(db, cfg, now=time.time()).candidates == []
    # With no grace window the same file is collectible.
    assert len(run_gc(db, cfg, grace_seconds=0, now=time.time()).candidates) == 1
    with pytest.raises(GcError, match="grace"):
        run_gc(db, cfg, grace_seconds=-1)


def test_refuses_while_job_running(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    db.execute("UPDATE jobs SET state = 'running' WHERE job_id = (SELECT MIN(job_id) FROM jobs)")
    try:
        with pytest.raises(GcError, match="running"):
            run_gc(db, cfg)
    finally:
        db.execute(
            "UPDATE jobs SET state = 'succeeded' WHERE job_id = (SELECT MIN(job_id) FROM jobs)"
        )
