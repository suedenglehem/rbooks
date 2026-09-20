"""M7 explicit removal (PRD §14).

A removed document leaves nothing reachable to the rest of the system: all of
its index points (every generation), every catalog row, and its archive
object once no ``source_revisions`` row references the SHA anymore. Source
files are never touched (re-scan re-registers deterministically); artifact
files are deliberately left for ``gc``. Dry-run is the default and reports
before-counts; ``execute`` performs the removal and then *verifies* that every
filtered table is empty and the point count is zero. Real embedded (local)
Qdrant storage is used, as in the GC tests: no server, no Docker, no GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from fixtures import ingest_and_publish, make_pdf
from library_rag.archive import ingest_source
from library_rag.catalog import Format, register_source
from library_rag.config import Config
from library_rag.db import Database
from library_rag.embeddings import FakeEmbedder
from library_rag.identity import normalize_path
from library_rag.indexing import FakeQdrant, FieldCond, IndexFilter, RealQdrantOps
from library_rag.removal import RemovalError, remove_document, resolve_target

_PAGES: list[str | None] = [
    "Chapter one introduces the lighthouse keeper.",
    "Chapter two follows the keeper's daughter to sea.",
    "Chapter three ends with the storm and the lamp.",
]


def _doc_filter(doc_id: str) -> IndexFilter:
    return IndexFilter.all(FieldCond("doc_id", "eq", doc_id))


def _count(db: Database, sql: str, params: Sequence[Any] = ()) -> int:
    row = db.query_one(sql, params)
    return int(row["n"]) if row is not None else 0


def _doc_id(db: Database) -> str:
    row = db.query_one("SELECT doc_id FROM documents")
    assert row is not None
    return str(row["doc_id"])


@pytest.fixture
def local_library(
    state_db: Database, base_config: Config
) -> tuple[Database, Config]:
    """One real book (3 pages) scanned and published into embedded Qdrant.

    The Qdrant client is closed before yielding: removal refuses while a job
    is running, and the publish job row only clears once the worker exits.
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


# --- 1. Dry run: report without touching anything ------------------------------


def test_dry_run_reports_only(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    doc = _doc_id(db)
    q = RealQdrantOps(cfg)
    try:
        live = q.count(_doc_filter(doc))
        report = remove_document(db, cfg, q, doc)
    finally:
        q.close()

    assert report.executed is False
    assert report.verified is False
    assert report.points == live > 0
    assert report.publications == 1
    assert report.revisions == 1
    assert report.extraction_runs == 1
    assert report.source_units == 3
    assert report.chunks >= 1
    assert report.embedding_batches >= 1
    assert report.aliases == 1
    assert report.scan_state_rows == 1
    assert len(report.archive_objects) == 1
    assert report.archive_kept == []
    # Nothing was touched: catalog rows and the archive object are intact.
    assert db.query_one("SELECT doc_id FROM documents WHERE doc_id = ?", (doc,)) is not None
    assert _count(db, "SELECT COUNT(*) AS n FROM chunks") >= 1
    rev = db.query_one("SELECT archive_relpath FROM source_revisions")
    assert rev is not None
    assert (cfg.paths.archive_root / str(rev["archive_relpath"])).is_file()


# --- 2. Execute: points, catalog rows, archive object --------------------------


def test_execute_removes_everything(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    doc = _doc_id(db)
    rev = db.query_one("SELECT sha256, size_bytes, archive_relpath FROM source_revisions")
    assert rev is not None
    sha = str(rev["sha256"])
    size = int(rev["size_bytes"])
    archive_rel = str(rev["archive_relpath"])
    unit = db.query_one("SELECT artifact_relpath FROM source_units")
    assert unit is not None
    unit_rel = str(unit["artifact_relpath"])
    batch = db.query_one("SELECT artifact_relpath FROM embedding_batches")
    assert batch is not None
    batch_rel = str(batch["artifact_relpath"])

    q = RealQdrantOps(cfg)
    try:
        report = remove_document(db, cfg, q, doc, execute=True)
    finally:
        q.close()

    assert report.executed is True
    assert report.verified is True
    assert report.points > 0
    assert report.archive_objects == [{"sha256": sha, "size_bytes": size}]
    assert report.archive_kept == []
    # The fixture holds exactly one book, so every table must now be empty.
    assert _count(db, "SELECT COUNT(*) AS n FROM documents") == 0
    assert _count(db, "SELECT COUNT(*) AS n FROM source_revisions") == 0
    assert _count(db, "SELECT COUNT(*) AS n FROM path_aliases") == 0
    assert _count(db, "SELECT COUNT(*) AS n FROM publications") == 0
    assert _count(db, "SELECT COUNT(*) AS n FROM extraction_runs") == 0
    assert _count(db, "SELECT COUNT(*) AS n FROM source_units") == 0
    assert _count(db, "SELECT COUNT(*) AS n FROM chunks") == 0
    assert _count(db, "SELECT COUNT(*) AS n FROM embedding_batches") == 0
    assert _count(db, "SELECT COUNT(*) AS n FROM index_generations") == 0
    assert _count(db, "SELECT COUNT(*) AS n FROM scan_state") == 0
    # The archive object is gone; artifact files remain for gc to reclaim.
    assert not (cfg.paths.archive_root / archive_rel).is_file()
    assert (cfg.paths.artifact_root / unit_rel).is_file()
    assert (cfg.paths.artifact_root / batch_rel).is_file()
    # The source file is never touched (PRD invariant).
    assert (cfg.paths.source_roots[0] / "book.pdf").is_file()


# --- 3. Target resolution -------------------------------------------------------


def test_resolve_by_path_and_doc_id(local_library: tuple[Database, Config]) -> None:
    db, _ = local_library
    doc = _doc_id(db)
    assert resolve_target(db, doc) == doc
    assert resolve_target(db, doc.upper()) == doc  # UUID match is case-insensitive
    rev = db.query_one("SELECT first_path FROM source_revisions")
    assert rev is not None
    assert resolve_target(db, str(rev["first_path"])) == doc


def test_unknown_target(local_library: tuple[Database, Config]) -> None:
    db, _ = local_library
    with pytest.raises(RemovalError, match="no document"):
        resolve_target(db, "/nowhere/missing.pdf")
    with pytest.raises(RemovalError, match="no document"):
        resolve_target(db, "01234567-89ab-4cde-8f01-23456789abcd")


# --- 4. Refusals ----------------------------------------------------------------


def test_refuses_while_job_running(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    doc = _doc_id(db)
    db.execute(
        "UPDATE jobs SET state = 'running' WHERE job_id = (SELECT MIN(job_id) FROM jobs)"
    )
    try:
        with pytest.raises(RemovalError, match="running"):
            remove_document(db, cfg, None, doc)
    finally:
        db.execute(
            "UPDATE jobs SET state = 'succeeded' "
            "WHERE job_id = (SELECT MIN(job_id) FROM jobs)"
        )


def test_refuses_points_without_client(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    doc = _doc_id(db)
    with pytest.raises(RemovalError, match="Qdrant"):
        remove_document(db, cfg, None, doc)
    # Nothing was removed.
    assert db.query_one("SELECT doc_id FROM documents WHERE doc_id = ?", (doc,)) is not None
    assert _count(db, "SELECT COUNT(*) AS n FROM chunks") >= 1


# --- 5. Open jobs are cancelled in the deletion transaction ---------------------


def test_cancels_open_jobs(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    doc = _doc_id(db)
    rev = db.query_one("SELECT rev_id FROM source_revisions")
    assert rev is not None
    db.execute(
        "INSERT INTO jobs (task_key, stage, input_id, state, attempts, max_attempts,"
        " created_at, updated_at) VALUES (?, 'extract', ?, 'pending', 0, 3, 0.0, 0.0)",
        ("remove-test-extract", str(rev["rev_id"])),
    )
    q = RealQdrantOps(cfg)
    try:
        report = remove_document(db, cfg, q, doc, execute=True)
    finally:
        q.close()
    assert report.jobs_cancelled == 1
    row = db.query_one("SELECT state FROM jobs WHERE task_key = 'remove-test-extract'")
    assert row is not None
    assert str(row["state"]) == "cancelled"


# --- 6. Catalog-only removal (no index points, no client) -----------------------


def test_catalog_only_removal_without_client(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    first = _doc_id(db)
    # A second book registered without running the pipeline: it has catalog
    # rows and an archive object, but no publications and no index points.
    src2 = cfg.paths.source_roots[0] / "second.pdf"
    make_pdf(src2, ["A second book about the harbor."])
    ingest = ingest_source(cfg.paths.archive_root, src2)
    register_source(
        db, normalize_path(str(src2)), ingest.sha256, ingest.size_bytes, Format.PDF
    )
    row = db.query_one(
        "SELECT p.doc_id, r.archive_relpath FROM path_aliases p"
        " JOIN source_revisions r ON r.rev_id = p.rev_id WHERE p.path = ?",
        (normalize_path(str(src2)),),
    )
    assert row is not None
    second = str(row["doc_id"])

    report = remove_document(db, cfg, None, second, execute=True)
    assert report.executed is True
    assert report.verified is True
    assert report.points == 0
    assert report.publications == 0
    assert report.revisions == 1
    assert not (cfg.paths.archive_root / str(row["archive_relpath"])).is_file()
    assert db.query_one(
        "SELECT doc_id FROM documents WHERE doc_id = ?", (second,)
    ) is None
    # The first book (with its points) is fully intact.
    assert db.query_one("SELECT doc_id FROM documents WHERE doc_id = ?", (first,)) is not None


# --- 7. Verification: a silent delete must be caught ----------------------------


def test_verification_fails_when_points_survive(
    state_db: Database, base_config: Config
) -> None:
    db, cfg = state_db, base_config
    cfg.embedding.fake = True
    cfg.services.qdrant_path = str(cfg.paths.qdrant_root)

    class NoopDeleteQdrant(FakeQdrant):
        def delete(self, f: IndexFilter) -> None:
            return  # a "successful" delete that deletes nothing

    q = NoopDeleteQdrant(cfg.embedding.dimensions)
    src = cfg.paths.source_roots[0] / "book.pdf"
    make_pdf(src, _PAGES)
    ingest_and_publish(db, cfg, src, qdrant=q, embedder=FakeEmbedder(cfg.embedding.dimensions))
    doc = _doc_id(db)

    with pytest.raises(RemovalError, match="verification"):
        remove_document(db, cfg, q, doc, execute=True)
    # The catalog was never touched (the point check runs first), and the
    # points are still there.
    assert db.query_one("SELECT doc_id FROM documents WHERE doc_id = ?", (doc,)) is not None
    assert _count(db, "SELECT COUNT(*) AS n FROM chunks") >= 1
    assert q.count(_doc_filter(doc)) > 0
