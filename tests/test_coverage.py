"""M7 coverage report (PRD §2 / §14).

The report must be honest about partial coverage: a fully-published library
shows a consistent funnel with an empty stalled list; a source file that was
never scanned shows up as *unindexed*; a deleted source shows up as
*orphaned*; a rewritten source shows up as *stale* (stat-only, no hashing);
a file with bad magic bytes shows up as *invalid*, not unindexed; a missing
archive object drops the archived count; an unavailable Qdrant degrades the
point counts to None without breaking the report. Real embedded (local)
Qdrant storage is used, as in the GC tests: no server, no Docker, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixtures import ingest_and_publish, make_pdf
from library_rag.archive import ingest_source
from library_rag.catalog import Format, register_source
from library_rag.config import Config
from library_rag.coverage import coverage_report
from library_rag.db import Database
from library_rag.embeddings import FakeEmbedder
from library_rag.identity import normalize_path
from library_rag.indexing import RealQdrantOps

_PAGES: list[str | None] = [
    "Chapter one introduces the lighthouse keeper.",
    "Chapter two follows the keeper's daughter to sea.",
    "Chapter three ends with the storm and the lamp.",
]


@pytest.fixture
def local_library(
    state_db: Database, base_config: Config
) -> tuple[Database, Config]:
    """One real book (3 pages) scanned and published into embedded Qdrant.

    The Qdrant client is closed before yielding: the report must be buildable
    while no worker holds the local storage lock, and each test opens its own
    client when it needs live point counts.
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


# --- 1. Fully-published library: consistent funnel, nothing to flag -----------


def test_published_library_is_clean(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    q = RealQdrantOps(cfg)
    try:
        rep = coverage_report(db, cfg, qdrant=q)
    finally:
        q.close()

    s = rep.stages
    assert s.documents == 1
    assert s.revisions == 1
    assert s.active_revisions == 1
    assert s.archived == 1
    assert s.archive_missing == []
    assert s.extracted == 1
    assert s.ocr_routed == 0
    assert s.ocr_done == 0
    assert s.chunked == 1
    assert s.embedded == 1
    assert s.indexed == 1
    assert s.indexed_points > 0
    assert s.published == 1
    assert s.staged == 0
    assert s.superseded == 0
    assert s.published_docs == 1
    # Catalog expectation, live active count, and live total all agree for a
    # single published book with no superseded publications.
    assert rep.points_expected == s.indexed_points
    assert rep.points_active == rep.points_expected
    assert rep.points_total == rep.points_active

    assert len(rep.roots) == 1
    rc = rep.roots[0]
    assert rc.mount_unavailable is False
    assert rc.discovered == 1
    assert rc.registered == 1
    assert rc.unindexed == []
    assert rc.orphaned == []
    assert rc.stale == []
    assert rc.invalid == []
    assert rep.stalled == []


# --- 2. Source on disk that was never scanned: unindexed -----------------------


def test_unindexed_source(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    src2 = cfg.paths.source_roots[0] / "loose.pdf"
    make_pdf(src2, ["A loose book that was never scanned."])

    rep = coverage_report(db, cfg)
    rc = rep.roots[0]
    assert rc.discovered == 2
    assert rc.registered == 1
    assert rc.unindexed == [normalize_path(str(src2))]
    assert rc.invalid == []
    assert rc.orphaned == []
    assert rc.stale == []


# --- 3. Registered book whose source file is gone: orphaned --------------------


def test_orphaned_alias(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    src = cfg.paths.source_roots[0] / "book.pdf"
    norm = normalize_path(str(src))
    src.unlink()  # temp-dir fixture source, not the operator's library

    rep = coverage_report(db, cfg)
    rc = rep.roots[0]
    assert rc.registered == 1  # the alias row still exists
    assert rc.orphaned == [norm]
    assert rc.stale == []  # a gone file is orphaned, not stale
    assert rc.unindexed == []
    assert rc.discovered == 0


# --- 4. Rewritten source: stale (stat-only, no hashing) ------------------------


def test_stale_source(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    src = cfg.paths.source_roots[0] / "book.pdf"
    norm = normalize_path(str(src))
    make_pdf(src, [*_PAGES, "A brand new chapter."])  # new bytes, size, mtime

    rep = coverage_report(db, cfg)
    rc = rep.roots[0]
    assert rc.stale == [norm]
    assert rc.orphaned == []
    assert rc.unindexed == []
    assert rc.registered == 1


# --- 5. Bad magic bytes: invalid, not unindexed ---------------------------------


def test_invalid_candidate(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    bad = cfg.paths.source_roots[0] / "notpdf.pdf"
    bad.write_bytes(b"this is not a real pdf")

    rep = coverage_report(db, cfg)
    rc = rep.roots[0]
    assert rc.invalid == [normalize_path(str(bad))]
    assert rc.unindexed == []
    assert rc.discovered == 2
    assert rc.registered == 1


# --- 6. Missing archive object: archived count drops ----------------------------


def test_missing_archive_object(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    rev = db.query_one("SELECT sha256, archive_relpath FROM source_revisions")
    assert rev is not None
    p = cfg.paths.archive_root / str(rev["archive_relpath"])
    assert p.is_file()
    p.unlink()

    rep = coverage_report(db, cfg)
    assert rep.stages.archived == 0
    assert rep.stages.archive_missing == [str(rev["archive_relpath"])]


# --- 7. Qdrant unavailable: point counts degrade to None ------------------------


def test_qdrant_unavailable_degrades(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    rep = coverage_report(db, cfg, qdrant=None)
    assert rep.points_active is None
    assert rep.points_total is None
    assert rep.points_expected > 0  # catalog side still reported
    assert rep.stages.published == 1
    assert rep.roots[0].registered == 1


# --- 8. Stalled documents: furthest stage without an active publication ---------


def test_stalled_docs(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library

    def _register(name: str, pages: list[str | None]) -> str:
        src = cfg.paths.source_roots[0] / name
        make_pdf(src, pages)
        ing = ingest_source(cfg.paths.archive_root, src)
        register_source(
            db, normalize_path(str(src)), ing.sha256, ing.size_bytes, Format.PDF
        )
        row = db.query_one(
            "SELECT r.doc_id FROM path_aliases a"
            " JOIN source_revisions r ON r.rev_id = a.rev_id WHERE a.path = ?",
            (normalize_path(str(src)),),
        )
        assert row is not None
        return str(row["doc_id"])

    doc_b = _register("book_b.pdf", ["Book B never left the shelf."])
    doc_c = _register("book_c.pdf", ["Book C was extracted but never chunked."])
    # Give book C a succeeded extraction run (no chunks) → stalled "extracted".
    row = db.query_one(
        "SELECT rev_id FROM source_revisions r JOIN documents d ON d.doc_id = r.doc_id"
        " WHERE d.doc_id = ?",
        (doc_c,),
    )
    assert row is not None
    db.execute(
        "INSERT INTO extraction_runs (run_id, rev_id, doc_id, parser_version,"
        " settings_sha, unit_count, state, created_at, updated_at)"
        " VALUES ('run-coverage-c', ?, ?, 'pdfium-1', 'settings-1', 1,"
        " 'succeeded', 0.0, 0.0)",
        (str(row["rev_id"]), doc_c),
    )

    published_doc = db.query_one(
        "SELECT doc_id FROM publications WHERE state = 'active'"
    )
    assert published_doc is not None

    rep = coverage_report(db, cfg)
    stages = {d.doc_id: d.stage for d in rep.stalled}
    assert stages == {doc_b: "registered", doc_c: "extracted"}
    assert str(published_doc["doc_id"]) not in stages  # the published book is not stalled
    # Furthest first: "extracted" ranks above "registered".
    assert [d.stage for d in rep.stalled] == ["extracted", "registered"]


# --- 9. JSON round trip -----------------------------------------------------------


def test_json_roundtrip(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    rep = coverage_report(db, cfg)
    payload = json.loads(json.dumps(rep.to_dict(), sort_keys=True))
    assert payload["stages"]["published"] == 1
    assert payload["stages"]["documents"] == 1
    assert payload["roots"][0]["registered"] == 1
    assert payload["points_active"] is None
    assert payload["stalled"] == []
    assert isinstance(payload["failures"]["jobs_by_state"], dict)


# --- 10. Mount unavailable: sentinel missing or root not a directory -------------


def test_mount_unavailable_root(
    local_library: tuple[Database, Config], tmp_path: Path
) -> None:
    db, cfg = local_library
    missing = tmp_path / "no-such-root"
    sent_root = tmp_path / "sentinel-root"
    sent_root.mkdir()
    cfg.mount_sentinels[str(sent_root)] = str(tmp_path / "absent-sentinel")
    cfg.paths.source_roots = [*cfg.paths.source_roots, missing, sent_root]

    rep = coverage_report(db, cfg)
    by_root = {rc.root: rc for rc in rep.roots}
    assert by_root[normalize_path(str(missing))].mount_unavailable is True
    assert by_root[normalize_path(str(sent_root))].mount_unavailable is True


# --- 11. Failures: jobs by state/stage, failed extractions ------------------------


def test_failure_counts(local_library: tuple[Database, Config]) -> None:
    db, cfg = local_library
    db.execute(
        "INSERT INTO jobs (task_key, stage, input_id, state, attempts, max_attempts,"
        " created_at, updated_at) VALUES ('cov-fail-ocr', 'ocr', NULL,"
        " 'retryable_failed', 3, 3, 0.0, 0.0)"
    )
    db.execute(
        "INSERT INTO jobs (task_key, stage, input_id, state, attempts, max_attempts,"
        " created_at, updated_at) VALUES ('cov-fail-embed', 'embed', NULL,"
        " 'permanent_failed', 3, 3, 0.0, 0.0)"
    )

    rep = coverage_report(db, cfg)
    f = rep.failures
    assert f.failed_jobs_by_stage == {"ocr": 1, "embed": 1}
    assert f.jobs_by_state.get("retryable_failed") == 1
    assert f.jobs_by_state.get("permanent_failed") == 1
    assert f.jobs_by_state.get("succeeded", 0) >= 1  # the fixture's finished jobs
    assert f.failed_extractions == 0
    assert f.ocr_failed_units == 0
