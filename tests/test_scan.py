"""Source discovery (PRD §8A): content-validated scan, scan_state fast path,
aliasing, change detection, and mount/availability handling.

The scan is a report: it never deletes catalog content, and files that change
during hashing are deferred to the next pass.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fixtures import make_pdf
from library_rag.archive import stream_hash
from library_rag.catalog import get_active_revision
from library_rag.config import Config, Paths
from library_rag.db import Database
from library_rag.identity import normalize_path
from library_rag.jobs import Jobs
from library_rag.scan import ScanReport, scan_roots


@pytest.fixture
def src(base_config: Config) -> Path:
    """The configured source root, created (it does not pre-exist)."""
    root = base_config.paths.source_roots[0]
    root.mkdir(parents=True)
    return root


def _scan(state_db: Database, cfg: Config, jobs: Jobs) -> ScanReport:
    reports = scan_roots(state_db, cfg, jobs)
    assert len(reports) == 1
    return reports[0]


def _one(db: Database, sql: str) -> dict[str, Any]:
    row = db.query_one(sql)
    assert row is not None
    return dict(row)


def test_scan_new_pdf_registers_and_enqueues(state_db: Database, base_config: Config, src: Path) -> None:
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)

    report = _scan(state_db, base_config, jobs)

    assert report.discovered == 1
    assert report.new_documents == 1
    assert report.jobs_enqueued == 1
    assert jobs.counts() == {"pending": 1}
    # Archived content-addressed and recorded in scan_state.
    row = state_db.query_one("SELECT sha256, size_bytes FROM source_revisions")
    assert row is not None
    archived = base_config.paths.archive_root / row["sha256"][:2] / row["sha256"]
    assert archived.is_file()
    assert state_db.query_one("SELECT path FROM scan_state") is not None


def test_second_scan_unchanged_no_rehash(
    state_db: Database, base_config: Config, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)
    assert _scan(state_db, base_config, jobs).new_documents == 1

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("stream_hash must not run for an unchanged file")

    monkeypatch.setattr("library_rag.scan.stream_hash", boom)
    report = _scan(state_db, base_config, jobs)

    assert report.unchanged == 1
    assert report.jobs_enqueued == 0
    assert jobs.counts() == {"pending": 1}  # no double enqueue


def test_rename_becomes_alias_no_job(state_db: Database, base_config: Config, src: Path) -> None:
    a = src / "A.pdf"
    make_pdf(a, ["a page of text " * 5])
    jobs = Jobs(state_db)
    assert _scan(state_db, base_config, jobs).new_documents == 1

    data = a.read_bytes()
    a.unlink()
    (src / "B.pdf").write_bytes(data)
    report = _scan(state_db, base_config, jobs)

    assert report.discovered == 1
    assert report.aliases == 1
    assert report.new_documents == 0
    assert report.new_revisions == 0
    assert report.jobs_enqueued == 0
    assert normalize_path(a) in report.missing  # the old alias is reported, not deleted
    # One document, one revision, two known paths.
    assert _one(state_db, "SELECT COUNT(*) AS n FROM documents")["n"] == 1
    assert _one(state_db, "SELECT COUNT(*) AS n FROM source_revisions")["n"] == 1
    assert _one(state_db, "SELECT COUNT(*) AS n FROM path_aliases")["n"] == 2


def test_content_change_new_revision_new_job(state_db: Database, base_config: Config, src: Path) -> None:
    a = src / "A.pdf"
    make_pdf(a, ["first edition page"])
    jobs = Jobs(state_db)
    assert _scan(state_db, base_config, jobs).new_documents == 1

    make_pdf(a, ["second edition page one", "second edition page two"])
    report = _scan(state_db, base_config, jobs)

    assert report.new_revisions == 1
    assert report.new_documents == 0
    assert report.jobs_enqueued == 1
    assert _one(state_db, "SELECT COUNT(*) AS n FROM documents")["n"] == 1
    assert _one(state_db, "SELECT COUNT(*) AS n FROM source_revisions")["n"] == 2
    doc = state_db.query_one("SELECT doc_id FROM documents")
    assert doc is not None
    active = get_active_revision(state_db, str(doc["doc_id"]))
    assert active is not None
    new_sha, _ = stream_hash(a)
    assert active["sha256"] == new_sha


def test_non_pdf_content_is_invalid(state_db: Database, base_config: Config, src: Path) -> None:
    (src / "bad.pdf").write_bytes(b"hello")
    jobs = Jobs(state_db)

    report = _scan(state_db, base_config, jobs)

    assert normalize_path(src / "bad.pdf") in report.invalid
    assert _one(state_db, "SELECT COUNT(*) AS n FROM documents")["n"] == 0
    assert jobs.counts() == {}


def test_deleted_file_reported_missing(state_db: Database, base_config: Config, src: Path) -> None:
    a = src / "A.pdf"
    make_pdf(a, ["a page of text " * 5])
    jobs = Jobs(state_db)
    assert _scan(state_db, base_config, jobs).new_documents == 1

    a.unlink()
    report = _scan(state_db, base_config, jobs)

    assert normalize_path(a) in report.missing
    # Catalog content is intact: the file is reported, never deleted.
    assert _one(state_db, "SELECT COUNT(*) AS n FROM path_aliases")["n"] == 1


def test_ignores_symlinks_and_ignored_dirs(state_db: Database, base_config: Config, src: Path) -> None:
    (src / "__pycache__").mkdir()
    make_pdf(src / "__pycache__" / "junk.pdf", ["hidden"])
    real = make_pdf(src / "C.pdf", ["visible page " * 5])
    (src / "link.pdf").symlink_to(real)
    (src / "notes.txt").write_bytes(b"not a candidate")
    jobs = Jobs(state_db)

    report = _scan(state_db, base_config, jobs)

    assert report.discovered == 1
    assert report.new_documents == 1
    assert _one(state_db, "SELECT COUNT(*) AS n FROM documents")["n"] == 1


def test_mount_unavailable_sentinel_and_missing_root(
    state_db: Database, base_config: Config, src: Path
) -> None:
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)

    cfg2 = Config(
        paths=base_config.paths,
        services=base_config.services,
        mount_sentinels={str(src): "/nonexistent/sentinel"},
    )
    reports = scan_roots(state_db, cfg2, jobs)
    assert reports[0].mount_unavailable is True
    assert reports[0].discovered == 0

    ghost = base_config.paths.state_root.parent / "ghost"  # does not exist
    paths = Paths(
        source_roots=[ghost],
        archive_root=base_config.paths.archive_root,
        artifact_root=base_config.paths.artifact_root,
        state_root=base_config.paths.state_root,
        qdrant_root=base_config.paths.qdrant_root,
        model_root=base_config.paths.model_root,
        scratch_root=base_config.paths.scratch_root,
    )
    cfg3 = Config(paths=paths, services=base_config.services)
    assert scan_roots(state_db, cfg3, jobs)[0].mount_unavailable is True


def test_changed_during_scan_deferred(
    state_db: Database, base_config: Config, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = src / "A.pdf"
    make_pdf(a, ["a page of text " * 5])
    real_hash = stream_hash

    def mutating_hash(path: Path, chunk: int = 1 << 20) -> tuple[str, int]:
        # The file changes between the pre- and post-hash stat: defer this pass.
        os.utime(path, (1_000_000_000, 1_000_000_000))
        return real_hash(path, chunk)

    monkeypatch.setattr("library_rag.scan.stream_hash", mutating_hash)
    jobs = Jobs(state_db)
    report = _scan(state_db, base_config, jobs)

    assert normalize_path(a) in report.changed_during_scan
    assert _one(state_db, "SELECT COUNT(*) AS n FROM documents")["n"] == 0
    assert jobs.counts() == {}
    # The next stable scan picks it up normally.
    monkeypatch.setattr("library_rag.scan.stream_hash", real_hash)
    report2 = _scan(state_db, base_config, jobs)
    assert report2.new_documents == 1
    assert jobs.counts() == {"pending": 1}
