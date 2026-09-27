"""M10 manual-selection ledger: recording submissions, the live status join,
and the /manual-selections route behind the Rag page's "Manual selection"
panel.

The ledger is the *final* set of operator-submitted books (one row per
normalized path, resubmission refreshes it); :func:`manual_selection_status`
joins live pipeline state onto each row so the UI can show where every
submitted book stands right now.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from library_rag.api import create_app
from library_rag.catalog import Format, RegistrationStatus, register_source
from library_rag.config import BrowseSettings, Config
from library_rag.db import Database
from library_rag.identity import make_task_key, normalize_path
from library_rag.indexing import FakeQdrant
from library_rag.jobs import BOOK_BUDGET_PARKED, Jobs
from library_rag.manual_selection import manual_selection_status, record_submission
from library_rag.scan import ProcessStatus

H1 = "ab" * 32
H2 = "cd" * 32
NOW = 1_700_000_000.0


def _client(cfg: Config, db: Database) -> TestClient:
    return TestClient(create_app(cfg, db, qdrant=FakeQdrant()))


def _register(path: str, sha: str, db: Database, size: int = 1024) -> tuple[str, str]:
    """Register *path* holding *sha*; return (doc_id, rev_id)."""
    reg = register_source(db, path, sha, size, Format.PDF)
    return reg.doc_id, reg.rev_id


def _scan_state_row(db: Database, path: str, sha: str, rev_id: str, size: int = 1024) -> None:
    """The fast-check row the scan writes for every registered file —
    :func:`record_submission` resolves its links from here."""
    db.execute(
        """
        INSERT INTO scan_state (path, size_bytes, mtime, sha256, format, rev_id, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (normalize_path(path), size, NOW, sha, "pdf", rev_id, NOW),
    )


def _submit(db: Database, path: str, status: ProcessStatus, now: float) -> None:
    record_submission(db, {normalize_path(path): status}, now=now)


def _status(db: Database, path: str) -> dict[str, Any]:
    rows = [dict(r) for r in manual_selection_status(db) if r["path"] == normalize_path(path)]
    assert len(rows) == 1
    return rows[0]


def _set_job_state(
    db: Database, task_key: str, state: str, error_category: str | None = None, error_detail: str | None = None
) -> None:
    db.execute(
        "UPDATE jobs SET state = ?, error_category = ?, error_detail = ? WHERE task_key = ?",
        (state, error_category, error_detail, task_key),
    )


def _run_row(
    db: Database, doc_id: str, rev_id: str, run_id: str, *, state: str = "running", created_at: float = 1.0
) -> None:
    # parser_version is derived from the run so two runs of one revision can
    # coexist (extraction_runs is UNIQUE on rev_id, parser_version, settings_sha).
    db.execute(
        """
        INSERT INTO extraction_runs
            (run_id, rev_id, doc_id, parser_version, settings_sha, unit_count, state, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
        """,
        (run_id, rev_id, doc_id, f"p-{run_id}", f"s-{run_id}", state, created_at, created_at),
    )


def _published_chain(db: Database, doc_id: str, rev_id: str, run_id: str) -> None:
    """A succeeded run plus its generation and an active publication, in FK
    order (publications.gen_id -> index_generations -> extraction_runs)."""
    _run_row(db, doc_id, rev_id, run_id, state="succeeded")
    db.execute(
        """
        INSERT INTO index_generations
            (gen_id, run_id, rev_id, model_revision, embedding_sha, sparse_stats_sha,
             dimensions, dtype, normalized, point_count, state, created_at, updated_at)
        VALUES (?, ?, ?, 'm1', 'e1', 'ss1', 8, 'float32', 1, 1, 'ready', 1.0, 1.0)
        """,
        (f"gen-{run_id}", run_id, rev_id),
    )
    db.execute(
        """
        INSERT INTO publications
            (pub_id, rev_id, doc_id, gen_id, run_id, expected_points, state, created_at, activated_at)
        VALUES (?, ?, ?, ?, ?, 1, 'active', 1.0, 1.0)
        """,
        (f"pub-{run_id}", rev_id, doc_id, f"gen-{run_id}", run_id),
    )


# --- recording ----------------------------------------------------------------


def test_record_submission_links_from_scan_state(state_db: Database) -> None:
    doc, rev = _register("/mnt/b/X.pdf", H1, state_db)
    _scan_state_row(state_db, "/mnt/b/X.pdf", H1, rev)

    _submit(state_db, "/mnt/b/X.pdf", ProcessStatus.NEW_DOCUMENT, now=1000.0)

    row = _status(state_db, "/mnt/b/X.pdf")
    assert row["outcome"] == "new_document"
    assert row["rev_id"] == rev
    assert row["doc_id"] == doc
    # No pipeline content yet: the join has nothing to say.
    assert row["status"] == "unknown"
    # The ledger row itself carries the content hash for the join.
    ledger = state_db.query_one("SELECT sha256 FROM manual_selections")
    assert ledger is not None
    assert ledger["sha256"] == H1


def test_record_submission_missing_file_has_no_links(state_db: Database) -> None:
    _submit(state_db, "/mnt/b/gone.pdf", ProcessStatus.MISSING, now=1000.0)

    row = _status(state_db, "/mnt/b/gone.pdf")
    assert row["outcome"] == "missing"
    assert row["rev_id"] is None
    assert row["doc_id"] is None
    assert row["status"] == "unknown"
    assert row["title"] == "gone"  # path stem is the fallback title


def test_resubmission_refreshes_row_keeps_links(state_db: Database) -> None:
    doc, rev = _register("/mnt/b/X.pdf", H1, state_db)
    _scan_state_row(state_db, "/mnt/b/X.pdf", H1, rev)
    _submit(state_db, "/mnt/b/X.pdf", ProcessStatus.NEW_DOCUMENT, now=1000.0)

    # The file is already registered: a resubmission reports "unchanged" and
    # the ledger refreshes the one row — latest outcome wins, no duplicate.
    _submit(state_db, "/mnt/b/X.pdf", ProcessStatus.UNCHANGED, now=2000.0)

    assert len(state_db.query("SELECT path FROM manual_selections")) == 1
    row = _status(state_db, "/mnt/b/X.pdf")
    assert row["outcome"] == "unchanged"
    assert row["submitted_at"] == 2000.0
    assert row["rev_id"] == rev  # COALESCE keeps the link
    assert row["doc_id"] == doc


def test_resubmission_of_vanished_file_keeps_links(state_db: Database) -> None:
    doc, rev = _register("/mnt/b/X.pdf", H1, state_db)
    _scan_state_row(state_db, "/mnt/b/X.pdf", H1, rev)
    _submit(state_db, "/mnt/b/X.pdf", ProcessStatus.NEW_DOCUMENT, now=1000.0)
    # The file disappeared and its fast-check row is gone: the resubmission
    # carries no links, but the ledger must keep pointing at its book.
    state_db.execute("DELETE FROM scan_state")
    _submit(state_db, "/mnt/b/X.pdf", ProcessStatus.MISSING, now=2000.0)

    row = _status(state_db, "/mnt/b/X.pdf")
    assert row["outcome"] == "missing"
    assert row["rev_id"] == rev
    assert row["doc_id"] == doc


# --- live status join -----------------------------------------------------------


def test_status_follows_the_pipeline(state_db: Database) -> None:
    doc, rev = _register("/mnt/b/X.pdf", H1, state_db)
    _scan_state_row(state_db, "/mnt/b/X.pdf", H1, rev)
    _submit(state_db, "/mnt/b/X.pdf", ProcessStatus.NEW_DOCUMENT, now=1000.0)
    extract_key = make_task_key("extract", rev, H1)

    Jobs(state_db).enqueue(extract_key, "extract", input_id=rev, input_version=H1)
    assert _status(state_db, "/mnt/b/X.pdf")["status"] == "queued"

    _set_job_state(state_db, extract_key, "running")
    row = _status(state_db, "/mnt/b/X.pdf")
    assert (row["status"], row["stage"]) == ("processing", "extract")

    _set_job_state(state_db, extract_key, "succeeded")
    # Extract done, nothing published yet: still on its way.
    row = _status(state_db, "/mnt/b/X.pdf")
    assert (row["status"], row["stage"]) == ("processing", None)

    _published_chain(state_db, doc, rev, "run-1")
    row = _status(state_db, "/mnt/b/X.pdf")
    assert row["status"] == "published"
    assert row["title"] == "X"
    assert row["format"] == "pdf"
    assert row["size_bytes"] == 1024

    _set_job_state(state_db, extract_key, "retryable_failed", "budget", BOOK_BUDGET_PARKED)
    row = _status(state_db, "/mnt/b/X.pdf")
    assert (row["status"], row["stage"]) == ("parked", "extract")

    _set_job_state(state_db, extract_key, "permanent_failed", "boom", "extract exploded")
    row = _status(state_db, "/mnt/b/X.pdf")
    assert (row["status"], row["stage"], row["error"]) == ("failed", "extract", "extract exploded")


def test_run_keyed_stages_use_the_run(state_db: Database) -> None:
    doc, rev = _register("/mnt/b/X.pdf", H1, state_db)
    _scan_state_row(state_db, "/mnt/b/X.pdf", H1, rev)
    _submit(state_db, "/mnt/b/X.pdf", ProcessStatus.NEW_DOCUMENT, now=1000.0)
    jobs = Jobs(state_db)
    extract_key = make_task_key("extract", rev, H1)
    jobs.enqueue(extract_key, "extract", input_id=rev, input_version=H1)
    _set_job_state(state_db, extract_key, "succeeded")
    _run_row(state_db, doc, rev, "run-1", state="running")

    chunk_key = make_task_key("chunk", "run-1", H1)
    jobs.enqueue(chunk_key, "chunk", input_id="run-1", input_version=H1)
    assert _status(state_db, "/mnt/b/X.pdf")["status"] == "queued"

    _set_job_state(state_db, chunk_key, "running")
    row = _status(state_db, "/mnt/b/X.pdf")
    assert (row["status"], row["stage"]) == ("processing", "chunk")

    _set_job_state(state_db, chunk_key, "permanent_failed", "bad", "chunk failed")
    row = _status(state_db, "/mnt/b/X.pdf")
    assert (row["status"], row["stage"], row["error"]) == ("failed", "chunk", "chunk failed")


def test_latest_run_wins_over_older_failed_jobs(state_db: Database) -> None:
    """A newer run supersedes an older run's failures: the book is on its way
    again, not failed (the status join keys run-staged jobs on the latest
    run per revision)."""
    doc, rev = _register("/mnt/b/X.pdf", H1, state_db)
    _scan_state_row(state_db, "/mnt/b/X.pdf", H1, rev)
    _submit(state_db, "/mnt/b/X.pdf", ProcessStatus.NEW_DOCUMENT, now=1000.0)
    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("extract", rev, H1), "extract", input_id=rev, input_version=H1)
    _set_job_state(state_db, make_task_key("extract", rev, H1), "succeeded")

    _run_row(state_db, doc, rev, "run-1", state="failed", created_at=1.0)
    k1 = make_task_key("chunk", "run-1", H1)
    jobs.enqueue(k1, "chunk", input_id="run-1", input_version=H1)
    _set_job_state(state_db, k1, "permanent_failed", "bad", "run-1 chunk failed")

    _run_row(state_db, doc, rev, "run-2", state="running", created_at=2.0)
    row = _status(state_db, "/mnt/b/X.pdf")
    assert (row["status"], row["stage"]) == ("processing", None)


def test_superseded_when_a_newer_revision_takes_over(state_db: Database) -> None:
    doc, rev_a = _register("/mnt/b/X.pdf", H1, state_db)
    _scan_state_row(state_db, "/mnt/b/X.pdf", H1, rev_a)
    _submit(state_db, "/mnt/b/X.pdf", ProcessStatus.NEW_DOCUMENT, now=1000.0)
    extract_key = make_task_key("extract", rev_a, H1)
    Jobs(state_db).enqueue(extract_key, "extract", input_id=rev_a, input_version=H1)
    _set_job_state(state_db, extract_key, "succeeded")

    # A second edition lands on the same path: rev B becomes active and rev A
    # is no longer the book's current revision — but the operator's ledger row
    # still points at the old edition.
    reg = register_source(state_db, "/mnt/b/X.pdf", H2, 2048, Format.PDF)
    assert reg.status is RegistrationStatus.NEW_REVISION
    assert reg.doc_id == doc
    state_db.execute(
        "UPDATE manual_selections SET rev_id = ? WHERE path = ?",
        (rev_a, normalize_path("/mnt/b/X.pdf")),
    )

    assert _status(state_db, "/mnt/b/X.pdf")["status"] == "superseded"


# --- HTTP routes ----------------------------------------------------------------


def test_manual_selections_route_empty_without_browse(
    state_db: Database, base_config: Config
) -> None:
    """The route serves the ledger regardless of Browse availability: a
    pulled mount must not hide books that were already selected."""
    res = _client(base_config, state_db).get("/manual-selections")
    assert res.status_code == 200
    assert res.json() == {"selections": []}


def test_browse_process_records_the_ledger(
    state_db: Database, base_config: Config, tmp_path: Path
) -> None:
    root = tmp_path / "browsebooks"
    root.mkdir()
    (root / "X.pdf").write_bytes(b"%PDF-1.4 x")
    base_config.browse = BrowseSettings(enabled=True, root=root)
    client = _client(base_config, state_db)

    res = client.post("/browse/process", json={"paths": ["X.pdf"]})
    assert res.status_code == 200
    assert res.json()["results"]["X.pdf"] == "new_document"

    res = client.get("/manual-selections")
    assert res.status_code == 200
    (sel,) = res.json()["selections"]
    assert sel["path"] == normalize_path(root / "X.pdf")
    assert sel["title"] == "X"
    assert sel["outcome"] == "new_document"
    assert sel["rev_id"] is not None
    assert sel["status"] == "queued"  # the extract job is pending


def test_browse_process_resubmit_refreshes_ledger(
    state_db: Database, base_config: Config, tmp_path: Path
) -> None:
    root = tmp_path / "browsebooks"
    root.mkdir()
    (root / "X.pdf").write_bytes(b"%PDF-1.4 x")
    base_config.browse = BrowseSettings(enabled=True, root=root)
    client = _client(base_config, state_db)

    assert client.post("/browse/process", json={"paths": ["X.pdf"]}).status_code == 200
    assert client.post("/browse/process", json={"paths": ["X.pdf"]}).status_code == 200

    (sel,) = client.get("/manual-selections").json()["selections"]
    assert sel["outcome"] == "unchanged"  # latest outcome wins
    assert len(state_db.query("SELECT path FROM manual_selections")) == 1
