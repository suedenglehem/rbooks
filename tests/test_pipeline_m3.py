"""M3 full-pipeline end-to-end (PRD §7/§8C/§8E): the extract -> ocr -> chunk
stages over a real mixed PDF (fake Tesseract shim, PRD §1) and a real EPUB,
the "originals unchanged" gate invariant, scratch cleanliness, and the
worker-start reconciliation that rebuilds vanished chunk rows with their
deterministic ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixtures import (
    ctx_for_rev,
    ingest_and_register,
    make_epub,
    make_fake_tesseract,
    make_mixed_pdf,
    make_pdf,
)
from library_rag.archive import archive_path_for
from library_rag.catalog import Format
from library_rag.config import Config
from library_rag.db import Database
from library_rag.extraction import extract_pdf, load_unit_artifact
from library_rag.identity import unit_id_for
from library_rag.jobs import Jobs
from library_rag.scan import scan_roots
from library_rag.worker import chunk_fingerprint_for_run, run_worker

_SENTENCE = "The quick brown fox jumps over the lazy dog."
# The fake shim emits "w{page}-{0,1,2}" per page, one word per line.
_OCR_TEXT = "w{p}-0\nw{p}-1\nw{p}-2"
# Union of the shim's three word boxes in page points (see test_ocr.py).
_WORD_UNION = [12.0, 24.0, 165.6, 36.0]


@pytest.fixture
def src(base_config: Config) -> Path:
    root = base_config.paths.source_roots[0]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _chunk_ids(db: Database, run_id: str) -> list[str]:
    return [
        r["chunk_id"]
        for r in db.query("SELECT chunk_id FROM chunks WHERE run_id = ? ORDER BY position", (run_id,))
    ]


def _drain_text_pdf(
    state_db: Database, base_config: Config, src: Path, tmp_path: Path, name: str
) -> str:
    """Two text-page PDF through the full pipeline; return the run id."""
    base_config.extraction.ocr.bin = str(make_fake_tesseract(tmp_path / "shim" / "tesseract"))
    make_pdf(src / name, [_SENTENCE, "Second page of the test book."])
    # The scan does ingest + register + extract-job enqueue in one pass
    # (pre-registering here would make the scan see an alias and enqueue
    # nothing).
    scan_roots(state_db, base_config, Jobs(state_db))
    jobs = Jobs(state_db)
    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 2  # extract + chunk
    assert jobs.counts() == {"succeeded": 2}
    row = state_db.query_one("SELECT run_id FROM extraction_runs")
    assert row is not None
    return str(row["run_id"])


def test_mixed_pdf_pipeline(state_db: Database, base_config: Config, src: Path, tmp_path: Path) -> None:
    path = make_mixed_pdf(src / "m.pdf", ["text", "scanned", "blank", "sparse", "sparse_image"])
    original = path.read_bytes()
    base_config.extraction.ocr.bin = str(make_fake_tesseract(tmp_path / "shim" / "tesseract"))
    scan_roots(state_db, base_config, Jobs(state_db))

    rev = state_db.query_one("SELECT rev_id, sha256 FROM source_revisions")
    assert rev is not None
    rev_id = str(rev["rev_id"])
    sha256 = str(rev["sha256"])

    jobs = Jobs(state_db)
    # extract + 2 OCR jobs (pages 1 and 4) + chunk.
    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 4
    assert jobs.counts() == {"succeeded": 4}

    run = state_db.query_one("SELECT * FROM extraction_runs")
    assert run is not None
    assert run["state"] == "succeeded"
    run_id = str(run["run_id"])

    units = state_db.query(
        "SELECT route, ocr_state, artifact_sha256 FROM source_units "
        "WHERE run_id = ? ORDER BY position",
        (run_id,),
    )
    assert [(r["route"], r["ocr_state"]) for r in units] == [
        ("reuse", "none"),  # clean text page
        ("ocr", "done"),  # scanned page
        ("skip", "none"),  # blank page
        ("skip", "none"),  # sparse text, no image
        ("ocr", "done"),  # sparse text over a full-page image
    ]

    # One chunk over the 16 surviving tokens (9 + 3 + 0 + 1 + 3); the blank
    # page contributes no tokens and therefore no span.
    chunks = state_db.query("SELECT * FROM chunks WHERE run_id = ? ORDER BY position", (run_id,))
    assert len(chunks) == 1
    assert chunks[0]["text"] == f"{_SENTENCE} w1-0 w1-1 w1-2 End w4-0 w4-1 w4-2"
    assert chunks[0]["token_count"] == 16
    assert chunks[0]["title"] is None
    uids = [unit_id_for(run_id, "page", i) for i in range(5)]
    spans = json.loads(chunks[0]["spans"])
    assert [(s["unit_id"], s["source_start"], s["source_end"]) for s in spans] == [
        (uids[0], 0, len(_SENTENCE)),
        (uids[1], 0, 14),
        (uids[3], 0, 3),
        (uids[4], 0, 14),
    ]
    # Native units carry no bbox; OCR units carry the union of their word boxes.
    assert [s["bbox"] for s in spans] == [None, _WORD_UNION, None, _WORD_UNION]

    # OCR output was committed into the scanned page's artifact...
    p_ocr = load_unit_artifact(
        base_config.paths.artifact_root, rev_id, uids[1], units[1]["artifact_sha256"]
    )
    assert p_ocr["ocr"]["text"] == _OCR_TEXT.format(p=1)
    # ...while the native text page kept its layer untouched (no ocr key).
    # PyMuPDF's get_text ends each text line with a newline; the artifact
    # stores the raw layer verbatim.
    p_text = load_unit_artifact(
        base_config.paths.artifact_root, rev_id, uids[0], units[0]["artifact_sha256"]
    )
    assert p_text["text"] == _SENTENCE + "\n"
    assert "ocr" not in p_text

    # Gate: the source original and its content-addressed archive copy are
    # bit-for-bit unchanged by the pipeline.
    assert path.read_bytes() == original
    archived = archive_path_for(base_config.paths.archive_root, sha256)
    assert archived.read_bytes() == original

    # Scratch is clean: no rendered page PNGs left behind.
    assert list((base_config.paths.scratch_root / "ocr" / run_id).rglob("*.png")) == []


def test_epub_pipeline(state_db: Database, base_config: Config, src: Path, tmp_path: Path) -> None:
    make_epub(
        src / "b.epub",
        [
            ("Chapter One", ["First paragraph here.", "Second paragraph here."]),
            ("Chapter Two", ["Third paragraph here."]),
        ],
    )
    base_config.extraction.ocr.bin = str(make_fake_tesseract(tmp_path / "shim" / "tesseract"))
    scan_roots(state_db, base_config, Jobs(state_db))

    jobs = Jobs(state_db)
    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 2  # extract + chunk
    assert jobs.counts() == {"succeeded": 2}
    stages = {
        r["stage"]: int(r["n"])
        for r in state_db.query("SELECT stage, COUNT(*) AS n FROM jobs GROUP BY stage")
    }
    assert stages == {"extract": 1, "chunk": 1}  # no OCR stage for EPUB

    row = state_db.query_one("SELECT run_id FROM extraction_runs")
    assert row is not None
    run_id = str(row["run_id"])
    chunks = state_db.query("SELECT * FROM chunks WHERE run_id = ? ORDER BY position", (run_id,))
    assert len(chunks) == 1
    # Headings are block-level paragraphs (M2-pinned: they count for
    # anchoring), so the h1 texts are in the body; the title is only the
    # *extra* prefix stored in its own column.
    assert (
        chunks[0]["text"]
        == "Chapter One First paragraph here. Second paragraph here. "
        "Chapter Two Third paragraph here."
    )
    assert chunks[0]["title"] == "Chapter One"  # first section's title prefixes chunk 0
    assert chunks[0]["token_count"] == 15  # 13 body + 2 title tokens
    spans = json.loads(chunks[0]["spans"])
    assert len(spans) == 2  # one span per section
    assert all(s["bbox"] is None for s in spans)  # native text: no OCR boxes


def test_reconcile_requeues_chunk_for_jobless_run(
    state_db: Database, base_config: Config, src: Path, tmp_path: Path
) -> None:
    base_config.extraction.ocr.bin = str(make_fake_tesseract(tmp_path / "shim" / "tesseract"))
    path = make_pdf(src / "r.pdf", [_SENTENCE, "Second page of the test book."])
    rev_id = ingest_and_register(state_db, base_config, path, Format.PDF)
    # M2-style manual extract: no Jobs wired, so no OCR/chunk jobs are enqueued.
    ctx = ctx_for_rev(state_db, base_config, rev_id)
    assert extract_pdf(ctx) == 2

    row = state_db.query_one("SELECT COUNT(*) AS n FROM jobs")
    assert row is not None
    assert int(row["n"]) == 0
    run = state_db.query_one("SELECT * FROM extraction_runs")
    assert run is not None
    assert run["state"] == "succeeded"
    assert run["chunk_fingerprint"] is None

    # Worker-start reconciliation enqueues the missing chunk job.
    jobs = Jobs(state_db)
    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1
    assert jobs.counts() == {"succeeded": 1}
    job = state_db.query_one("SELECT * FROM jobs")
    assert job is not None
    assert job["stage"] == "chunk"
    run_id = str(run["run_id"])
    assert len(_chunk_ids(state_db, run_id)) == 1
    after = state_db.query_one("SELECT chunk_fingerprint FROM extraction_runs WHERE run_id = ?", (run_id,))
    assert after is not None
    assert after["chunk_fingerprint"] == chunk_fingerprint_for_run(state_db, run_id, base_config)


def test_rebuild_after_lost_chunk_job_and_rows(
    state_db: Database, base_config: Config, src: Path, tmp_path: Path
) -> None:
    run_id = _drain_text_pdf(state_db, base_config, src, tmp_path, "b.pdf")
    before = _chunk_ids(state_db, run_id)
    assert len(before) == 1

    # Simulate a lost chunk job row, wiped chunk rows, and a cleared fingerprint.
    state_db.execute("DELETE FROM chunks WHERE run_id = ?", (run_id,))
    state_db.execute("DELETE FROM jobs WHERE stage = 'chunk' AND input_id = ?", (run_id,))
    state_db.execute(
        "UPDATE extraction_runs SET chunk_fingerprint = NULL WHERE run_id = ?", (run_id,)
    )

    # Reconciliation re-enqueues (stored fp NULL != current) and rebuilds with
    # the deterministic ids.
    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1
    assert _chunk_ids(state_db, run_id) == before
    assert Jobs(state_db).counts() == {"succeeded": 2}  # extract + the new chunk job


def test_rebuild_when_chunks_vanished_but_fingerprint_current(
    state_db: Database, base_config: Config, src: Path, tmp_path: Path
) -> None:
    run_id = _drain_text_pdf(state_db, base_config, src, tmp_path, "b.pdf")
    before = _chunk_ids(state_db, run_id)

    # Chunk rows vanished but the fingerprint and the chunk job row are intact:
    # re-arming the job exercises the no-op check's rebuild fall-through.
    state_db.execute("DELETE FROM chunks WHERE run_id = ?", (run_id,))
    state_db.execute(
        "UPDATE jobs SET state = 'pending', attempts = 0, lease_token = NULL, "
        "lease_expires_at = NULL, error_category = NULL, error_detail = NULL, "
        "next_attempt_at = NULL WHERE stage = 'chunk' AND input_id = ?",
        (run_id,),
    )

    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1
    assert _chunk_ids(state_db, run_id) == before  # deterministic rebuild


def test_chunk_noop_when_fingerprint_current(
    state_db: Database, base_config: Config, src: Path, tmp_path: Path
) -> None:
    _drain_text_pdf(state_db, base_config, src, tmp_path, "b.pdf")
    # Fingerprint matches and chunks exist: reconciliation adds nothing and
    # the queue is empty.
    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 0
