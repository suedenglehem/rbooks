"""Selective OCR routing (PRD §8C): the page-route decision matrix, the
image-area measurement over real page shapes, the extraction quality flags,
and the end-to-end route/ocr_state assignment for a mixed PDF.

The end-to-end case drives the real worker with the fake Tesseract shim
(PRD §1: no system installs on the test host).
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from fixtures import make_fake_tesseract, make_mixed_pdf
from library_rag.config import Config, OcrSettings
from library_rag.db import Database
from library_rag.extraction.pdf import assess_page, extract_pdf
from library_rag.extraction.routing import (
    ROUTE_OCR,
    ROUTE_REUSE,
    ROUTE_SKIP,
    image_area_ratio,
    route_page,
)
from library_rag.jobs import Jobs
from library_rag.scan import scan_roots
from library_rag.worker import build_ctx, run_worker

_S = OcrSettings()  # default image_area_threshold = 0.5

# --- decision matrix ----------------------------------------------------------
def test_route_decision_matrix() -> None:
    # A broken text layer is always worth re-reading, even with no image.
    assert route_page(["many_replacement"], 0.0, _S) == ROUTE_OCR
    assert route_page(["sparse", "many_replacement"], 0.0, _S) == ROUTE_OCR
    # No/sparse text: OCR only when an embedded image plausibly is the page.
    assert route_page(["no_text"], 0.5, _S) == ROUTE_OCR
    assert route_page(["sparse"], 0.9, _S) == ROUTE_OCR
    assert route_page(["no_text"], 0.49, _S) == ROUTE_SKIP
    assert route_page(["sparse"], 0.0, _S) == ROUTE_SKIP
    # A clean text layer is reused no matter the image coverage (figures with
    # captions, infographics): the native layer is what gets indexed.
    assert route_page([], 1.0, _S) == ROUTE_REUSE
    assert route_page([], 0.0, _S) == ROUTE_REUSE


def test_route_threshold_is_configurable() -> None:
    assert route_page(["sparse"], 0.6, OcrSettings(image_area_threshold=0.9)) == ROUTE_SKIP
    assert route_page(["sparse"], 0.9, OcrSettings(image_area_threshold=0.9)) == ROUTE_OCR


# --- image-area measurement ----------------------------------------------------
def test_image_area_ratio_per_page_shape(tmp_path: Path) -> None:
    kinds = ["text", "scanned", "blank", "sparse", "sparse_image"]
    path = make_mixed_pdf(tmp_path / "m.pdf", kinds)
    doc = pymupdf.open(str(path))  # type: ignore[no-untyped-call]
    try:
        ratios = [image_area_ratio(doc[i]) for i in range(len(kinds))]
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    assert ratios[0] == 0.0  # text page: no image
    assert ratios[1] == pytest.approx(1.0, abs=0.01)  # scanned: full-page image
    assert ratios[2] == 0.0  # blank
    assert ratios[3] == 0.0  # sparse: text only
    assert ratios[4] == pytest.approx(1.0, abs=0.01)  # sparse + full-page image


# --- quality flags --------------------------------------------------------------
def test_assess_page_flags() -> None:
    assert assess_page("The quick brown fox jumps over the lazy dog.", 40, 0.05) == []
    assert assess_page("", 40, 0.05) == ["no_text"]
    assert assess_page("   \n  ", 40, 0.05) == ["no_text"]
    assert assess_page("End", 40, 0.05) == ["sparse"]
    # 5 non-whitespace chars < 40, and 4/5 replacement chars > 0.05.
    assert assess_page("a����", 40, 0.05) == ["sparse", "many_replacement"]
    # One replacement in a long clean page stays below the threshold.
    assert assess_page("word " * 10 + "�", 40, 0.05) == []


# --- end-to-end: routes and ocr_state over a real extract -----------------------
@pytest.fixture
def src(base_config: Config) -> Path:
    root = base_config.paths.source_roots[0]
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_extract_assigns_routes_and_ocr_states(
    state_db: Database,
    base_config: Config,
    src: Path,
    tmp_path: Path,
) -> None:
    base_config.extraction.ocr.bin = str(make_fake_tesseract(tmp_path / "shim" / "tesseract"))

    # The scan does ingest + register + extract-job enqueue in one pass.
    make_mixed_pdf(src / "m.pdf", ["text", "scanned", "blank", "sparse", "sparse_image"])
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)

    # extract + OCR of the 2 scanned pages + chunk.
    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 4
    assert jobs.counts() == {"succeeded": 4}

    rows = state_db.query("SELECT route, ocr_state FROM source_units ORDER BY position")
    assert [(r["route"], r["ocr_state"]) for r in rows] == [
        ("reuse", "none"),  # clean text page
        ("ocr", "done"),  # scanned page
        ("skip", "none"),  # blank page
        ("skip", "none"),  # sparse text, no image
        ("ocr", "done"),  # sparse text over a full-page image
    ]


def test_reextract_is_a_job_noop(
    state_db: Database,
    base_config: Config,
    src: Path,
    tmp_path: Path,
) -> None:
    base_config.extraction.ocr.bin = str(make_fake_tesseract(tmp_path / "shim" / "tesseract"))

    make_mixed_pdf(src / "m.pdf", ["text", "scanned", "blank", "sparse", "sparse_image"])
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)
    run_worker(state_db, base_config, once=True, poll_delay=0)
    row = state_db.query_one("SELECT COUNT(*) AS n FROM jobs")
    assert row is not None
    n_jobs = int(row["n"])
    assert n_jobs == 4

    # Re-run extraction on the same (source, parser, settings) key: the run
    # already succeeded, so no units are re-inserted and the idempotent
    # re-enqueues add no jobs.
    rev = state_db.query_one("SELECT * FROM source_revisions")
    assert rev is not None
    ctx = build_ctx(state_db, base_config, dict(rev), None, jobs=jobs)
    assert extract_pdf(ctx) == 5  # reports the existing run's unit count
    row = state_db.query_one("SELECT COUNT(*) AS n FROM jobs")
    assert row is not None
    n_after = int(row["n"])
    assert n_after == n_jobs
    row = state_db.query_one("SELECT COUNT(*) AS n FROM source_units")
    assert row is not None
    n_units = int(row["n"])
    assert n_units == 5
