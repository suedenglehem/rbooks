"""Selective OCR (PRD §8C): TSV parsing, raster→page coordinate mapping
(including rotated/cropped pages — the highlight-mapping gate item), the
ocr_page failure taxonomy, and the kill-after-N → resume behavior of the
durable OCR jobs (the M3 gate item).

Tesseract is not installed on the test host (installing it needs operator
approval, PRD §1), so every subprocess case drives ``make_fake_tesseract``,
a CLI shim that speaks the same interface and emits a known 3-word grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from fixtures import (
    make_cropped_pdf,
    make_fake_tesseract,
    make_mixed_pdf,
    make_pdf,
    make_rotated_pdf,
)
from library_rag.config import Config, OcrSettings
from library_rag.db import Database
from library_rag.embeddings import FakeEmbedder
from library_rag.extraction.errors import ExtractionFailure, is_permanent
from library_rag.extraction.ocr import (
    build_transform,
    ocr_page,
    page_to_raster_box,
    parse_tsv,
    raster_to_page_box,
)
from library_rag.indexing import FakeQdrant
from library_rag.jobs import Claimed, Jobs
from library_rag.scan import scan_roots
from library_rag.worker import run_worker

_TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num"
    "\tleft\ttop\twidth\theight\tconf\ttext"
)


# --- parse_tsv ---------------------------------------------------------------
def test_parse_tsv_keeps_word_rows_with_text() -> None:
    tsv = "\n".join(
        [
            _TSV_HEADER,
            "1\t1\t1\t0\t0\t0\t0\t0\t100\t100\t0\t",  # page row: skipped
            "4\t1\t1\t1\t2\t0\t10\t40\t300\t15\t0\t",  # line row: skipped
            "5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t95.5\tHello",
            "5\t1\t1\t1\t1\t2\t50\t20\t30\t12\t-1\t",  # empty text: skipped
            "5\t1\t1\t1\t2\t1\t10\t40\t60\t12\t80\tworld",
        ]
    )
    words = parse_tsv(tsv)
    assert [w["text"] for w in words] == ["Hello", "world"]
    assert words[0]["box"] == [10, 20, 30, 12]
    assert words[0]["conf"] == 95.5
    assert (words[0]["block"], words[0]["par"], words[0]["line"]) == (1, 1, 1)
    assert words[1]["line"] == 2


def test_parse_tsv_drops_malformed_rows() -> None:
    tsv = "\n".join(
        [
            _TSV_HEADER,
            "5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t95\tok",
            "5\t1\t1\t1\t1\t1\tabc\t20\t30\t12\t95\tbad",  # left not numeric
            "5\t1\t1\t1\t1",  # too few columns
            "5\t1\t1\t1\t1\t2\t40\t20\t30\t12\t90\tfine",
        ]
    )
    words = parse_tsv(tsv)
    assert [w["text"] for w in words] == ["ok", "fine"]


# --- raster <-> page coordinate mapping (highlight-mapping gate item) --------
def _roundtrip(path: Path, page_index: int) -> dict[str, Any]:
    doc = pymupdf.open(str(path))  # type: ignore[no-untyped-call]
    try:
        page = doc[page_index]
        transform = build_transform(page, OcrSettings())
        # The pixmap the renderer actually produces must match the promise.
        pix = page.get_pixmap(matrix=pymupdf.Matrix(transform["scale"], transform["scale"]))  # type: ignore[no-untyped-call]
        assert (pix.width, pix.height) == (transform["render_w"], transform["render_h"])
        # Raster corners map back to the rotation-aware page rect corners.
        page_box = raster_to_page_box([0.0, 0.0, transform["render_w"], transform["render_h"]], transform)
        assert page_box[0] == pytest.approx(0.0, abs=0.5)
        assert page_box[1] == pytest.approx(0.0, abs=0.5)
        assert page_box[2] == pytest.approx(transform["page_w"], abs=0.5)
        assert page_box[3] == pytest.approx(transform["page_h"], abs=0.5)
        # And the inverse maps page corners back to raster corners.
        raster = page_to_raster_box([0.0, 0.0, transform["page_w"], transform["page_h"]], transform)
        assert raster[0] == pytest.approx(0.0, abs=0.5)
        assert raster[1] == pytest.approx(0.0, abs=0.5)
        assert raster[2] == pytest.approx(transform["render_w"], abs=1.5)
        assert raster[3] == pytest.approx(transform["render_h"], abs=1.5)
        return transform
    finally:
        doc.close()  # type: ignore[no-untyped-call]


def test_coordinate_roundtrip_straight(tmp_path: Path) -> None:
    transform = _roundtrip(make_mixed_pdf(tmp_path / "b.pdf", ["text"]), 0)
    assert transform["rotation"] == 0
    # A4 at 300 dpi: 595x842 points -> 2480x3509 pixels (exact product, ceil).
    assert (transform["render_w"], transform["render_h"]) == (2480, 3509)


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_coordinate_roundtrip_rotated(tmp_path: Path, rotation: int) -> None:
    # The raster box of a rendered (rotated) page must map back onto the
    # rotation-aware page rect — otherwise OCR highlights land on the wrong
    # PDF.js page region (M3 gate: "rotated OCR highlight mapping tested").
    transform = _roundtrip(make_rotated_pdf(tmp_path / "b.pdf", rotation=rotation), 0)
    assert transform["rotation"] == rotation
    if rotation % 180 == 90:  # 90/270: page rect is transposed
        assert (transform["page_w"], transform["page_h"]) == (842.0, 595.0)
        assert (transform["render_w"], transform["render_h"]) == (3509, 2480)


def test_coordinate_roundtrip_cropped(tmp_path: Path) -> None:
    # Cropbox 50,50,500,500 -> a 450x450 point page, rendered at 1875x1875.
    transform = _roundtrip(make_cropped_pdf(tmp_path / "b.pdf"), 0)
    assert (transform["page_w"], transform["page_h"]) == (450.0, 450.0)
    assert (transform["render_w"], transform["render_h"]) == (1875, 1875)


def test_transform_shrinks_to_max_side(tmp_path: Path) -> None:
    doc = pymupdf.open(str(make_pdf(tmp_path / "b.pdf", ["x"])))  # type: ignore[no-untyped-call]
    try:
        transform = build_transform(doc[0], OcrSettings(max_side_px=1000))
        assert transform["render_h"] == 1000  # longest side hits the bound
        assert transform["render_w"] < 1000
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(transform["scale"], transform["scale"]))  # type: ignore[no-untyped-call]
        assert (pix.width, pix.height) == (transform["render_w"], transform["render_h"])
    finally:
        doc.close()  # type: ignore[no-untyped-call]


# --- ocr_page via the fake tesseract ------------------------------------------
@pytest.fixture
def shim_bin(tmp_path: Path) -> Path:
    return make_fake_tesseract(tmp_path / "shim" / "tesseract")


def _one_page_doc(tmp_path: Path) -> Path:
    return make_pdf(tmp_path / "one.pdf", ["x"])


def test_ocr_page_words_map_to_page_points(tmp_path: Path, shim_bin: Path) -> None:
    scratch = tmp_path / "scratch"
    doc = pymupdf.open(str(_one_page_doc(tmp_path)))  # type: ignore[no-untyped-call]
    try:
        result = ocr_page(doc, 0, OcrSettings(bin=str(shim_bin)), scratch)
    finally:
        doc.close()  # type: ignore[no-untyped-call]

    # The shim emits one word per line: w0-0, w0-1, w0-2.
    assert result["text"] == "w0-0\nw0-1\nw0-2"
    assert [w["text"] for w in result["words"]] == ["w0-0", "w0-1", "w0-2"]
    assert [w["conf"] for w in result["words"]] == [85.0, 86.0, 87.0]
    assert [w["line"] for w in result["words"]] == [1, 2, 3]
    # Raster boxes (50,100,200,50) & co divided by scale 300/72.
    expected = [
        [12.0, 24.0, 48.0, 12.0],
        [64.8, 24.0, 48.0, 12.0],
        [117.6, 24.0, 48.0, 12.0],
    ]
    assert result["words"][0]["box"] == expected[0]
    for word, box in zip(result["words"][1:], expected[1:], strict=True):
        assert [round(c, 3) for c in word["box"]] == box
    assert result["engine"] == "tesseract 5.3.0-fake"
    assert result["languages"] == ["eng"]
    assert result["psm"] == 3
    assert result["transform"]["rotation"] == 0
    # The raster scratch file is always cleaned up.
    assert scratch.exists()
    assert list(scratch.iterdir()) == []


def test_ocr_page_failure_taxonomy(tmp_path: Path, shim_bin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / "scratch"
    doc = pymupdf.open(str(_one_page_doc(tmp_path)))  # type: ignore[no-untyped-call]
    try:
        # Permanent: binary missing from PATH.
        with pytest.raises(ExtractionFailure) as exc:
            ocr_page(doc, 0, OcrSettings(bin="/nonexistent/tesseract"), scratch)
        assert exc.value.category == "ocr_unavailable"
        assert is_permanent(exc.value.category)

        # Permanent: language pack not installed.
        monkeypatch.setenv("FAKE_TESSERACT_LANGS", "osd")
        with pytest.raises(ExtractionFailure) as exc:
            ocr_page(doc, 0, OcrSettings(bin=str(shim_bin)), scratch)
        assert exc.value.category == "ocr_language_missing"
        assert is_permanent(exc.value.category)

        # Transient: the engine itself failed.
        monkeypatch.delenv("FAKE_TESSERACT_LANGS")
        monkeypatch.setenv("FAKE_TESSERACT_FAIL", "1")
        with pytest.raises(ExtractionFailure) as exc:
            ocr_page(doc, 0, OcrSettings(bin=str(shim_bin)), scratch)
        assert exc.value.category == "ocr_error"
        assert not is_permanent(exc.value.category)

        # Transient: the engine timed out.
        monkeypatch.delenv("FAKE_TESSERACT_FAIL")
        monkeypatch.setenv("FAKE_TESSERACT_SLEEP", "2")
        with pytest.raises(ExtractionFailure) as exc:
            ocr_page(doc, 0, OcrSettings(bin=str(shim_bin), timeout_seconds=0.5), scratch)
        assert exc.value.category == "ocr_timeout"
        assert not is_permanent(exc.value.category)
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    assert list(scratch.iterdir()) == []  # no PNG left behind on failure


# --- kill after page N, resume from unfinished units (M3 gate item) -----------
@pytest.fixture
def src(base_config: Config) -> Path:
    root = base_config.paths.source_roots[0]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _drain(state_db: Database, cfg: Config, q: FakeQdrant, emb: FakeEmbedder) -> int:
    return run_worker(state_db, cfg, once=True, poll_delay=0, qdrant=q, embedder=emb)


def test_ocr_kill_after_page_n_resumes(
    state_db: Database,
    base_config: Config,
    src: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shim = make_fake_tesseract(tmp_path / "shim" / "tesseract")
    base_config.extraction.ocr.bin = str(shim)
    log = tmp_path / "tess.log"
    monkeypatch.setenv("FAKE_TESSERACT_LOG", str(log))

    make_mixed_pdf(src / "scan.pdf", ["scanned"] * 5)
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)

    # Kill the worker after the second OCR page has committed: subsequent
    # claims stop at the next loop check.
    ocr_done = [0]
    stop = [False]
    real_succeed = Jobs.succeed

    def stopping_succeed(self: Jobs, job: Claimed, output_manifest: str, now: float | None = None) -> None:
        real_succeed(self, job, output_manifest, now=now)
        if job.stage == "ocr":
            ocr_done[0] += 1
            if ocr_done[0] >= 2:
                stop[0] = True

    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)

    monkeypatch.setattr(Jobs, "succeed", stopping_succeed)
    # The "kill": a stop signal that the loop observes at its next check, i.e.
    # right after page 1's OCR job has committed.
    first = run_worker(
        state_db, base_config, once=True, poll_delay=0,
        stop_event=lambda: stop[0], qdrant=q, embedder=emb,
    )
    monkeypatch.setattr(Jobs, "succeed", real_succeed)

    # extract + OCR pages 0,1 — then the stop check ends the loop.
    assert first == 3
    assert ocr_done[0] == 2

    # Resume: a fresh worker picks up the three unfinished OCR units, the
    # chunk job, and the M4 embed/publish tail. No page is re-OCR'd.
    second = _drain(state_db, base_config, q, emb)
    assert second == 6
    assert jobs.counts() == {"succeeded": 9}  # extract + 5 OCR + chunk + embed + publish

    units = state_db.query(
        "SELECT unit_id, ocr_state, route FROM source_units ORDER BY position"
    )
    assert len(units) == 5
    assert all(u["route"] == "ocr" for u in units)
    assert all(u["ocr_state"] == "done" for u in units)

    # Exactly one tesseract call per page: the two pages done before the kill
    # were not re-run.
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert sorted(json.loads(line)["page"] for line in lines) == [0, 1, 2, 3, 4]

    # One chunk (15 tokens < target): one span per page unit, each with the
    # union bbox of that page's three OCR word boxes.
    chunks = state_db.query("SELECT * FROM chunks")
    assert len(chunks) == 1
    assert chunks[0]["text"] == " ".join(f"w{p}-{j}" for p in range(5) for j in range(3))
    spans = json.loads(chunks[0]["spans"])
    assert len(spans) == 5
    assert [s["unit_id"] for s in spans] == [u["unit_id"] for u in units]
    for span in spans:
        # [12,24]..[117.6,24] + 48x12 -> union [12, 24, 165.6, 36].
        assert span["bbox"] == [12.0, 24.0, 165.6, 36.0]

    # No raster scratch files survive anywhere in this run.
    run = state_db.query_one("SELECT run_id FROM extraction_runs")
    assert run is not None
    scratch = base_config.paths.scratch_root / "ocr" / run["run_id"]
    assert list(scratch.iterdir()) == []


def test_ocr_noop_replay_skips_finished_units(
    state_db: Database,
    base_config: Config,
    src: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shim = make_fake_tesseract(tmp_path / "shim" / "tesseract")
    base_config.extraction.ocr.bin = str(shim)
    log = tmp_path / "tess.log"
    monkeypatch.setenv("FAKE_TESSERACT_LOG", str(log))

    make_mixed_pdf(src / "scan.pdf", ["scanned"] * 3)
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)

    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    assert _drain(state_db, base_config, q, emb) == 7  # extract + 3 OCR + chunk + embed + publish
    assert log.read_text(encoding="utf-8").count("\n") == 3

    # A lost/requeued OCR job for an already-finished page is a durable no-op:
    # the unit is already done under the same OCR settings.
    job = state_db.query_one("SELECT * FROM jobs WHERE stage = 'ocr' AND range_spec = 'page:0'")
    assert job is not None
    state_db.execute(
        "UPDATE jobs SET state='pending', attempts=0, lease_token=NULL, "
        "lease_expires_at=NULL, error_category=NULL, error_detail=NULL, "
        "next_attempt_at=NULL WHERE job_id = ?",
        (job["job_id"],),
    )

    assert _drain(state_db, base_config, q, emb) == 1
    row = state_db.query_one("SELECT * FROM jobs WHERE job_id = ?", (job["job_id"],))
    assert row is not None
    assert row["state"] == "succeeded"
    manifest = json.loads(row["output_manifest"])
    assert manifest["noop"] is True
    # No tesseract call was made on replay.
    assert log.read_text(encoding="utf-8").count("\n") == 3
    assert jobs.counts() == {"succeeded": 7}


def test_ocr_transient_failure_retries_then_recovers(
    state_db: Database,
    base_config: Config,
    src: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shim = make_fake_tesseract(tmp_path / "shim" / "tesseract")
    base_config.extraction.ocr.bin = str(shim)
    monkeypatch.setenv("FAKE_TESSERACT_FAIL", "1")  # every OCR call fails

    make_mixed_pdf(src / "scan.pdf", ["scanned"] * 2)
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)

    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    # Extract succeeds; both OCR pages fail transiently (retryable_failed);
    # the chunk job sees open OCR work and defers (embed is not yet enqueued:
    # only a successful chunk hands off to embed).
    assert _drain(state_db, base_config, q, emb) == 4
    counts = jobs.counts()
    assert counts["succeeded"] == 1
    assert counts.get("retryable_failed", 0) == 3
    ocr_jobs = state_db.query("SELECT error_category FROM jobs WHERE stage = 'ocr'")
    assert {r["error_category"] for r in ocr_jobs} == {"ocr_error"}

    # Operator clears the fault and retries: OCR recovers, the chunk runs,
    # and the M4 embed/publish tail completes the publication.
    monkeypatch.delenv("FAKE_TESSERACT_FAIL")
    assert jobs.retry() >= 3
    assert _drain(state_db, base_config, q, emb) == 5  # 2 OCR + chunk + embed + publish
    assert jobs.counts() == {"succeeded": 6}
    units = state_db.query("SELECT ocr_state FROM source_units")
    assert all(u["ocr_state"] == "done" for u in units)
    row = state_db.query_one("SELECT COUNT(*) AS n FROM chunks")
    assert row is not None
    assert int(row["n"]) == 1
