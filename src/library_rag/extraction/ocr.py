"""Selective OCR engine (PRD §8C).

Only pages routed to ``ocr`` by :mod:`library_rag.extraction.routing` are
rasterized, and the raster is a bounded scratch file (deleted after the
Tesseract call) that is never archived. The engine runs the Tesseract CLI in
TSV mode so every recognized word carries a pixel box; those boxes are mapped
back to page coordinates for citation highlights.

**Coordinate mapping.** The pixmap is rendered with
``page.get_pixmap(matrix=Matrix(s, s))`` where *s* is the (possibly shrunk)
DPI scale. PyMuPDF's ``page.rect`` and ``page.search_for`` already work in the
rotation-aware display space, and the pixmap is the page rect at the exact
scale product rounded outward (see :func:`build_transform`), so raster→page is
*division by the scale only* — no rotation math (empirically verified for
0°/90°/180°/270° pages and cropped boxes).
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pymupdf

from ..config import OcrSettings
from .errors import ExtractionFailure

__all__ = [
    "build_transform",
    "ocr_page",
    "page_to_raster_box",
    "parse_tsv",
    "raster_to_page_box",
]

_LANG_LINE = re.compile(r"[A-Za-z0-9\-_]+\Z")


def build_transform(page: Any, settings: OcrSettings) -> dict[str, Any]:
    """Raster transform for one page: DPI scale, shrunk to the pixel bound.

    Returns the scale plus the page and render dimensions so callers (and the
    per-word boxes stored in the artifact) can map raster coordinates back to
    page points with a single division.

    The render size must predict the pixmap PyMuPDF actually produces, within
    1 px per dimension: PyMuPDF sizes the raster from the matrix product in
    its own float association, which for the max-side shrink can land 1 px
    below the ``ceil(opp * max_side / longest)`` predicted here (pilot run:
    2058 vs 2059). The caller checks with a ±1 px tolerance, and citation
    boxes divide by the exact scale, so the discrepancy is harmless.
    """
    scale = settings.dpi / 72.0
    rect = page.rect
    longest = max(rect.width, rect.height)
    if longest * scale > settings.max_side_px:
        scale = settings.max_side_px / longest
        if rect.width >= rect.height:
            render_w: int = settings.max_side_px
            render_h: int = math.ceil(rect.height * settings.max_side_px / rect.width)
        else:
            render_h = settings.max_side_px
            render_w = math.ceil(rect.width * settings.max_side_px / rect.height)
    else:
        render_w = math.ceil(rect.width * settings.dpi / 72)
        render_h = math.ceil(rect.height * settings.dpi / 72)
    return {
        "dpi": settings.dpi,
        "scale": scale,
        "rotation": int(page.rotation),
        "page_w": float(rect.width),
        "page_h": float(rect.height),
        "render_w": render_w,
        "render_h": render_h,
    }


def raster_to_page_box(box: list[float], transform: dict[str, Any]) -> list[float]:
    """Map a raster-pixel box [x, y, w, h] to page points (pure scale division)."""
    s = float(transform["scale"])
    return [round(c / s, 3) for c in box]


def page_to_raster_box(box: list[float], transform: dict[str, Any]) -> list[float]:
    """Inverse of :func:`raster_to_page_box` (page points → raster pixels)."""
    s = float(transform["scale"])
    return [round(c * s, 3) for c in box]


def parse_tsv(tsv: str) -> list[dict[str, Any]]:
    """Parse Tesseract TSV output into word rows.

    Only level-5 (word) rows with non-empty text are kept. Tesseract emits the
    12 standard columns: level, page_num, block_num, par_num, line_num,
    word_num, left, top, width, height, conf, text.
    """
    words: list[dict[str, Any]] = []
    for line in tsv.splitlines():
        parts = line.split("\t")
        if len(parts) < 12 or parts[0] == "level":
            continue
        if parts[0].strip() != "5":
            continue
        text = parts[11]
        if not text.strip():
            continue
        try:
            left = round(float(parts[6]))
            top = round(float(parts[7]))
            width = round(float(parts[8]))
            height = round(float(parts[9]))
            conf = float(parts[10])
            block = int(parts[2])
            par = int(parts[3])
            line_no = int(parts[4])
        except ValueError:
            continue  # malformed row: drop it, keep the rest
        words.append(
            {"text": text, "box": [left, top, width, height], "conf": conf,
             "block": block, "par": par, "line": line_no}
        )
    return words


def _installed_languages(bin_path: str) -> set[str]:
    """Language packs tesseract reports for ``--list-langs``.

    Scans both streams: tesseract prints the language list on stdout but its
    version/opencv banners on stderr; the strict full-match keeps both.
    """
    try:
        proc = subprocess.run([bin_path, "--list-langs"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if proc.returncode != 0:
        return set()
    out = proc.stdout.decode("utf-8", "replace") + "\n" + proc.stderr.decode("utf-8", "replace")
    return {
        line.strip()
        for line in out.splitlines()
        if _LANG_LINE.fullmatch(line.strip())
    }


def _engine_version(bin_path: str) -> str:
    try:
        proc = subprocess.run([bin_path, "--version"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    text = proc.stderr.decode("utf-8", "replace") or proc.stdout.decode("utf-8", "replace")
    first = text.splitlines()
    return first[0].strip() if first and first[0].strip() else "unknown"


def ocr_page(doc: Any, index: int, settings: OcrSettings, scratch_dir: Path | str) -> dict[str, Any]:
    """OCR one page of an open PyMuPDF document via the Tesseract CLI.

    Returns ``{text, words, engine, languages, psm, transform}`` where each
    word carries a *page-point* box (raster box divided by the scale) ready for
    citation highlight mapping. Raises :class:`ExtractionFailure`:
    ``ocr_unavailable`` / ``ocr_language_missing`` (permanent), ``ocr_timeout`` /
    ``ocr_error`` (transient). The raster scratch file is always removed.
    """
    bin_path = shutil.which(settings.bin)
    if bin_path is None:
        raise ExtractionFailure("ocr_unavailable", f"tesseract binary not found: {settings.bin!r}")
    missing = [lang for lang in settings.languages if lang not in _installed_languages(bin_path)]
    if missing:
        raise ExtractionFailure(
            "ocr_language_missing",
            f"tesseract language pack(s) missing: {missing}",
        )

    page = doc[index]
    transform = build_transform(page, settings)
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    png = scratch / f"page-{index}.png"
    try:
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(transform["scale"], transform["scale"]))  # type: ignore[no-untyped-call]
        # Sanity: the render size must be what the transform promised. PyMuPDF
        # computes the raster size from the matrix in a different float
        # association than build_transform's ceil(), so it can land 1 px off
        # the predicted bound (pilot run: 1400x2058 vs expected 1400x2059 on
        # 162 pages). Tolerate ±1 px per dimension: coordinate mapping divides
        # by the exact scale, never by the render size, so a 1 px raster
        # difference cannot skew a citation box.
        if (
            abs(pixmap.width - transform["render_w"]) > 1
            or abs(pixmap.height - transform["render_h"]) > 1
        ):
            raise ExtractionFailure(
                "ocr_error",
                f"unexpected raster size {pixmap.width}x{pixmap.height}, "
                f"expected {transform['render_w']}x{transform['render_h']}",
            )
        pixmap.save(str(png))
        cmd = [
            bin_path, str(png), "stdout",
            "-l", "+".join(settings.languages),
            "--psm", str(settings.psm),
            "tsv",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=settings.timeout_seconds,
                env={**os.environ, "OMP_THREAD_LIMIT": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            raise ExtractionFailure(
                "ocr_timeout", f"tesseract exceeded {settings.timeout_seconds}s on page {index}"
            ) from exc
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", "replace")[-500:]
            raise ExtractionFailure("ocr_error", f"tesseract exit {proc.returncode}: {stderr}")
        words = [
            {**w, "box": raster_to_page_box(w["box"], transform)}
            for w in parse_tsv(proc.stdout.decode("utf-8", "replace"))
        ]
    finally:
        with contextlib.suppress(OSError):
            png.unlink(missing_ok=True)

    # Reassemble text: words joined within a line, lines joined with newlines.
    lines: dict[tuple[int, int, int], list[str]] = {}
    order: list[tuple[int, int, int]] = []
    for w in words:
        key = (w["block"], w["par"], w["line"])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(w["text"])
    text = "\n".join(" ".join(lines[k]) for k in order)

    return {
        "text": text,
        "words": [
            {"text": w["text"], "box": w["box"], "conf": w["conf"], "line": w["line"]}
            for w in words
        ],
        "engine": _engine_version(bin_path),
        "languages": list(settings.languages),
        "psm": settings.psm,
        "transform": transform,
    }
