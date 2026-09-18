"""Source extraction stages (M2).

One module per format, plus shared plumbing:

* :mod:`~library_rag.extraction.errors` — failure taxonomy (permanent vs. transient)
* :mod:`~library_rag.extraction.store` — run lifecycle + unit-artifact storage
* :mod:`~library_rag.extraction.pdf` — per-page PDF units (PyMuPDF)
* :mod:`~library_rag.extraction.epub` — per-spine-item EPUB units (ebooklib)
* :mod:`~library_rag.extraction.sanitize` — XHTML sanitization for EPUB bodies
* :mod:`~library_rag.extraction.routing` — selective-OCR page routing (PRD §8C)
* :mod:`~library_rag.extraction.ocr` — Tesseract CLI OCR engine + coordinate mapping
"""

from __future__ import annotations

from .epub import check_epub_safety, extract_epub
from .epub import parser_version as epub_parser_version
from .errors import ExtractionFailure, is_permanent
from .ocr import (
    build_transform,
    ocr_page,
    page_to_raster_box,
    parse_tsv,
    raster_to_page_box,
)
from .pdf import extract_pdf
from .pdf import parser_version as pdf_parser_version
from .routing import ROUTE_OCR, ROUTE_REUSE, ROUTE_SKIP, image_area_ratio, route_page
from .store import (
    SCHEMA_VERSION,
    ExtractorCtx,
    commit_unit_artifact,
    existing_verified_unit,
    fail_run,
    finish_run,
    insert_unit,
    load_unit_artifact,
    refresh_unit_artifact,
    set_unit_ocr,
    start_run,
    unit_artifact_path,
)

__all__ = [
    "ROUTE_OCR",
    "ROUTE_REUSE",
    "ROUTE_SKIP",
    "SCHEMA_VERSION",
    "ExtractionFailure",
    "ExtractorCtx",
    "build_transform",
    "check_epub_safety",
    "commit_unit_artifact",
    "epub_parser_version",
    "existing_verified_unit",
    "extract_epub",
    "extract_pdf",
    "fail_run",
    "finish_run",
    "image_area_ratio",
    "insert_unit",
    "is_permanent",
    "load_unit_artifact",
    "ocr_page",
    "page_to_raster_box",
    "parse_tsv",
    "pdf_parser_version",
    "raster_to_page_box",
    "refresh_unit_artifact",
    "route_page",
    "set_unit_ocr",
    "start_run",
    "unit_artifact_path",
]
