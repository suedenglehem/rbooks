"""Source extraction stages (M2).

One module per format, plus shared plumbing:

* :mod:`~library_rag.extraction.errors` — failure taxonomy (permanent vs. transient)
* :mod:`~library_rag.extraction.store` — run lifecycle + unit-artifact storage
* :mod:`~library_rag.extraction.pdf` — per-page PDF units (PyMuPDF)
* :mod:`~library_rag.extraction.epub` — per-spine-item EPUB units (ebooklib)
* :mod:`~library_rag.extraction.sanitize` — XHTML sanitization for EPUB bodies
"""

from __future__ import annotations

from .epub import check_epub_safety, extract_epub
from .epub import parser_version as epub_parser_version
from .errors import ExtractionFailure, is_permanent
from .pdf import extract_pdf
from .pdf import parser_version as pdf_parser_version
from .store import (
    SCHEMA_VERSION,
    ExtractorCtx,
    commit_unit_artifact,
    existing_verified_unit,
    fail_run,
    finish_run,
    insert_unit,
    load_unit_artifact,
    start_run,
    unit_artifact_path,
)

__all__ = [
    "SCHEMA_VERSION",
    "ExtractionFailure",
    "ExtractorCtx",
    "check_epub_safety",
    "commit_unit_artifact",
    "epub_parser_version",
    "existing_verified_unit",
    "extract_epub",
    "extract_pdf",
    "fail_run",
    "finish_run",
    "insert_unit",
    "is_permanent",
    "load_unit_artifact",
    "pdf_parser_version",
    "start_run",
    "unit_artifact_path",
]
