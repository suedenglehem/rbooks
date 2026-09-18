"""Extraction failure taxonomy (PRD §7, §8).

Corrupt, encrypted, and structurally invalid sources are *permanent* failures:
retrying the same bytes with the same parser cannot succeed, so the job fails
permanently until the operator explicitly requeues it (``retry
--include-permanent``). Everything that might clear itself (I/O hiccups, a file
that was mid-write) is *transient* and gets backoff/retry.
"""

from __future__ import annotations

__all__ = ["ExtractionFailure", "is_permanent"]


class ExtractionFailure(Exception):
    """A source could not be extracted; the job should be failed accordingly."""

    def __init__(self, category: str, detail: str) -> None:
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


# Error categories that cannot be fixed by retrying the same input.
# OCR: a missing tesseract binary or a missing language pack will not appear
# from re-running the same job; a subprocess error or timeout might clear
# (transient: "ocr_error", "ocr_timeout").
# Pipeline (M3): dangling job inputs (a run or unit that does not exist) and a
# structurally bad range spec are data inconsistencies, not hiccups.
PERMANENT_CATEGORIES = frozenset(
    {"encrypted", "corrupt", "invalid_source", "missing_source",
     "ocr_unavailable", "ocr_language_missing",
     "missing_run", "missing_unit", "run_failed", "invalid_range"}
)


def is_permanent(category: str) -> bool:
    return category in PERMANENT_CATEGORIES
