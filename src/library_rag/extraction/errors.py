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
PERMANENT_CATEGORIES = frozenset({"encrypted", "corrupt", "invalid_source", "missing_source"})


def is_permanent(category: str) -> bool:
    return category in PERMANENT_CATEGORIES
