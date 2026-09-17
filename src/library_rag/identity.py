"""Deterministic identities.

The identity contract (PRD §6) is content-anchored so that the same bytes always
resolve to the same catalog entry and, later, the same Qdrant point ID:

* **Document** — a catalog entry. Its identity is derived from the SHA-256 of the
  *first* content that established it (its "anchor"). Because the anchor is the
  content, a rename (same bytes, new path) never changes the document, and a
  rebuild from a rescan of unchanged content reproduces the same document ID.
* **Source revision** — a specific content of a document. Identified by
  (document, content hash); a rename reuses the existing revision, a content
  change at a known path creates a new one.
* **Task key** — the idempotency key for a durable job, a pure function of the
  stage, input identity, and the versions/configs that define its work.

IDs are UUID-version-5 (name-based, SHA-1) so they are stable across processes
and across a database rebuild, while remaining "UUID-format" for use as Qdrant
point IDs. The full content hashes are stored separately for audit; the UUIDs
are *derived from* them, not substitutes.
"""

from __future__ import annotations

import os
import uuid
from pathlib import PurePosixPath

__all__ = [
    "NAMESPACE_DOC",
    "NAMESPACE_REV",
    "NAMESPACE_TASK",
    "document_id",
    "make_task_key",
    "normalize_path",
    "revision_id",
]

# Fixed name-registry UUIDs. Chosen once and never changed; changing them would
# re-identify the whole library, so they are treated like a schema constant.
NAMESPACE_DOC = uuid.UUID("8f3e2a10-0000-4000-8000-0000000000d1")
NAMESPACE_REV = uuid.UUID("8f3e2a10-0000-4000-8000-0000000000d2")
NAMESPACE_TASK = uuid.UUID("8f3e2a10-0000-4000-8000-0000000000d3")


def normalize_path(path: str | os.PathLike[str]) -> str:
    """Return the canonical path string used as a catalog key.

    Absolute and lexically normalized (``os.path.normpath``), stored in POSIX
    form. We deliberately use ``abspath`` (not ``resolve``) so identity does not
    depend on symlink resolution — consistent with the path-overlap rules in
    :mod:`library_rag.config`.
    """
    ap = os.path.normpath(os.path.abspath(os.fspath(path)))
    return PurePosixPath(ap).as_posix()


def document_id(anchor_sha256: str) -> str:
    """Deterministic document UUID from the first (anchor) content hash."""
    return str(uuid.uuid5(NAMESPACE_DOC, anchor_sha256))


def revision_id(doc_id: str, sha256: str) -> str:
    """Deterministic revision UUID for a given document and content hash."""
    return str(uuid.uuid5(NAMESPACE_REV, f"{doc_id}:{sha256}"))


def make_task_key(
    stage: str,
    input_id: str,
    input_version: str,
    range_spec: str = "",
) -> str:
    """Deterministic idempotency key for a durable job.

    Two jobs with the same stage, input identity, and defining versions produce
    the same key, so re-enqueueing after a crash is a no-op (at-least-once with
    idempotent effects, PRD §7).
    """
    return str(uuid.uuid5(NAMESPACE_TASK, f"{stage}:{input_id}:{input_version}:{range_spec}"))
