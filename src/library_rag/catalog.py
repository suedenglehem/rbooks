"""Catalog and identity resolution (PRD §6).

Resolving a scanned file means deciding which *document* it belongs to and
which *revision* it instantiates, then recording the path as an alias. The
rules the PRD requires, and how they fall out of content-anchored identity:

* **Exact duplicates become aliases** — two paths with the same SHA-256 resolve
  to the same document and revision; the second path is recorded as an alias.
* **A rename does not re-embed** — moving a file to a new path (same bytes)
  resolves by content to the existing revision; no new revision is created.
* **A content change at a known path creates a new revision** — a known path
  whose bytes changed resolves to a new revision of the *same* document, which
  becomes active and supersedes the old one.
* **Distinct editions are never auto-collapsed** — identity is anchored to exact
  bytes (content hash), never to titles or similarity, so two genuinely
  different files always yield different revisions.

``register_source`` writes only catalog rows; it does not touch the filesystem.
Archiving the bytes is the caller's job (see :mod:`library_rag.archive`); the
recorded ``archive_relpath`` is a pure function of the content hash, so it agrees
regardless of which process performed the copy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .db import Database
from .identity import document_id, normalize_path, revision_id

__all__ = [
    "Format",
    "Registration",
    "RegistrationStatus",
    "get_active_revision",
    "get_document",
    "list_aliases",
    "register_source",
    "source_file_count",
]


class Format(StrEnum):
    PDF = "pdf"
    EPUB = "epub"


class RegistrationStatus(StrEnum):
    NEW_DOCUMENT = "new_document"
    NEW_REVISION = "new_revision"
    ALIAS = "alias"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class Registration:
    doc_id: str
    rev_id: str
    status: RegistrationStatus


def _upsert_alias(db: Database, path: str, doc_id: str, rev_id: str, now: float) -> None:
    db.execute(
        """
        INSERT INTO path_aliases (path, doc_id, rev_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            doc_id = excluded.doc_id,
            rev_id = excluded.rev_id,
            updated_at = excluded.updated_at
        """,
        (path, doc_id, rev_id, now, now),
    )


def _create_revision(
    db: Database,
    doc_id: str,
    sha256: str,
    size_bytes: int,
    fmt: Format,
    first_path: str,
    now: float,
) -> str:
    rev_id = revision_id(doc_id, sha256)
    db.execute(
        """
        INSERT INTO source_revisions
            (rev_id, doc_id, sha256, size_bytes, format, archive_relpath, first_path,
             is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
        """,
        (rev_id, doc_id, sha256, size_bytes, fmt.value, f"{sha256[:2]}/{sha256}", first_path, now),
    )
    return rev_id


def _set_active(db: Database, doc_id: str, rev_id: str, now: float) -> None:
    db.execute(
        "UPDATE source_revisions SET is_active = 0 WHERE doc_id = ? AND is_active = 1",
        (doc_id,),
    )
    db.execute("UPDATE source_revisions SET is_active = 1 WHERE rev_id = ?", (rev_id,))
    db.execute("UPDATE documents SET updated_at = ? WHERE doc_id = ?", (now, doc_id))


def register_source(
    db: Database,
    path: str,
    sha256: str,
    size_bytes: int,
    fmt: Format,
    now: float | None = None,
) -> Registration:
    """Register *path* holding content *sha256*; resolve its document/revision.

    See the module docstring for the resolution rules. All catalog writes happen
    inside a single short transaction (with the commit crash hooks).
    """
    ts = time.time() if now is None else now
    norm = normalize_path(path)

    with db.transaction():
        alias = db.query_one(
            "SELECT doc_id, rev_id FROM path_aliases WHERE path = ?", (norm,)
        )
        if alias is not None:
            doc_id = alias["doc_id"]
            active = db.query_one(
                "SELECT sha256 FROM source_revisions WHERE rev_id = ?", (alias["rev_id"],)
            )
            active_sha = active["sha256"] if active is not None else None
            if active_sha == sha256:
                return Registration(doc_id, alias["rev_id"], RegistrationStatus.UNCHANGED)

            # Known path, changed bytes -> a new (or reactivated) revision.
            prior = db.query_one(
                "SELECT rev_id FROM source_revisions WHERE doc_id = ? AND sha256 = ?",
                (doc_id, sha256),
            )
            if prior is not None:
                rev_id = prior["rev_id"]
            else:
                rev_id = _create_revision(db, doc_id, sha256, size_bytes, fmt, norm, ts)
            _set_active(db, doc_id, rev_id, ts)
            _upsert_alias(db, norm, doc_id, rev_id, ts)
            return Registration(doc_id, rev_id, RegistrationStatus.NEW_REVISION)

        # Path never seen. Is the content already known (rename / duplicate)?
        known = db.query_one(
            "SELECT doc_id, rev_id FROM source_revisions WHERE sha256 = ?", (sha256,)
        )
        if known is not None:
            _upsert_alias(db, norm, known["doc_id"], known["rev_id"], ts)
            return Registration(known["doc_id"], known["rev_id"], RegistrationStatus.ALIAS)

        # Genuinely new content -> a new document anchored to this content.
        doc_id = document_id(sha256)
        db.execute(
            "INSERT INTO documents (doc_id, anchor_sha256, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, sha256, ts, ts),
        )
        rev_id = _create_revision(db, doc_id, sha256, size_bytes, fmt, norm, ts)
        _upsert_alias(db, norm, doc_id, rev_id, ts)
        return Registration(doc_id, rev_id, RegistrationStatus.NEW_DOCUMENT)


def get_document(db: Database, doc_id: str) -> dict[str, Any] | None:
    row = db.query_one("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
    return dict(row) if row is not None else None


def get_active_revision(db: Database, doc_id: str) -> dict[str, Any] | None:
    row = db.query_one(
        "SELECT * FROM source_revisions WHERE doc_id = ? AND is_active = 1", (doc_id,)
    )
    return dict(row) if row is not None else None


def list_aliases(db: Database, doc_id: str) -> list[str]:
    return [r["path"] for r in db.query(
        "SELECT path FROM path_aliases WHERE doc_id = ? ORDER BY path", (doc_id,)
    )]


def source_file_count(db: Database) -> int:
    """Number of registered source paths (aliases), i.e. files in the catalog."""
    row = db.query_one("SELECT COUNT(*) AS n FROM path_aliases")
    return int(row["n"]) if row is not None else 0
