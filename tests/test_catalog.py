"""Catalog identity resolution: duplicate / rename / change / unchanged.

These are the M1 gate cases from PRD §6. Each drives the real resolution path
(``register_source``) and asserts the document/revision/alias state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from library_rag.catalog import (
    Format,
    RegistrationStatus,
    get_active_revision,
    list_aliases,
    register_source,
    source_file_count,
)
from library_rag.db import Database
from library_rag.identity import document_id, revision_id

H1 = "11" * 32
H2 = "22" * 32


def _revs(db: Database, doc_id: str) -> list[dict[str, Any]]:
    return [dict(r) for r in db.query(
        "SELECT * FROM source_revisions WHERE doc_id = ? ORDER BY created_at", (doc_id,)
    )]


def test_new_document(state_db: Database) -> None:
    reg = register_source(state_db, "/books/A.pdf", H1, 100, Format.PDF)
    assert reg.status is RegistrationStatus.NEW_DOCUMENT
    assert reg.doc_id == document_id(H1)
    assert reg.rev_id == revision_id(reg.doc_id, H1)
    assert len(_revs(state_db, reg.doc_id)) == 1
    assert source_file_count(state_db) == 1


def test_unchanged_is_idempotent(state_db: Database) -> None:
    first = register_source(state_db, "/books/A.pdf", H1, 100, Format.PDF)
    again = register_source(state_db, "/books/A.pdf", H1, 100, Format.PDF)
    assert again.status is RegistrationStatus.UNCHANGED
    assert again.doc_id == first.doc_id
    assert again.rev_id == first.rev_id
    assert len(_revs(state_db, first.doc_id)) == 1  # no new revision
    assert source_file_count(state_db) == 1  # no duplicate alias


def test_exact_duplicates_become_aliases(state_db: Database) -> None:
    a = register_source(state_db, "/books/A.pdf", H1, 100, Format.PDF)
    c = register_source(state_db, "/books/C.pdf", H1, 100, Format.PDF)
    assert a.status is RegistrationStatus.NEW_DOCUMENT
    assert c.status is RegistrationStatus.ALIAS
    # Same document, same revision; the second path is just an alias.
    assert c.doc_id == a.doc_id
    assert c.rev_id == a.rev_id
    assert len(_revs(state_db, a.doc_id)) == 1
    assert set(list_aliases(state_db, a.doc_id)) == {"/books/A.pdf", "/books/C.pdf"}


def test_rename_does_not_create_a_new_revision(state_db: Database) -> None:
    before = register_source(state_db, "/books/A.pdf", H1, 100, Format.PDF)
    # The operator renames A.pdf -> B.pdf; content is unchanged.
    after = register_source(state_db, "/books/B.pdf", H1, 100, Format.PDF)
    assert after.status is RegistrationStatus.ALIAS
    assert after.doc_id == before.doc_id
    assert after.rev_id == before.rev_id  # same revision => no re-embed
    # Still exactly one revision for the document.
    assert len(_revs(state_db, before.doc_id)) == 1
    active = get_active_revision(state_db, before.doc_id)
    assert active is not None
    assert active["rev_id"] == before.rev_id


def test_content_change_at_known_path_creates_new_revision(state_db: Database) -> None:
    v1 = register_source(state_db, "/books/A.pdf", H1, 100, Format.PDF)
    v2 = register_source(state_db, "/books/A.pdf", H2, 200, Format.PDF)
    assert v1.status is RegistrationStatus.NEW_DOCUMENT
    assert v2.status is RegistrationStatus.NEW_REVISION
    # Same document, but a second, now-active revision.
    assert v2.doc_id == v1.doc_id
    assert v2.rev_id != v1.rev_id
    revs = _revs(state_db, v1.doc_id)
    assert len(revs) == 2
    active = get_active_revision(state_db, v1.doc_id)
    assert active is not None
    assert active["rev_id"] == v2.rev_id
    assert active["sha256"] == H2


def test_change_back_reactivates_prior_revision(state_db: Database) -> None:
    v1 = register_source(state_db, "/books/A.pdf", H1, 100, Format.PDF)
    register_source(state_db, "/books/A.pdf", H2, 200, Format.PDF)
    back = register_source(state_db, "/books/A.pdf", H1, 100, Format.PDF)
    assert back.status is RegistrationStatus.NEW_REVISION
    assert back.rev_id == v1.rev_id  # reactivates, does not fork a third
    revs = _revs(state_db, v1.doc_id)
    assert len(revs) == 2
    active = get_active_revision(state_db, v1.doc_id)
    assert active is not None
    assert active["rev_id"] == v1.rev_id


def test_distinct_editions_are_not_collapsed(state_db: Database) -> None:
    a = register_source(state_db, "/books/A.pdf", H1, 100, Format.PDF)
    b = register_source(state_db, "/books/B.pdf", H2, 300, Format.PDF)
    assert a.doc_id != b.doc_id  # different bytes -> different document


def test_full_vertical_slice_archive_plus_catalog(
    state_db: Database, roots: dict[str, Path], tmp_path: Path
) -> None:
    """Hash -> archive -> register, then rename: the archive is untouched by the rename."""
    from library_rag.archive import ingest_source

    src = tmp_path / "A.pdf"
    data = b"%PDF content for the vertical slice"
    src.write_bytes(data)
    arch = ingest_source(roots["archive_root"], src)
    reg = register_source(state_db, "/books/A.pdf", arch.sha256, arch.size_bytes, Format.PDF)
    assert reg.status is RegistrationStatus.NEW_DOCUMENT

    # Rename: same bytes at a new path, re-archived (dedup) and re-registered.
    renamed = tmp_path / "B.pdf"
    renamed.write_bytes(data)
    arch2 = ingest_source(roots["archive_root"], renamed)
    assert arch2.already_present is True  # content already archived
    reg2 = register_source(state_db, "/books/B.pdf", arch2.sha256, arch2.size_bytes, Format.PDF)
    assert reg2.doc_id == reg.doc_id
    assert reg2.rev_id == reg.rev_id
    assert len(_revs(state_db, reg.doc_id)) == 1
