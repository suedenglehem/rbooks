"""Reconciliation: a missing mount cannot delete content; present mounts prune."""

from __future__ import annotations

from pathlib import Path

from library_rag.archive import archive_path_for, ingest_source
from library_rag.catalog import Format, get_document, register_source, source_file_count
from library_rag.db import Database
from library_rag.reconcile import Mount, reconcile_catalog

H = "44" * 32


def test_missing_mount_cannot_delete_content(
    state_db: Database, roots: dict[str, Path], tmp_path: Path
) -> None:
    mount_root = tmp_path / "books"
    mount_root.mkdir()
    sentinel_missing = tmp_path / "sentinel_missing"  # never created
    path = str(mount_root / "A.pdf")
    register_source(state_db, path, H, 10, Format.PDF)

    # Also archive the bytes so we can prove reconcile never touches the archive.
    src = tmp_path / "A.pdf.orig"
    src.write_bytes(b"%PDF original bytes")
    arch = ingest_source(roots["archive_root"], src)

    report = reconcile_catalog(
        state_db,
        [Mount(label="books", root=mount_root, sentinel=sentinel_missing)],
        visible_paths=set(),  # the file appears "gone"
    )
    assert "books" in report.unavailable_mounts
    assert report.pruned_aliases == 0
    assert report.deleted_content == 0
    # The catalog entry survived the "disconnected mount".
    assert source_file_count(state_db) == 1
    doc = get_document(state_db, _doc_id_of(state_db))
    assert doc is not None
    # And the archived original is intact.
    assert archive_path_for(roots["archive_root"], arch.sha256).exists()


def _doc_id_of(db: Database) -> str:
    row = db.query_one("SELECT doc_id FROM path_aliases LIMIT 1")
    return row["doc_id"] if row is not None else ""


def test_present_mount_prunes_stale_alias_but_keeps_content(
    state_db: Database, roots: dict[str, Path], tmp_path: Path
) -> None:
    mount_root = tmp_path / "books"
    mount_root.mkdir()
    sentinel_ok = tmp_path / "sentinel_ok"
    sentinel_ok.write_text("ok")
    path = str(mount_root / "A.pdf")
    reg = register_source(state_db, path, H, 10, Format.PDF)

    # A.pdf is now deleted from the (present) mount: visible is empty.
    report = reconcile_catalog(
        state_db,
        [Mount(label="books", root=mount_root, sentinel=sentinel_ok)],
        visible_paths=set(),
    )
    assert "books" not in report.unavailable_mounts
    assert report.pruned_aliases == 1
    assert source_file_count(state_db) == 0  # alias pruned
    # But the document and its revision (the content) are NOT deleted.
    assert get_document(state_db, reg.doc_id) is not None
    rev = state_db.query_one("SELECT 1 AS x FROM source_revisions WHERE rev_id = ?", (reg.rev_id,))
    assert rev is not None


def test_present_mount_keeps_visible_alias(
    state_db: Database, roots: dict[str, Path], tmp_path: Path
) -> None:
    from library_rag.identity import normalize_path

    mount_root = tmp_path / "books"
    mount_root.mkdir()
    sentinel_ok = tmp_path / "sentinel_ok"
    sentinel_ok.write_text("ok")
    path = str(mount_root / "A.pdf")
    register_source(state_db, path, H, 10, Format.PDF)

    report = reconcile_catalog(
        state_db,
        [Mount(label="books", root=mount_root, sentinel=sentinel_ok)],
        visible_paths={normalize_path(path)},
    )
    assert report.pruned_aliases == 0
    assert source_file_count(state_db) == 1
