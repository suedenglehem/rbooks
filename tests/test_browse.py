"""M10 browse: path containment, one-level listings, active-rev join, routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from library_rag.api import create_app
from library_rag.browse import (
    BrowseError,
    BrowseNotFound,
    list_browse_dir,
    lookup_active_revisions,
    resolve_browse_path,
)
from library_rag.catalog import Format, RegistrationStatus, register_source
from library_rag.config import BrowseSettings, Config
from library_rag.db import Database
from library_rag.identity import document_id, revision_id
from library_rag.indexing import FakeQdrant

# Same sha convention as test_catalog.py.
H1 = "11" * 32
H2 = "22" * 32

FT = [".pdf", ".epub"]


def make_root(root: Path) -> None:
    """A small books tree exercising filter, sort, and nesting."""
    (root / "sub").mkdir(parents=True)
    (root / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (root / "B.PDF").write_bytes(b"%PDF-1.4 b")
    (root / "c.epub").write_bytes(b"EPUB c")
    (root / "notes.txt").write_bytes(b"nope")
    (root / "sub" / "d.EPUB").write_bytes(b"EPUB d")
    (root / "sub" / "readme.md").write_bytes(b"nope")


# --- resolve_browse_path ------------------------------------------------------


def test_resolve_root_itself(tmp_path: Path) -> None:
    root = tmp_path / "books"
    root.mkdir()
    assert resolve_browse_path(root, "") == str(root)
    assert resolve_browse_path(root, ".") == str(root)


def test_resolve_plain_relative(tmp_path: Path) -> None:
    root = tmp_path / "books"
    (root / "sub").mkdir(parents=True)
    assert resolve_browse_path(root, "sub") == str(root / "sub")
    assert resolve_browse_path(root, "sub/") == str(root / "sub")


def test_resolve_dotdot_escapes(tmp_path: Path) -> None:
    root = tmp_path / "books"
    (root / "sub").mkdir(parents=True)
    with pytest.raises(BrowseError, match="escapes the root"):
        resolve_browse_path(root, "../etc")
    # normalizes to the root's parent even after a descent
    with pytest.raises(BrowseError, match="escapes the root"):
        resolve_browse_path(root, "sub/../..")


def test_resolve_absolute_rejected(tmp_path: Path) -> None:
    root = tmp_path / "books"
    root.mkdir()
    with pytest.raises(BrowseError, match="relative to the root"):
        resolve_browse_path(root, "/etc")


def test_resolve_null_byte_rejected(tmp_path: Path) -> None:
    root = tmp_path / "books"
    root.mkdir()
    with pytest.raises(BrowseError, match="invalid browse path"):
        resolve_browse_path(root, "sub\x00")


def test_resolve_symlink_out_of_root_escapes(tmp_path: Path) -> None:
    root = tmp_path / "books"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(BrowseError, match="escapes the root"):
        resolve_browse_path(root, "link")


def test_resolve_symlink_inside_root_ok(tmp_path: Path) -> None:
    root = tmp_path / "books"
    root.mkdir()
    (root / "a.pdf").write_bytes(b"%PDF")
    (root / "inlink").symlink_to(root / "a.pdf")
    assert resolve_browse_path(root, "inlink") == str(root / "inlink")


# --- list_browse_dir ----------------------------------------------------------


def test_listing_shape_and_filter(state_db: Database, tmp_path: Path) -> None:
    root = tmp_path / "books"
    make_root(root)
    out = list_browse_dir(state_db, root, "", FT)
    assert out["path"] == ""
    entries = out["entries"]
    assert [e["name"] for e in entries] == ["sub", "a.pdf", "B.PDF", "c.epub"]
    # dirs first, files sorted case-insensitively; .txt excluded
    sub, a, b, c = entries
    assert (sub["path"], sub["is_dir"]) == ("sub", True)
    assert sub["size_bytes"] is None and sub["rev_id"] is None
    assert (a["path"], a["is_dir"]) == ("a.pdf", False)
    assert a["size_bytes"] == len(b"%PDF-1.4 a")  # on-disk size, unindexed
    assert b["path"] == "B.PDF" and b["title"] is None
    assert c["path"] == "c.epub"


def test_listing_directory_named_like_file(state_db: Database, tmp_path: Path) -> None:
    root = tmp_path / "books"
    (root / "weird.txt").mkdir(parents=True)
    out = list_browse_dir(state_db, root, "", FT)
    assert [(e["name"], e["is_dir"]) for e in out["entries"]] == [
        ("weird.txt", True),
    ]


def test_listing_drill_down_relative_paths(state_db: Database, tmp_path: Path) -> None:
    root = tmp_path / "books"
    make_root(root)
    out = list_browse_dir(state_db, root, "sub", FT)
    assert out["path"] == "sub"
    assert [(e["name"], e["path"]) for e in out["entries"]] == [
        ("d.EPUB", "sub/d.EPUB"),
    ]


def test_listing_empty_dir(state_db: Database, tmp_path: Path) -> None:
    root = tmp_path / "books"
    (root / "empty").mkdir(parents=True)
    out = list_browse_dir(state_db, root, "empty", FT)
    assert out == {"path": "empty", "entries": []}


def test_listing_symlinks_skipped(state_db: Database, tmp_path: Path) -> None:
    root = tmp_path / "books"
    root.mkdir()
    (root / "real").mkdir()
    (root / "real" / "x.pdf").write_bytes(b"%PDF")
    (root / "f.pdf").write_bytes(b"%PDF")
    (root / "dirlink").symlink_to(root / "real")
    (root / "filelink.pdf").symlink_to(root / "f.pdf")
    out = list_browse_dir(state_db, root, "", FT)
    # only the symlinks are skipped; the real dir and file stay (dirs first)
    assert [e["name"] for e in out["entries"]] == ["real", "f.pdf"]


def test_listing_not_a_directory(state_db: Database, tmp_path: Path) -> None:
    root = tmp_path / "books"
    root.mkdir()
    (root / "a.pdf").write_bytes(b"%PDF")
    with pytest.raises(BrowseNotFound, match="not a directory"):
        list_browse_dir(state_db, root, "a.pdf", FT)
    with pytest.raises(BrowseNotFound, match="not a directory"):
        list_browse_dir(state_db, root, "does-not-exist", FT)


def test_listing_stale_alias_uses_active_revision(
    state_db: Database, tmp_path: Path
) -> None:
    """An alias path keeps its old rev_id row; the browse lookup must
    still surface the doc's ACTIVE revision (rev + catalog size)."""
    root = tmp_path / "books"
    root.mkdir()
    a = root / "A.pdf"
    b = root / "B.pdf"
    a.write_bytes(b"%PDF a")
    b.write_bytes(b"%PDF b")

    st1 = register_source(state_db, str(a), H1, 100, Format.PDF)
    st2 = register_source(state_db, str(b), H1, 100, Format.PDF)
    st3 = register_source(state_db, str(a), H2, 200, Format.PDF)
    assert st1.status is RegistrationStatus.NEW_DOCUMENT
    assert st2.status is RegistrationStatus.ALIAS
    assert st3.status is RegistrationStatus.NEW_REVISION
    doc = document_id(H1)
    r1, r2 = revision_id(doc, H1), revision_id(doc, H2)
    assert r1 != r2

    out = list_browse_dir(state_db, root, "", FT)
    by_name = {e["name"]: e for e in out["entries"]}
    # A.pdf: anchor path, trivially the active rev.
    assert by_name["A.pdf"]["rev_id"] == r2
    assert by_name["A.pdf"]["size_bytes"] == 200
    # B.pdf: alias row still points at the now-inactive r1 — the join must
    # hand back r2 with the active rev's catalog size, not 100.
    assert by_name["B.pdf"]["rev_id"] == r2
    assert by_name["B.pdf"]["size_bytes"] == 200
    assert by_name["B.pdf"]["title"] == "B"


def test_lookup_active_revisions_empty(state_db: Database) -> None:
    assert lookup_active_revisions(state_db, []) == {}


# --- HTTP routes --------------------------------------------------------------


def _client(cfg: Config, db: Database) -> TestClient:
    app = create_app(cfg, db, qdrant=FakeQdrant(cfg.embedding.dimensions))
    return TestClient(app)


@pytest.fixture
def browse_client(
    state_db: Database, base_config: Config, tmp_path: Path
) -> tuple[TestClient, Path]:
    root = tmp_path / "browsebooks"
    (root / "sub").mkdir(parents=True)
    (root / "a.pdf").write_bytes(b"%PDF")
    (root / "sub" / "b.epub").write_bytes(b"EPUB")
    base_config.browse = BrowseSettings(
        enabled=True, root=root, file_types=[".pdf", ".epub"]
    )
    return _client(base_config, state_db), root


def test_route_listing_shape(browse_client: tuple[TestClient, Path]) -> None:
    client, _ = browse_client
    res = client.get("/browse/dir")
    assert res.status_code == 200
    body: dict[str, Any] = res.json()
    assert body["path"] == ""
    assert [(e["name"], e["is_dir"]) for e in body["entries"]] == [
        ("sub", True),
        ("a.pdf", False),
    ]
    for e in body["entries"]:
        assert not Path(e["path"]).is_absolute()  # relative to the root


def test_route_drill_down(browse_client: tuple[TestClient, Path]) -> None:
    client, _ = browse_client
    res = client.get("/browse/dir", params={"path": "sub"})
    assert res.status_code == 200
    assert [e["path"] for e in res.json()["entries"]] == ["sub/b.epub"]


def test_route_containment_rejected(browse_client: tuple[TestClient, Path]) -> None:
    client, _ = browse_client
    assert client.get("/browse/dir", params={"path": "../x"}).status_code == 400
    assert client.get("/browse/dir", params={"path": "/etc"}).status_code == 400


def test_route_not_a_directory_404(browse_client: tuple[TestClient, Path]) -> None:
    client, _ = browse_client
    assert client.get("/browse/dir", params={"path": "a.pdf"}).status_code == 404
    assert client.get("/browse/dir", params={"path": "nope"}).status_code == 404


def test_ready_browse_flag_true(browse_client: tuple[TestClient, Path]) -> None:
    client, _ = browse_client
    assert client.get("/ready").json()["browse"] is True


def test_route_disabled_by_default(
    state_db: Database, base_config: Config
) -> None:
    """No browse section at all: /ready says false, /browse/* 404s."""
    client = _client(base_config, state_db)
    assert client.get("/ready").json()["browse"] is False
    assert client.get("/browse/dir").status_code == 404


def test_route_missing_root_degrades(
    state_db: Database, base_config: Config, tmp_path: Path
) -> None:
    base_config.browse = BrowseSettings(
        enabled=True, root=tmp_path / "gone-mount", file_types=FT
    )
    client = _client(base_config, state_db)
    assert client.get("/ready").json()["browse"] is False
    assert client.get("/browse/dir").status_code == 404
