"""Source reader (PRD §8B/§8D/§12): manifest, ranged original delivery,
verified unit artifacts, and the zero-based -> one-based page mapping.

Covers the M2 gate case "citations open correct physical pages/sections on
fixtures": PDF citations resolve to a PDF.js page number; EPUB citations
resolve to (section, paragraph anchor) — never a fabricated page.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fixtures import ctx_for_rev, ingest_and_register, make_epub, make_pdf
from library_rag.catalog import Format
from library_rag.config import Config
from library_rag.db import Database
from library_rag.extraction import extract_epub, extract_pdf
from library_rag.identity import unit_id_for
from library_rag.reader import active_run, create_app, pdfjs_page_for

_PAGES = ["The quick brown fox jumps over the lazy dog. " * 5, "abc", None]
_CHAPTERS = [("Intro", ["Hello world.", "Second paragraph."]), ("Body", ["Just one."])]


class Env:
    """A fully-extracted PDF + EPUB, plus a registered-but-unextracted rev."""

    def __init__(self, db: Database, cfg: Config) -> None:
        self.db = db
        self.cfg = cfg
        src = cfg.paths.source_roots[0]
        src.mkdir(parents=True)
        self.pdf_rev = ingest_and_register(db, cfg, make_pdf(src / "book.pdf", _PAGES), Format.PDF)
        extract_pdf(ctx_for_rev(db, cfg, self.pdf_rev))
        self.epub_rev = ingest_and_register(
            db, cfg, make_epub(src / "ebook.epub", _CHAPTERS), Format.EPUB
        )
        extract_epub(ctx_for_rev(db, cfg, self.epub_rev))
        self.unextracted_rev = ingest_and_register(
            db, cfg, make_pdf(src / "pending.pdf", ["not extracted yet " * 5]), Format.PDF
        )
        self.client = TestClient(create_app(cfg, db))


@pytest.fixture
def env(state_db: Database, base_config: Config) -> Env:
    return Env(state_db, base_config)


def _run_id(env: Env, rev_id: str) -> str:
    run = active_run(env.db, rev_id)
    assert run is not None
    return str(run["run_id"])


def test_pdfjs_page_mapping() -> None:
    assert pdfjs_page_for(0) == {"page": 1, "label": "1"}
    assert pdfjs_page_for(2, "iii") == {"page": 3, "label": "iii"}
    with pytest.raises(ValueError):
        pdfjs_page_for(-1)


def test_manifest_lists_units(env: Env) -> None:
    r = env.client.get(f"/books/{env.pdf_rev}")
    assert r.status_code == 200
    body = r.json()
    assert body["rev_id"] == env.pdf_rev
    assert body["format"] == "pdf"
    assert body["unit_count"] == 3
    assert body["run_id"] == _run_id(env, env.pdf_rev)
    assert [u["position"] for u in body["units"]] == [0, 1, 2]
    assert all(u["kind"] == "page" for u in body["units"])


def test_manifest_source_path_hidden_by_default(env: Env) -> None:
    # Without services.show_path_to_original the manifest must not even
    # carry the key — a deployed app never leaks server filesystem layout.
    body = env.client.get(f"/books/{env.pdf_rev}").json()
    assert "source_path" not in body


def test_manifest_source_path_when_opted_in(env: Env) -> None:
    cfg = env.cfg.model_copy(
        update={
            "services": env.cfg.services.model_copy(
                update={"show_path_to_original": True}
            )
        }
    )
    client = TestClient(create_app(cfg, env.db))
    body = client.get(f"/books/{env.pdf_rev}").json()
    row = env.db.query_one(
        "SELECT first_path FROM source_revisions WHERE rev_id = ?", (env.pdf_rev,)
    )
    assert row is not None
    assert body["source_path"] == row["first_path"]
    assert body["source_path"].endswith("book.pdf")


def test_manifest_unknown_revision_404(env: Env) -> None:
    assert env.client.get("/books/rev-does-not-exist").status_code == 404


def test_manifest_not_extracted_409(env: Env) -> None:
    assert env.client.get(f"/books/{env.unextracted_rev}").status_code == 409


def _archived_bytes(env: Env, rev_id: str) -> bytes:
    row = env.db.query_one(
        "SELECT archive_relpath FROM source_revisions WHERE rev_id = ?", (rev_id,)
    )
    assert row is not None
    return (env.cfg.paths.archive_root / str(row["archive_relpath"])).read_bytes()


def test_source_full_delivery(env: Env) -> None:
    r = env.client.get(f"/books/{env.pdf_rev}/source")
    assert r.status_code == 200
    assert r.content == _archived_bytes(env, env.pdf_rev)
    assert r.headers["accept-ranges"] == "bytes"


def test_source_range_206(env: Env) -> None:
    total = len(_archived_bytes(env, env.pdf_rev))
    r = env.client.get(f"/books/{env.pdf_rev}/source", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert len(r.content) == 100
    assert r.headers["content-range"] == f"bytes 0-99/{total}"


def test_source_range_to_end(env: Env) -> None:
    total = len(_archived_bytes(env, env.pdf_rev))
    r = env.client.get(f"/books/{env.pdf_rev}/source", headers={"Range": "bytes=10-"})
    assert r.status_code == 206
    assert len(r.content) == total - 10


def test_source_suffix_range(env: Env) -> None:
    data = _archived_bytes(env, env.pdf_rev)
    r = env.client.get(f"/books/{env.pdf_rev}/source", headers={"Range": "bytes=-5"})
    assert r.status_code == 206
    assert r.content == data[-5:]


def test_source_range_unsatisfiable(env: Env) -> None:
    total = len(_archived_bytes(env, env.pdf_rev))
    r = env.client.get(
        f"/books/{env.pdf_rev}/source", headers={"Range": f"bytes={total}-{total + 5}"}
    )
    assert r.status_code == 416
    assert r.headers["content-range"] == f"bytes */{total}"


def test_source_range_invalid(env: Env) -> None:
    r = env.client.get(f"/books/{env.pdf_rev}/source", headers={"Range": "bytes=abc"})
    assert r.status_code == 416


def test_missing_archive_410(env: Env) -> None:
    row = env.db.query_one(
        "SELECT archive_relpath FROM source_revisions WHERE rev_id = ?", (env.pdf_rev,)
    )
    assert row is not None
    (env.cfg.paths.archive_root / str(row["archive_relpath"])).unlink()
    assert env.client.get(f"/books/{env.pdf_rev}/source").status_code == 410


def test_unit_page_carrys_pdfjs_page(env: Env) -> None:
    run_id = _run_id(env, env.pdf_rev)
    unit_id = unit_id_for(run_id, "page", 0)
    r = env.client.get(f"/books/{env.pdf_rev}/units/{unit_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "page"
    assert body["position"] == 0
    assert body["pdfjs_page"] == {"page": 1, "label": "1"}
    assert body["text"].startswith("The quick brown fox")


def test_unit_unknown_404(env: Env) -> None:
    r = env.client.get(f"/books/{env.pdf_rev}/units/unit-does-not-exist")
    assert r.status_code == 404


def test_epub_section_unit_and_anchor(env: Env) -> None:
    run_id = _run_id(env, env.epub_rev)
    unit_id = unit_id_for(run_id, "section", 0)
    r = env.client.get(f"/books/{env.epub_rev}/units/{unit_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "section"
    assert body["title"] == "Intro"
    assert [p["anchor"] for p in body["paragraphs"]] == ["a0000", "a0001", "a0002"]
    # a0001 is the first real paragraph (a0000 is the h1 title).
    a = env.client.get(f"/books/{env.epub_rev}/units/{unit_id}/anchors/a0001")
    assert a.status_code == 200
    assert a.json()["text"] == "Hello world."
    assert a.json()["section"] == "Intro"
    assert env.client.get(f"/books/{env.epub_rev}/units/{unit_id}/anchors/a9999").status_code == 404


def test_anchor_endpoint_rejects_page_unit(env: Env) -> None:
    run_id = _run_id(env, env.pdf_rev)
    unit_id = unit_id_for(run_id, "page", 0)
    r = env.client.get(f"/books/{env.pdf_rev}/units/{unit_id}/anchors/a0000")
    assert r.status_code == 409
