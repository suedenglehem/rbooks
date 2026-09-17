"""EPUB extraction (PRD §8D): section units with paragraph anchors, XHTML
sanitization (XSS defense), and archive-safety limits (zip bomb / traversal).

Covers the M2 gate cases "citations open correct sections" and the
"XSS/ZIP-bomb limits tested" requirements.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from ebooklib import epub

from fixtures import ctx_for_rev, ingest_and_register, make_epub, make_epub_zip
from library_rag.catalog import Format
from library_rag.config import Config, EpubLimits
from library_rag.db import Database
from library_rag.extraction import (
    ExtractionFailure,
    check_epub_safety,
    extract_epub,
    is_permanent,
    load_unit_artifact,
)
from library_rag.extraction.sanitize import sanitize
from library_rag.identity import unit_id_for

_CHAPTERS: list[tuple[str, list[str]]] = [
    ("Intro", ["Hello world.", "Second paragraph."]),
    ("Body", ["Just one."]),
]


def _epub_rev(state_db: Database, base_config: Config) -> str:
    src = base_config.paths.scratch_root / "book.epub"
    make_epub(src, _CHAPTERS)
    return ingest_and_register(state_db, base_config, src, Format.EPUB)


def test_extract_epub_sections_and_anchors(state_db: Database, base_config: Config) -> None:
    rev_id = _epub_rev(state_db, base_config)
    ctx = ctx_for_rev(state_db, base_config, rev_id)

    assert extract_epub(ctx) == 2
    rows = state_db.query(
        """
        SELECT position, kind, ref, char_count, artifact_sha256
        FROM source_units WHERE run_id = ? ORDER BY position
        """,
        (ctx.run_id,),
    )
    assert [r["position"] for r in rows] == [0, 1]
    assert all(r["kind"] == "section" for r in rows)
    assert [r["ref"] for r in rows] == ["ch0.xhtml", "ch1.xhtml"]
    # char_count is the total of the anchored paragraph texts (the h1 title
    # counts: headings are block-level paragraphs for anchoring purposes).
    assert rows[0]["char_count"] == 5 + 12 + 17  # "Intro" + "Hello world." + "Second paragraph."

    # The artifact verifies against its stored checksum; the section title
    # lives in the artifact (the unit row carries no title column); anchors
    # are deterministic and in document order (a0000 is the h1 title).
    unit_id = unit_id_for(ctx.run_id, "section", 0)
    payload = load_unit_artifact(
        base_config.paths.artifact_root, rev_id, unit_id, rows[0]["artifact_sha256"]
    )
    assert payload["title"] == "Intro"
    assert [p["text"] for p in payload["paragraphs"]] == [
        "Intro",
        "Hello world.",
        "Second paragraph.",
    ]
    assert [p["anchor"] for p in payload["paragraphs"]] == ["a0000", "a0001", "a0002"]
    unit_id1 = unit_id_for(ctx.run_id, "section", 1)
    payload1 = load_unit_artifact(
        base_config.paths.artifact_root, rev_id, unit_id1, rows[1]["artifact_sha256"]
    )
    assert payload1["title"] == "Body"


def test_sanitize_strips_active_content() -> None:
    data = (
        b'<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        b'<p onclick="alert(1)">hi</p>'
        b"<script>alert(2)</script>"
        b'<a href="javascript:alert(3)">x</a>'
        b'<img src="http://evil.example/x.png" />'
        b'<a href="//evil.example">y</a>'
        b'<a href="#frag">ok</a>'
        b"</body></html>"
    )
    out = sanitize(data)
    assert "hi" in out and "ok" in out
    assert "script" not in out
    assert "onclick" not in out
    assert "javascript:" not in out
    assert "evil.example" not in out
    assert "#frag" in out  # in-document fragment links are kept (anchors need them)


def test_epub_safety_rejects_zip_bomb(tmp_path: Path) -> None:
    path = make_epub_zip(
        tmp_path / "bomb.epub",
        [("mimetype", b"application/epub+zip"), ("big.bin", b"\x00" * (100 * 1024))],
        deflate=frozenset({"big.bin"}),
    )
    with pytest.raises(ExtractionFailure) as ei:
        check_epub_safety(path.read_bytes(), EpubLimits(max_compression_ratio=10.0))
    assert ei.value.category == "invalid_source"
    assert is_permanent(ei.value.category)


@pytest.mark.parametrize("entry", ["../evil.txt", "/abs.txt"])
def test_epub_safety_rejects_path_traversal(tmp_path: Path, entry: str) -> None:
    path = make_epub_zip(
        tmp_path / "traversal.epub",
        [("mimetype", b"application/epub+zip"), (entry, b"x")],
    )
    with pytest.raises(ExtractionFailure) as ei:
        check_epub_safety(path.read_bytes(), EpubLimits())
    assert ei.value.category == "invalid_source"
    assert "traversal" in ei.value.detail or "absolute" in ei.value.detail


def test_epub_safety_rejects_too_many_entries(tmp_path: Path) -> None:
    path = make_epub_zip(
        tmp_path / "many.epub",
        [("e0", b"a"), ("e1", b"b"), ("e2", b"c")],
    )
    with pytest.raises(ExtractionFailure) as ei:
        check_epub_safety(path.read_bytes(), EpubLimits(max_entries=2))
    assert ei.value.category == "invalid_source"


def test_epub_safety_rejects_too_large(tmp_path: Path) -> None:
    path = make_epub_zip(
        tmp_path / "large.epub",
        [("mimetype", b"application/epub+zip"), ("data.bin", b"x" * 100)],
    )
    with pytest.raises(ExtractionFailure) as ei:
        check_epub_safety(path.read_bytes(), EpubLimits(max_uncompressed_bytes=10))
    assert ei.value.category == "invalid_source"


def test_epub_safety_rejects_not_a_zip() -> None:
    with pytest.raises(ExtractionFailure) as ei:
        check_epub_safety(b"garbage", EpubLimits())
    assert ei.value.category == "invalid_source"


def _write_epub_with_raw_chapter(path: Path, good: str, bad: str) -> Path:
    """A two-chapter EPUB whose second chapter holds verbatim (malformed) bytes.

    ebooklib re-serializes chapter content into well-formed XHTML on write,
    so the malformed bytes are patched back in as a raw ZIP entry afterwards.
    """
    book = epub.EpubBook()
    book.set_identifier(path.stem)
    book.set_title("Bad Book")
    book.set_language("en")
    c0 = epub.EpubHtml(uid="c0", file_name="ch0.xhtml", title="Good", lang="en")
    c0.content = "<html><body><h1>Good</h1><p>Fine paragraph.</p></body></html>"
    c1 = epub.EpubHtml(uid="c1", file_name="ch1.xhtml", title="Broken", lang="en")
    c1.content = bad
    book.add_item(c0)
    book.add_item(c1)
    book.spine.append("c0")
    book.spine.append("c1")
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book, {})
    # Rebuild the archive with ch1.xhtml holding the raw bytes ebooklib
    # would have normalized away.
    entries: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(path) as zf:
        entries = [(i.filename, zf.read(i.filename)) for i in zf.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            # ebooklib prefixes entries with EPUB/, so match by suffix.
            zf.writestr(name, bad.encode("utf-8") if name.endswith("ch1.xhtml") else data)
    return path


def test_extract_epub_bad_html_flagged(state_db: Database, base_config: Config) -> None:
    src = base_config.paths.scratch_root / "bad.epub"
    _write_epub_with_raw_chapter(src, "Good", "<html><body><p>unclosed")
    rev_id = ingest_and_register(state_db, base_config, src, Format.EPUB)
    ctx = ctx_for_rev(state_db, base_config, rev_id)

    assert extract_epub(ctx) == 2  # run still succeeds
    rows = {
        int(r["position"]): r
        for r in state_db.query(
            "SELECT position, quality_flags FROM source_units WHERE run_id = ? ORDER BY position",
            (ctx.run_id,),
        )
    }
    assert rows[0]["quality_flags"] is None
    assert set(json.loads(rows[1]["quality_flags"])) == {"bad_html", "no_text"}
    # The good chapter is unaffected.
    unit_id = unit_id_for(ctx.run_id, "section", 0)
    stored = state_db.query_one(
        "SELECT artifact_sha256 FROM source_units WHERE unit_id = ?", (unit_id,)
    )
    assert stored is not None
    payload = load_unit_artifact(
        base_config.paths.artifact_root, rev_id, unit_id, stored["artifact_sha256"]
    )
    assert [p["text"] for p in payload["paragraphs"]] == ["Good", "Fine paragraph."]


def test_extract_epub_missing_source(state_db: Database, base_config: Config) -> None:
    rev_id = _epub_rev(state_db, base_config)
    ctx = ctx_for_rev(state_db, base_config, rev_id)
    ctx.source_path.unlink()
    with pytest.raises(ExtractionFailure) as ei:
        extract_epub(ctx)
    assert ei.value.category == "missing_source"
    assert is_permanent(ei.value.category)
