"""Test source builders: real PDFs and EPUBs for the M2 pipeline.

These produce *structurally real* files (PyMuPDF PDFs, ebooklib EPUBs) so the
pipeline is exercised exactly as in production, plus a raw-ZIP builder for the
malicious-archive (zip-bomb / traversal) cases that ebooklib would never emit.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymupdf
from ebooklib import epub

from library_rag.archive import ingest_source
from library_rag.catalog import Format, RegistrationStatus, register_source
from library_rag.config import Config
from library_rag.db import Database
from library_rag.extraction import ExtractorCtx
from library_rag.identity import normalize_path
from library_rag.worker import build_ctx

__all__ = [
    "ctx_for_rev",
    "ingest_and_register",
    "make_encrypted_pdf",
    "make_epub",
    "make_epub_zip",
    "make_pdf",
]

# A4, the same geometry the smoke test used; text baseline kept clear of edges.
_PAGE_W, _PAGE_H = 595, 842


def ingest_and_register(db: Database, cfg: Config, src: Path, fmt: Format) -> str:
    """Archive *src* and register it in the catalog; return the revision ID.

    Shared by the extraction/worker/reader tests so every one of them drives
    the *real* ingest->register path rather than fabricating catalog rows.
    """
    ing = ingest_source(cfg.paths.archive_root, src)
    reg = register_source(db, normalize_path(src), ing.sha256, ing.size_bytes, fmt)
    assert reg.status in (RegistrationStatus.NEW_DOCUMENT, RegistrationStatus.NEW_REVISION)
    return reg.rev_id


def ctx_for_rev(
    db: Database, cfg: Config, rev_id: str, on_progress: Callable[[], None] | None = None
) -> ExtractorCtx:
    """Build the extractor context for *rev_id* (deterministic run ID, PRD §6)."""
    row = db.query_one("SELECT * FROM source_revisions WHERE rev_id = ?", (rev_id,))
    assert row is not None
    rev: dict[str, Any] = dict(row)
    return build_ctx(db, cfg, rev, on_progress)


def _build_pdf(doc: Any, pages: list[str | None]) -> None:
    for text in pages:
        page = doc.new_page(width=_PAGE_W, height=_PAGE_H)
        if text:
            page.insert_text((72, 100), text, fontsize=11)


def make_pdf(path: Path, pages: list[str | None]) -> Path:
    """Write a PDF with one page per entry (None = a text-less page)."""
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        _build_pdf(doc, pages)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(path))  # type: ignore[no-untyped-call]
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    return path


def make_encrypted_pdf(path: Path, pages: list[str | None], password: str = "secret") -> Path:
    """Write an AES-256 user-password-protected PDF (the "encrypted" fixture)."""
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        _build_pdf(doc, pages)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(  # type: ignore[no-untyped-call]
            str(path),
            encryption=pymupdf.PDF_ENCRYPT_AES_256,  # type: ignore[attr-defined]
            user_pw=password,
            owner_pw=password,
        )
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    return path


def make_epub(
    path: Path, chapters: list[tuple[str, list[str]]], title: str = "Test Book"
) -> Path:
    """Write an EPUB: one spine item per ``(chapter_title, paragraphs)``.

    Each chapter's body is ``<h1>title</h1>`` followed by one ``<p>`` per
    paragraph, matching the shape the sanitizer/anchoring code expects.
    """
    book = epub.EpubBook()
    book.set_identifier(path.stem)
    book.set_title(title)
    book.set_language("en")
    for i, (ch_title, paras) in enumerate(chapters):
        ch = epub.EpubHtml(uid=f"c{i}", file_name=f"ch{i}.xhtml", title=ch_title, lang="en")
        ch.content = (
            f"<html><body><h1>{ch_title}</h1>"
            + "".join(f"<p>{p}</p>" for p in paras)
            + "</body></html>"
        )
        book.add_item(ch)
        book.spine.append(f"c{i}")
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book, {})
    return path


def make_epub_zip(
    path: Path, entries: list[tuple[str, bytes]], deflate: frozenset[str] = frozenset()
) -> Path:
    """Write a raw ZIP with caller-controlled entry names and compression.

    *entries* is ``(name, data)`` in write order (mimetype conventionally
    first). Entry names are passed through verbatim, so ``../`` and absolute
    paths can be injected for the path-traversal defenses. Entries named in
    *deflate* are compressed (needed to build a compression-bomb ratio); all
    others are stored, which keeps sizes predictable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries:
            compress = zipfile.ZIP_DEFLATED if name in deflate else zipfile.ZIP_STORED
            zf.writestr(name, data, compress_type=compress)
    return path
