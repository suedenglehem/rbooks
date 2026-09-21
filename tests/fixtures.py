"""Test source builders: real PDFs and EPUBs for the pipeline.

These produce *structurally real* files (PyMuPDF PDFs, ebooklib EPUBs) so the
pipeline is exercised exactly as in production, plus a raw-ZIP builder for the
malicious-archive (zip-bomb / traversal) cases that ebooklib would never emit.

For M3 the host has no Tesseract (and installing it needs operator approval,
PRD §1), so ``make_fake_tesseract`` writes a deterministic CLI shim that speaks
the same interface (``--list-langs``, ``--version``,
``<image> stdout -l <langs> --psm <n> tsv``) and emits a known word grid per
page — good enough to test routing, coordinate mapping, retries, and resume.
"""

from __future__ import annotations

import json
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
from library_rag.embeddings import (
    Embedder,
    FakeEmbedder,
    checkpoint_path,
    embedding_sha,
    tokenize,
    write_checkpoint,
)
from library_rag.extraction import ExtractorCtx
from library_rag.identity import normalize_path
from library_rag.indexing import QdrantOps, publish_generation
from library_rag.jobs import Jobs
from library_rag.scan import scan_roots
from library_rag.worker import build_ctx, chunk_fingerprint_for_run, run_worker

__all__ = [
    "ctx_for_rev",
    "ingest_and_publish",
    "ingest_and_register",
    "make_cropped_pdf",
    "make_encrypted_pdf",
    "make_epub",
    "make_epub_zip",
    "make_fake_tesseract",
    "make_mixed_pdf",
    "make_pdf",
    "make_rotated_pdf",
    "publish_handbuilt",
]

# A4, the same geometry the smoke test used; text baseline kept clear of edges.
_PAGE_W, _PAGE_H = 595, 842
_SAMPLE_SENTENCE = "The quick brown fox jumps over the lazy dog."


def _fill_page(page: Any, kind: str) -> None:
    """Draw one page in the requested shape (see :func:`make_mixed_pdf`)."""
    if kind == "text":
        page.insert_text((72, 100), _SAMPLE_SENTENCE, fontsize=11)
    elif kind == "sparse":
        page.insert_text((72, 100), "End", fontsize=11)
    elif kind == "sparse_image":
        page.insert_text((72, 100), "End", fontsize=11)
        _insert_gray_image(page)
    elif kind == "scanned":
        _insert_gray_image(page)


def _insert_gray_image(page: Any) -> None:
    # A solid gray image over the whole page: get_image_info reports it with
    # bbox == page.rect, so image_area_ratio is 1.0 (a "scanned" page).
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 200, 200), 0)  # type: ignore[no-untyped-call]
    pix.clear_with(128)  # type: ignore[no-untyped-call]
    page.insert_image(page.rect, pixmap=pix)


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


_MIXED_KINDS = ("text", "scanned", "blank", "sparse", "sparse_image")


def make_mixed_pdf(path: Path, kinds: list[str]) -> Path:
    """One A4 page per *kind*: text / scanned (image only) / blank / sparse /
    sparse_image — the page shapes that exercise every OCR route (PRD §8C)."""
    unknown = sorted(set(kinds) - set(_MIXED_KINDS))
    if unknown:
        raise ValueError(f"unknown page kinds: {unknown}")
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        for kind in kinds:
            page = doc.new_page(width=_PAGE_W, height=_PAGE_H)
            _fill_page(page, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(path))  # type: ignore[no-untyped-call]
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    return path


def make_rotated_pdf(path: Path, kind: str = "scanned", rotation: int = 90) -> Path:
    """A single page rotated by *rotation* degrees (the OCR highlight-mapping
    gate case: raster boxes must map back to the rotation-aware page rect)."""
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        page = doc.new_page(width=_PAGE_W, height=_PAGE_H)
        _fill_page(page, kind)
        page.set_rotation(rotation)  # type: ignore[no-untyped-call]
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(path))  # type: ignore[no-untyped-call]
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    return path


def make_cropped_pdf(path: Path, kind: str = "text") -> Path:
    """A single page whose cropbox is smaller than the media box (the render
    size must follow the cropbox, not the media box)."""
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        page = doc.new_page(width=_PAGE_W, height=_PAGE_H)
        _fill_page(page, kind)
        page.set_cropbox(pymupdf.Rect(50, 50, 500, 500))  # type: ignore[no-untyped-call]
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(path))  # type: ignore[no-untyped-call]
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    return path


# The shim is a real executable Python script (tesseract is a CLI, so the fake
# must be one too). Its behavior is steered by environment variables the tests
# set per case: FAKE_TESSERACT_LANGS (installed packs), FAKE_TESSERACT_FAIL
# (non-zero exit), FAKE_TESSERACT_SLEEP (seconds), FAKE_TESSERACT_LOG (append a
# JSON line per invocation — the kill/resume and no-op tests count calls on it).
_FAKE_TESSERACT = """\
#!/usr/bin/env python3
import json, os, re, sys, time

argv = sys.argv[1:]
if "--list-langs" in argv:
    sys.stdout.write(os.environ.get("FAKE_TESSERACT_LANGS", "eng\\nosd") + "\\n")
    raise SystemExit(0)
if "--version" in argv:
    sys.stderr.write("tesseract 5.3.0-fake\\n")
    raise SystemExit(0)
if os.environ.get("FAKE_TESSERACT_FAIL") == "1":
    sys.stderr.write("fake tesseract: forced failure\\n")
    raise SystemExit(1)
sleep_s = float(os.environ.get("FAKE_TESSERACT_SLEEP", "0") or 0)
if sleep_s > 0:
    time.sleep(sleep_s)
image = next((a for a in argv if a.endswith(".png")), "")
m = re.search(r"page-(\\d+)\\.png$", os.path.basename(image))
page = int(m.group(1)) if m else 0
log = os.environ.get("FAKE_TESSERACT_LOG")
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"image": image, "page": page, "args": argv}) + "\\n")
rows = ["level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num"
        "\\tleft\\ttop\\twidth\\theight\\tconf\\ttext"]
rows.append("1\\t1\\t1\\t0\\t0\\t0\\t0\\t0\\t100\\t100\\t0\\t")
# Three words on distinct lines; the raster boxes are what the mapping tests
# map back to page coordinates.
for j in range(3):
    rows.append("5\\t1\\t1\\t1\\t%d\\t%d\\t%d\\t100\\t200\\t50\\t%.1f\\tw%d-%d"
                % (j + 1, j + 1, 50 + j * 220, 85.0 + j, page, j))
sys.stdout.write("\\n".join(rows) + "\\n")
"""


def make_fake_tesseract(path: Path) -> Path:
    """Write the fake Tesseract executable to *path* (mode 0o755)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_FAKE_TESSERACT, encoding="utf-8")
    path.chmod(0o755)
    return path


# --- M4 pipeline drivers ----------------------------------------------------------


def ingest_and_publish(
    db: Database,
    cfg: Config,
    src: Path,
    *,
    qdrant: QdrantOps,
    embedder: Embedder | None = None,
) -> tuple[str, str]:
    """Scan *src* into the queue and drain the pipeline to an active publication.

    Drives the real scan + worker loop (the M4 end-to-end path); callers
    configure ``cfg.embedding`` (e.g. ``fake = True``) beforehand. Returns
    ``(rev_id, pub_id)`` and asserts exactly one publication is active.
    """
    scan_roots(db, cfg, Jobs(db))
    run_worker(db, cfg, once=True, poll_delay=0, qdrant=qdrant, embedder=embedder)
    pubs = db.query("SELECT pub_id, rev_id FROM publications WHERE state = 'active'")
    assert len(pubs) == 1, f"expected exactly one active publication, got {pubs}"
    return pubs[0]["rev_id"], pubs[0]["pub_id"]


def publish_handbuilt(
    db: Database,
    cfg: Config,
    qdrant: QdrantOps,
    *,
    doc_id: str,
    rev_id: str,
    run_id: str,
    texts: list[str],
    title: str | None = None,
    pub_state: str = "active",
    rev_active: int = 1,
) -> tuple[str, str, list[str]]:
    """Register one document/revision/chunk set *without* the pipeline, then publish it.

    The retrieval tests need valid ``chunks`` rows plus real checkpoint
    artifacts, not archives or extraction; this inserts the catalog/run/chunk
    rows and writes genuine ``batch_*.bin`` checkpoints (the worker's exact
    conventions), then runs the real :func:`publish_generation`. Returns
    ``(gen_id, pub_id, chunk_ids)``.

    *pub_state*/*rev_active* rewind the committed state afterwards, so a test
    can simulate a crash between publication boundaries.
    """
    assert cfg.embedding.is_configured, "configure cfg.embedding before publish_handbuilt"
    emb_sha = embedding_sha(cfg)
    emb = FakeEmbedder(dimensions=cfg.embedding.dimensions)
    ts = 0.0
    # ON CONFLICT: the supersede tests publish a second revision of the same
    # document (B4 only supersedes a doc's *other* active publications, so the
    # doc row must stay the first generation's).
    db.execute(
        "INSERT INTO documents (doc_id, anchor_sha256, created_at, updated_at) "
        "VALUES (?, ?, ?, ?) ON CONFLICT(doc_id) DO NOTHING",
        (doc_id, "a" * 64, ts, ts),
    )
    db.execute(
        """
        INSERT INTO source_revisions (
            rev_id, doc_id, sha256, size_bytes, format, archive_relpath, first_path,
            is_active, created_at
        ) VALUES (?, ?, ?, ?, 'pdf', ?, ?, ?, ?)
        """,
        (rev_id, doc_id, "b" * 64, len("\n".join(texts)),
         f"{rev_id}.pdf", f"/books/{rev_id}.pdf", rev_active, ts),
    )
    db.execute(
        """
        INSERT INTO extraction_runs (
            run_id, rev_id, doc_id, parser_version, settings_sha, unit_count, state,
            created_at, updated_at
        ) VALUES (?, ?, ?, 'pdf/1', 's1', ?, 'succeeded', ?, ?)
        """,
        (run_id, rev_id, doc_id, len(texts), ts, ts),
    )
    # Stamp the chunk fingerprint exactly as the worker's chunk stage would:
    # a run whose stored fingerprint is NULL (or stale) is a re-chunk
    # candidate to the worker's reconcile pass and to `migrate` alike.
    db.execute(
        "UPDATE extraction_runs SET chunk_fingerprint = ?, updated_at = ? WHERE run_id = ?",
        (chunk_fingerprint_for_run(db, run_id, cfg), ts, run_id),
    )
    chunk_ids = [f"{run_id}:chunk-{i}" for i in range(len(texts))]
    for position, (cid, text) in enumerate(zip(chunk_ids, texts, strict=True)):
        db.execute(
            """
            INSERT INTO chunks (
                chunk_id, run_id, rev_id, position, text, token_count, title, spans,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, '[]', ?)
            """,
            (cid, run_id, rev_id, position, text, len(tokenize(text)), title, ts),
        )
    batch_size = max(1, cfg.embedding.batch_size)
    for i in range(0, len(texts), batch_size):
        batch_index = i // batch_size
        batch = texts[i : i + batch_size]
        cids = chunk_ids[i : i + batch_size]
        cp = checkpoint_path(cfg.paths.artifact_root, run_id, emb_sha, batch_index)
        vector_sha = write_checkpoint(
            cp,
            emb.encode_documents(batch),
            emb.dimensions,
            {
                "run_id": run_id,
                "embedding_sha": emb_sha,
                "model_revision": emb.model_revision,
                "batch_index": batch_index,
                "chunk_ids": cids,
            },
        )
        db.execute(
            """
            INSERT INTO embedding_batches (
                batch_id, run_id, model_revision, embedding_sha, batch_index,
                chunk_ids, vector_sha256, artifact_relpath, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{run_id}:{emb_sha}:{batch_index:05d}",
                run_id,
                emb.model_revision,
                emb_sha,
                batch_index,
                json.dumps(cids),
                vector_sha,
                cp.relative_to(cfg.paths.artifact_root).as_posix(),
                ts,
            ),
        )
    publish_generation(db, cfg, qdrant, rev_id=rev_id, run_id=run_id, on_progress=lambda: None)
    row = db.query_one(
        "SELECT pub_id, gen_id FROM publications WHERE rev_id = ? AND state = 'active'",
        (rev_id,),
    )
    assert row is not None, f"publish_generation did not activate a publication for {rev_id}"
    if pub_state != "active":
        db.execute(
            "UPDATE publications SET state = ?, activated_at = NULL WHERE pub_id = ?",
            (pub_state, row["pub_id"]),
        )
    return row["gen_id"], row["pub_id"], chunk_ids
