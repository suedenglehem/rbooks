"""EPUB extraction via ebooklib (PRD §8D).

One source unit per spine item, in spine order (the *position* is the zero-based
spine index). Before ebooklib ever touches the bytes, the archive is screened:

* entry count, total uncompressed size, and compression ratio bounds
  (zip-bomb defense),
* absolute paths and ``..`` segments (path-traversal defense),

and each section's XHTML is sanitized (:mod:`library_rag.extraction.sanitize`)
before any text is kept: no scripts, no event handlers, no external URIs.

Paragraphs carry deterministic anchors (``a0000``, ``a0001``, ...) in document
order, so a citation is *(unit, anchor)* — a chapter/paragraph reference that
needs no page numbers, which EPUBs do not have (PRD §8D "never fabricated
page numbers").
"""

from __future__ import annotations

import importlib.metadata
import io
import xml.etree.ElementTree as ET
import zipfile
from typing import Any

from ebooklib import epub

from ..config import EpubLimits
from ..identity import unit_id_for
from .errors import ExtractionFailure
from .sanitize import parse_blocks, sanitize
from .store import (
    ExtractorCtx,
    commit_unit_artifact,
    existing_verified_unit,
    finish_run,
    insert_unit,
    start_run,
)

__all__ = ["check_epub_safety", "extract_epub", "parser_version"]

# Heartbeat every N sections so the worker's lease stays fresh on big books.
_SECTIONS_PER_HEARTBEAT = 20

_XHTML_MEDIA_TYPE = "application/xhtml+xml"


def parser_version() -> str:
    """Parser identity for the extraction key (changes => fresh run)."""
    return f"ebooklib-{importlib.metadata.version('ebooklib')}"


def check_epub_safety(data: bytes, limits: EpubLimits) -> None:
    """Reject path-traversal and zip-bomb archives before parsing (PRD §8D).

    Raises :class:`ExtractionFailure` with a permanent category for anything
    structurally unsafe: retrying the same bytes cannot make it safe.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ExtractionFailure("invalid_source", f"not a readable ZIP/EPUB archive: {exc}") from exc
    with zf:
        entries = zf.infolist()
        if len(entries) > limits.max_entries:
            raise ExtractionFailure(
                "invalid_source", f"too many archive entries: {len(entries)} > {limits.max_entries}"
            )
        total = sum(e.file_size for e in entries)
        if total > limits.max_uncompressed_bytes:
            raise ExtractionFailure(
                "invalid_source",
                f"uncompressed size {total} bytes exceeds limit {limits.max_uncompressed_bytes}",
            )
        compressed = sum(e.compress_size for e in entries)
        if compressed > 0 and total / compressed > limits.max_compression_ratio:
            raise ExtractionFailure(
                "invalid_source",
                f"compression ratio {total / compressed:.0f}x exceeds limit "
                f"{limits.max_compression_ratio}x (zip bomb)",
            )
        for e in entries:
            if e.filename.startswith("/"):
                raise ExtractionFailure("invalid_source", f"absolute archive path: {e.filename}")
            if any(part == ".." for part in e.filename.split("/")):
                raise ExtractionFailure(
                    "invalid_source", f"path traversal in archive entry: {e.filename}"
                )


def _spine_items(book: Any) -> list[Any]:
    """Spine items in spine order (defensive over ebooklib's spine shape)."""
    spine = book.spine if isinstance(book.spine, list) else []
    items: list[Any] = []
    for ref in spine:
        idref = ref[0] if isinstance(ref, (tuple, list)) else ref
        item = book.get_item_with_id(idref)
        if item is not None:
            items.append(item)
    return items


def _section_payload(
    ctx: ExtractorCtx, position: int, item: Any
) -> dict[str, Any]:
    """Build one spine item's unit payload: sanitized, anchored paragraphs."""
    # Raw archive bytes: item.get_content() re-parses and re-serializes the
    # chapter (lxml repairs malformed markup), which would defeat the
    # sanitizer's strict-parse bad_html classification (PRD §8D).
    content = item.content or b""
    flags: list[str] = []
    sanitized = ""
    paragraphs: list[str] = []
    title: str | None = None
    if content:
        try:
            sanitized = sanitize(content)
            paragraphs, title = parse_blocks(sanitized)
        except ET.ParseError:
            flags.append("bad_html")  # untrusted, unparseable markup: keep nothing
    if not paragraphs:
        flags.append("no_text")
    anchored = [
        {"anchor": f"a{ord:04d}", "text": text} for ord, text in enumerate(paragraphs)
    ]
    return {
        "kind": "section",
        "rev_id": ctx.rev["rev_id"],
        "run_id": ctx.run_id,
        "position": position,
        "ref": item.file_name,
        "title": title,
        "paragraphs": anchored,
        "sanitized_html": sanitized,
        "char_count": sum(len(p["text"]) for p in anchored),
        "quality": {"flags": flags},
    }


def extract_epub(ctx: ExtractorCtx) -> int:
    """Extract every spine section of the archived EPUB; return the unit count.

    Raises ExtractionFailure with a permanent category for unsafe, corrupt, or
    missing sources; transient errors propagate to the worker.
    """
    path = ctx.source_path
    if not path.is_file():
        raise ExtractionFailure("missing_source", f"archived original missing: {path}")
    data = path.read_bytes()
    check_epub_safety(data, ctx.settings.epub)
    try:
        book = epub.read_epub(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionFailure("corrupt", f"cannot parse EPUB: {exc}") from exc
    if book is None:
        raise ExtractionFailure("corrupt", "ebooklib returned no book")
    items = _spine_items(book)
    if not items:
        raise ExtractionFailure("invalid_source", "EPUB has no spine items")

    if not start_run(ctx, parser_version(), ctx.settings.settings_sha()):
        if ctx.on_extract_done is not None:
            ctx.on_extract_done()
        row = ctx.db.query_one(
            "SELECT unit_count FROM extraction_runs WHERE run_id = ?", (ctx.run_id,)
        )
        return int(row["unit_count"] or 0) if row is not None else 0

    done = 0
    for position, item in enumerate(items):
        if item.media_type != _XHTML_MEDIA_TYPE:
            continue  # media overlays etc. carry no text units
        unit_id = unit_id_for(ctx.run_id, "section", position)
        if existing_verified_unit(ctx, "section", position, unit_id):
            done += 1  # completed by a previous (crashed) run
            continue
        payload = _section_payload(ctx, position, item)
        sha = commit_unit_artifact(ctx, unit_id, payload)
        flags = payload["quality"]["flags"]
        insert_unit(
            ctx,
            unit_id=unit_id,
            kind="section",
            position=position,
            ref=payload["ref"],
            char_count=payload["char_count"],
            rotation=None,
            width=None,
            height=None,
            quality_flags=flags or None,
            artifact_relpath=f"extract/{ctx.rev['rev_id']}/{unit_id}.json.gz",
            artifact_sha256=sha,
        )
        done += 1
        if ctx.on_progress is not None and position % _SECTIONS_PER_HEARTBEAT == _SECTIONS_PER_HEARTBEAT - 1:
            ctx.on_progress()
    finish_run(ctx, done)
    if ctx.on_extract_done is not None:
        ctx.on_extract_done()
    return done
