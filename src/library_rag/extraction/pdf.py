"""PDF extraction via PyMuPDF (PRD §8B).

One source unit per physical page. Text, geometry, and per-span metadata are
written to a compressed JSON artifact (committed *before* the database row, so
a crash never leaves a row pointing at a missing artifact). Quality flags are
recorded now and routed in M3 (OCR selection):

* ``no_text`` — page has no extractable text (scanned image; OCR candidate).
* ``sparse`` — fewer non-whitespace characters than ``pdf.sparse_chars``.
* ``many_replacement`` — share of U+FFFD above ``pdf.max_replacement_ratio``
  (a broken/mojibake text layer; OCR candidate).
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

import pymupdf

from ..identity import unit_id_for
from .errors import ExtractionFailure
from .store import (
    ExtractorCtx,
    commit_unit_artifact,
    existing_verified_unit,
    finish_run,
    insert_unit,
    start_run,
)

__all__ = ["assess_page", "extract_pdf", "parser_version"]

# Heartbeat every N pages so the worker's lease stays fresh on big books.
_PAGES_PER_HEARTBEAT = 10


def parser_version() -> str:
    """Parser identity for the extraction key (changes => fresh run)."""
    return f"pymupdf-{importlib.metadata.version('pymupdf')}"


def assess_page(text: str, sparse_chars: int, max_replacement_ratio: float) -> list[str]:
    """Return the quality flags for a page's extracted text (PRD §8B)."""
    flags: list[str] = []
    n_chars = len("".join(text.split()))  # non-whitespace count
    if n_chars == 0:
        flags.append("no_text")
    elif n_chars < sparse_chars:
        flags.append("sparse")
    if text and (text.count("�") / len(text)) > max_replacement_ratio:
        flags.append("many_replacement")
    return flags


def _page_payload(ctx: ExtractorCtx, index: int, page: Any) -> dict[str, Any]:
    text = page.get_text("text")
    blocks: list[dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") == 0:  # text block
            spans = [
                {
                    "text": span.get("text", ""),
                    "bbox": list(span.get("bbox", [0, 0, 0, 0])),
                    "font": span.get("font", ""),
                    "size": span.get("size"),
                }
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ]
            blocks.append({"bbox": list(block.get("bbox", [0, 0, 0, 0])), "spans": spans})
        else:  # image / drawing block: geometry only
            blocks.append({"bbox": list(block.get("bbox", [0, 0, 0, 0])), "spans": []})

    try:
        label = page.get_label()
        if not isinstance(label, str) or not label:
            raise ValueError("no label")
    except Exception:
        label = str(index + 1)

    flags = assess_page(text, ctx.settings.pdf.sparse_chars, ctx.settings.pdf.max_replacement_ratio)
    return {
        "kind": "page",
        "rev_id": ctx.rev["rev_id"],
        "run_id": ctx.run_id,
        "position": index,
        "label": label,
        "rotation": int(page.rotation),
        "width": float(page.rect.width),
        "height": float(page.rect.height),
        "text": text,
        "blocks": blocks,
        "quality": {"chars": len("".join(text.split())), "flags": flags},
    }


def extract_pdf(ctx: ExtractorCtx) -> int:
    """Extract every page of the archived PDF; return the unit count.

    Raises ExtractionFailure with a permanent category for encrypted, corrupt,
    or missing sources; transient errors propagate to the worker.
    """
    path = ctx.source_path
    if not path.is_file():
        raise ExtractionFailure("missing_source", f"archived original missing: {path}")
    try:
        # pymupdf's stubs type the open factory as returning the untyped
        # Document constructor; the call is validated by the tests.
        doc = pymupdf.open(str(path))  # type: ignore[no-untyped-call]
    except Exception as exc:  # FileDataError et al. — unreadable structure
        raise ExtractionFailure("corrupt", f"cannot open PDF: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise ExtractionFailure("encrypted", "PDF requires a password")
        if not start_run(ctx, parser_version(), ctx.settings.settings_sha()):
            row = ctx.db.query_one(
                "SELECT unit_count FROM extraction_runs WHERE run_id = ?", (ctx.run_id,)
            )
            return int(row["unit_count"] or 0) if row is not None else 0

        done = 0
        for index in range(doc.page_count):
            unit_id = unit_id_for(ctx.run_id, "page", index)
            if existing_verified_unit(ctx, "page", index, unit_id):
                done += 1  # completed by a previous (crashed) run
                continue
            payload = _page_payload(ctx, index, doc[index])
            sha = commit_unit_artifact(ctx, unit_id, payload)
            flags = payload["quality"]["flags"]
            insert_unit(
                ctx,
                unit_id=unit_id,
                kind="page",
                position=index,
                ref=payload["label"],
                char_count=payload["quality"]["chars"],
                rotation=payload["rotation"],
                width=payload["width"],
                height=payload["height"],
                quality_flags=flags or None,
                artifact_relpath=f"extract/{ctx.rev['rev_id']}/{unit_id}.json.gz",
                artifact_sha256=sha,
            )
            done += 1
            if ctx.on_progress is not None and index % _PAGES_PER_HEARTBEAT == _PAGES_PER_HEARTBEAT - 1:
                ctx.on_progress()
        finish_run(ctx, done)
        return done
