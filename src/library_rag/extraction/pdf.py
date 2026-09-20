"""PDF extraction via PyMuPDF (PRD §8B).

One source unit per physical page. Text, geometry, and per-span metadata are
written to a compressed JSON artifact (committed *before* the database row, so
a crash never leaves a row pointing at a missing artifact). Quality flags
drive the M3 OCR route (PRD §8C) recorded on the unit row:

* ``no_text`` — page has no extractable text (scanned image; OCR candidate).
* ``sparse`` — fewer non-whitespace characters than ``pdf.sparse_chars``.
* ``many_replacement`` — share of U+FFFD above ``pdf.max_replacement_ratio``
  (a broken/mojibake text layer; OCR candidate).

Routing decides ``reuse`` (native text is fine), ``ocr`` (enqueue a durable
OCR job; the page's ``ocr_state`` becomes ``pending``), or ``skip`` (blank /
illustration page — no OCR job). Because the enqueue is a durable, idempotent
job row, extraction completion also re-enqueues any ``route='ocr'`` units
whose job row might have been lost (crash between the insert and the
enqueue), closing that window on replay.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

import pymupdf

from ..identity import unit_id_for
from .errors import ExtractionFailure
from .routing import ROUTE_OCR, image_area_ratio, route_page
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
    stripped = text.strip()
    if not stripped:
        flags.append("no_text")
    elif len(stripped) < sparse_chars:
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


def _requeue_ocr_units(ctx: ExtractorCtx) -> None:
    """Idempotently re-enqueue the OCR jobs of every page unit routed to OCR.

    Closes the two lost-enqueue windows: a crash between ``insert_unit`` and
    ``enqueue_ocr`` on a fresh extract, and a resume where a previous run
    inserted a routed row but was killed before enqueuing. Enqueue is
    INSERT OR IGNORE on the task key, so this is a no-op for healthy jobs.
    """
    if ctx.enqueue_ocr is None:
        return
    rows = ctx.db.query(
        "SELECT position FROM source_units WHERE run_id = ? AND route = ?",
        (ctx.run_id, ROUTE_OCR),
    )
    for row in rows:
        ctx.enqueue_ocr(int(row["position"]))


def extract_pdf(ctx: ExtractorCtx) -> int:
    """Extract every page of the archived PDF (or the first ``ctx.page_cap``
    pages when a pilot cap is set); return the unit count.

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
        if start_run(ctx, parser_version(), ctx.settings.settings_sha()):
            done = 0
            # M6 pilot: a page cap bounds the workload per book (PRD §12). The
            # cap is part of the run's extraction key, so a capped run never
            # shares state with the full run of the same bytes.
            limit = (
                min(doc.page_count, ctx.page_cap) if ctx.page_cap is not None else doc.page_count
            )
            for index in range(limit):
                unit_id = unit_id_for(ctx.run_id, "page", index)
                if existing_verified_unit(ctx, "page", index, unit_id):
                    done += 1  # completed by a previous (crashed) run
                    continue
                payload = _page_payload(ctx, index, doc[index])
                sha = commit_unit_artifact(ctx, unit_id, payload)
                flags = payload["quality"]["flags"]
                # Selective OCR route (PRD §8C): a missing or broken text
                # layer with significant image coverage gets an OCR job.
                route = route_page(flags, image_area_ratio(doc[index]), ctx.settings.ocr)
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
                    route=route,
                    ocr_state="pending" if route == ROUTE_OCR else "none",
                )
                if route == ROUTE_OCR and ctx.enqueue_ocr is not None:
                    ctx.enqueue_ocr(index)
                done += 1
                if ctx.on_progress is not None and index % _PAGES_PER_HEARTBEAT == _PAGES_PER_HEARTBEAT - 1:
                    ctx.on_progress()
            finish_run(ctx, done)
        # Run complete (or replay of an already-succeeded run): make sure the
        # downstream jobs exist. Both enqueues are idempotent no-ops otherwise.
        _requeue_ocr_units(ctx)
        if ctx.on_extract_done is not None:
            ctx.on_extract_done()
        row = ctx.db.query_one(
            "SELECT unit_count FROM extraction_runs WHERE run_id = ?", (ctx.run_id,)
        )
        return int(row["unit_count"] or 0) if row is not None else 0
