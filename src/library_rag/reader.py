"""Source reader: ranged delivery of archived sources, page/anchor mapping.

Serves only what is durable — the archived original (HTTP Range-capable, so
PDF.js can fetch the byte ranges it needs) and verified unit artifacts
(checksum-checked before trust, PRD §6/§7). It never re-derives content and
never resolves a path outside the archive root.

Page conventions (PRD §8B): internal unit positions are zero-based; reader and
PDF.js page numbers are one-based. EPUB citations are *(section, paragraph
anchor)* — chapter/paragraph, never a fabricated page number (PRD §8D).
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from .config import Config
from .db import Database
from .extraction.store import load_unit_artifact

__all__ = ["active_run", "create_app", "pdfjs_page_for", "resolve_anchor"]

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK = 1 << 20  # 1 MiB stream unit


def pdfjs_page_for(position: int, label: str | None = None) -> dict[str, Any]:
    """Map a zero-based internal unit position to its one-based reader page.

    PDF.js numbers pages 1..N in internal order; *label* is the page label
    recorded at extraction time, kept for display only.
    """
    if position < 0:
        raise ValueError("position must be a zero-based non-negative index")
    return {"page": position + 1, "label": label or str(position + 1)}


def resolve_anchor(payload: dict[str, Any], anchor: str) -> str | None:
    """Return the paragraph text for *anchor* in a section unit, or None."""
    for para in payload.get("paragraphs", []):
        if para.get("anchor") == anchor:
            text: str | None = para.get("text")
            return text
    return None


def active_run(db: Database, rev_id: str) -> dict[str, Any] | None:
    """The most recent succeeded extraction run for *rev_id*, or None."""
    row = db.query_one(
        """
        SELECT * FROM extraction_runs
        WHERE rev_id = ? AND state = 'succeeded'
        ORDER BY updated_at DESC, run_id DESC LIMIT 1
        """,
        (rev_id,),
    )
    return dict(row) if row is not None else None


def _media_type(fmt: str) -> str:
    return "application/pdf" if fmt == "pdf" else "application/epub+zip"


def create_app(cfg: Config, db: Database) -> FastAPI:
    """Build the read-only source-reader app over *cfg*/*db*."""
    app = FastAPI(title="library-rag source reader", docs_url=None, redoc_url=None)

    def _rev(rev_id: str) -> dict[str, Any]:
        row = db.query_one("SELECT * FROM source_revisions WHERE rev_id = ?", (rev_id,))
        if row is None:
            raise HTTPException(404, "unknown revision")
        return dict(row)

    def _unit_payload(rev_id: str, unit_id: str) -> dict[str, Any]:
        row = db.query_one(
            "SELECT artifact_sha256 FROM source_units WHERE unit_id = ? AND rev_id = ?",
            (unit_id, rev_id),
        )
        if row is None:
            raise HTTPException(404, "unknown unit for this revision")
        try:
            return load_unit_artifact(cfg.paths.artifact_root, rev_id, unit_id, row["artifact_sha256"])
        except (OSError, ValueError) as exc:
            raise HTTPException(502, f"unit artifact unreadable: {exc}") from exc

    @app.get("/books/{rev_id}")
    def book_manifest(rev_id: str) -> dict[str, Any]:
        rev = _rev(rev_id)
        run = active_run(db, rev_id)
        if run is None:
            raise HTTPException(409, "revision not extracted yet")
        units = db.query(
            """
            SELECT unit_id, position, kind, ref, char_count, quality_flags,
                   route, ocr_state
            FROM source_units WHERE run_id = ? ORDER BY position
            """,
            (run["run_id"],),
        )
        return {
            "rev_id": rev_id,
            "doc_id": rev["doc_id"],
            "format": rev["format"],
            "run_id": run["run_id"],
            "unit_count": int(run["unit_count"] or 0),
            "units": [dict(u) for u in units],
        }

    @app.get("/books/{rev_id}/source")
    def source(rev_id: str, range: str | None = Header(default=None)) -> StreamingResponse:
        """The archived original, with HTTP Range support (PRD M2 "ranged
        delivery"). 206 with Content-Range for valid ranges; 416 otherwise."""
        rev = _rev(rev_id)
        path = cfg.paths.archive_root / rev["archive_relpath"]
        if not path.is_file():
            raise HTTPException(410, "archived original missing")
        total = path.stat().st_size
        start, end = 0, total - 1
        status = 200
        headers: dict[str, str] = {
            "Accept-Ranges": "bytes",
            "Content-Type": _media_type(rev["format"]),
        }
        if range is not None:
            m = _RANGE_RE.match(range.strip())
            if m is None:
                raise HTTPException(416, "invalid Range header",
                                    headers={"Content-Range": f"bytes */{total}"})
            s, e = m.group(1), m.group(2)
            if s == "" and e == "":
                raise HTTPException(416, "invalid Range header",
                                    headers={"Content-Range": f"bytes */{total}"})
            if s == "":  # suffix range: last N bytes
                n = int(e)
                if n == 0:
                    raise HTTPException(416, "unsatisfiable range",
                                        headers={"Content-Range": f"bytes */{total}"})
                start, end = max(0, total - n), total - 1
            else:
                start = int(s)
                end = int(e) if e else total - 1
                if start >= total or end < start:
                    raise HTTPException(416, "unsatisfiable range",
                                        headers={"Content-Range": f"bytes */{total}"})
                end = min(end, total - 1)
            status = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{total}"
        headers["Content-Length"] = str(end - start + 1)

        def _stream() -> Any:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(_CHUNK, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(_stream(), status_code=status, headers=headers)

    @app.get("/books/{rev_id}/units/{unit_id}")
    def unit(rev_id: str, unit_id: str) -> dict[str, Any]:
        """A verified unit artifact. Page units get their one-based reader
        page attached; section units carry their paragraph anchors inline."""
        _rev(rev_id)
        payload = _unit_payload(rev_id, unit_id)
        out = dict(payload)
        if out.get("kind") == "page":
            out["pdfjs_page"] = pdfjs_page_for(int(out["position"]), out.get("label"))
        return out

    @app.get("/books/{rev_id}/units/{unit_id}/anchors/{anchor}")
    def anchor(rev_id: str, unit_id: str, anchor: str) -> dict[str, Any]:
        """Resolve an EPUB paragraph citation to its text (PRD §8D)."""
        _rev(rev_id)
        payload = _unit_payload(rev_id, unit_id)
        if payload.get("kind") != "section":
            raise HTTPException(409, "unit is not an EPUB section")
        text = resolve_anchor(payload, anchor)
        if text is None:
            raise HTTPException(404, "unknown anchor for this section")
        return {
            "anchor": anchor,
            "text": text,
            "section": payload.get("title"),
            "ref": payload.get("ref"),
            "position": payload.get("position"),
        }

    return app
