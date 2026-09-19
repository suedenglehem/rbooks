"""Evidence manifests and citation validation (PRD §9/§12).

The evidence manifest is the *only* vocabulary the model may cite: numbered
entries ``E1..En`` built from validated passages plus metadata taken from
trusted sources — the ``source_revisions``/``chunks``/``source_units`` rows
and the checksum-verified unit artifacts. The model's own output is never
trusted for location data.

Each saved answer persists its manifest as a frozen snapshot
(:meth:`Manifest.to_json`, schema 1). Citation resolution reads the snapshot,
so re-chunking or re-publishing the library (new run, new chunk IDs, new
publication) cannot change what a saved answer points at; the only live check
is that the source revision is still registered and its archive readable.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .db import Database
from .extraction.store import load_unit_artifact
from .retrieval import Passage

__all__ = [
    "Evidence",
    "Manifest",
    "build_manifest",
    "parse_abstention",
    "parse_citations",
    "unknown_citations",
]

_CITATION_RE = re.compile(r"\[(E(\d{1,4}))\]")


@dataclass(frozen=True)
class Evidence:
    """One numbered manifest entry (``E1..En``), frozen at answer time.

    *location* is the primary citation target::

        {"kind": "page", "page": <1-based PDF.js page>, "label": "<page label>"}
        {"kind": "section", "ref": "<spine href>", "anchor": "<paragraph anchor>",
         "title": "<section title or None>"}

    *boxes* carries per-span highlight geometry for PDF evidence
    (``{"unit_id", "page", "bbox"}``; ``page`` is 1-based or None for
    sections, ``bbox`` is page points or None). *quality_flags* are the unit
    quality warnings (no_text, sparse, many_replacement, bad_html, ...) the
    citation UI must surface.
    """

    evidence_id: str
    chunk_id: str
    doc_id: str
    rev_id: str
    run_id: str
    text: str
    title: str | None
    source_title: str
    format: str
    location: dict[str, object]
    boxes: tuple[dict[str, object], ...]
    quality_flags: tuple[str, ...]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Evidence:
        """Rebuild an entry from its stored (dict) representation."""
        return cls(
            evidence_id=str(d["evidence_id"]),
            chunk_id=str(d["chunk_id"]),
            doc_id=str(d["doc_id"]),
            rev_id=str(d["rev_id"]),
            run_id=str(d["run_id"]),
            text=str(d["text"]),
            title=d.get("title") if isinstance(d.get("title"), str) else None,
            source_title=str(d["source_title"]),
            format=str(d["format"]),
            location=dict(d.get("location") or {}),
            boxes=tuple(dict(b) for b in (d.get("boxes") or ())),
            quality_flags=tuple(str(f) for f in (d.get("quality_flags") or ())),
        )


@dataclass(frozen=True)
class Manifest:
    """The numbered evidence set handed to the model and persisted with it."""

    evidence: tuple[Evidence, ...]

    def to_json(self) -> str:
        return json.dumps(
            {"schema": 1, "evidence": [asdict(e) for e in self.evidence]},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, data: str) -> Manifest:
        parsed = json.loads(data)
        if not isinstance(parsed, dict) or parsed.get("schema") != 1:
            raise ValueError(f"unsupported evidence manifest: {parsed.get('schema') if isinstance(parsed, dict) else type(parsed).__name__}")
        raw = parsed.get("evidence")
        if not isinstance(raw, list):
            raise ValueError("evidence manifest: 'evidence' must be a list")
        return cls(tuple(Evidence.from_dict(e) for e in raw))


# --- manifest construction ------------------------------------------------------


def _unit_bounds(payload: dict[str, object]) -> list[tuple[int, int, str]]:
    """Paragraph ``(start, end, anchor)`` bounds over the section text.

    Offsets are relative to the exact text the chunker consumed
    (paragraphs joined with ``"\\n\\n"`` — see the worker's ``_unit_text``),
    so chunk span offsets map onto them directly.
    """
    raw = payload.get("paragraphs")
    bounds: list[tuple[int, int, str]] = []
    pos = 0
    if isinstance(raw, list):
        for p in raw:
            if not isinstance(p, dict):
                continue
            text = str(p.get("text", ""))
            bounds.append((pos, pos + len(text), str(p.get("anchor", ""))))
            pos += len(text) + 2
    return bounds


def _as_int(value: object, default: int) -> int:
    """Best-effort int from an untrusted payload value (never fabricates)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return default


def _anchor_at(payload: dict[str, object], source_start: int) -> str | None:
    for start, end, anchor in _unit_bounds(payload):
        if start <= source_start < end:
            return anchor or None
    return None


def _primary_location(
    first_unit: dict[str, object],
    first_span: dict[str, object] | None,
    section_payload: dict[str, object] | None,
) -> dict[str, object]:
    if str(first_unit.get("kind")) == "section":
        anchor = _anchor_at(section_payload, _as_int(first_span.get("source_start"), 0)) if (
            section_payload is not None and first_span is not None
        ) else None
        return {
            "kind": "section",
            "ref": str(first_unit.get("ref") or ""),
            "anchor": anchor,
            "title": section_payload.get("title") if section_payload is not None else None,
        }
    return {
        "kind": "page",
        "page": _as_int(first_unit.get("position"), 0) + 1,
        "label": first_unit.get("ref"),
    }


def build_manifest(db: Database, cfg: Config, passages: Sequence[Passage]) -> Manifest:
    """Build ``E1..En`` from validated passages with trusted location metadata.

    Metadata sources: ``source_revisions`` (format, display title),
    ``source_units`` rows (page geometry, quality flags), and — for EPUB
    sections — the checksum-verified unit artifact (paragraph anchors). A
    missing unit row or unreadable artifact degrades that entry's geometry to
    an explicit ``None`` rather than fabricating a location.
    """
    revs: dict[str, dict[str, object] | None] = {}
    units: dict[str, dict[str, object] | None] = {}
    sections: dict[tuple[str, str], dict[str, object] | None] = {}

    def rev_row(rev_id: str) -> dict[str, object] | None:
        if rev_id not in revs:
            row = db.query_one(
                "SELECT * FROM source_revisions WHERE rev_id = ?", (rev_id,)
            )
            revs[rev_id] = dict(row) if row is not None else None
        return revs[rev_id]

    def unit_row(unit_id: str) -> dict[str, object] | None:
        if unit_id not in units:
            row = db.query_one(
                "SELECT * FROM source_units WHERE unit_id = ?", (unit_id,)
            )
            units[unit_id] = dict(row) if row is not None else None
        return units[unit_id]

    def section_payload(rev_id: str, row: dict[str, object]) -> dict[str, object] | None:
        key = (rev_id, str(row["unit_id"]))
        if key not in sections:
            payload: dict[str, object] | None = None
            try:
                payload = load_unit_artifact(
                    cfg.paths.artifact_root,
                    rev_id,
                    str(row["unit_id"]),
                    expect_sha256=str(row["artifact_sha256"]),
                )
            except (OSError, ValueError):
                payload = None  # degraded geometry, never a fabricated one
            sections[key] = payload
        return sections[key]

    evidence: list[Evidence] = []
    for i, p in enumerate(passages, start=1):
        rev = rev_row(p.rev_id)
        source_title = Path(str(rev.get("first_path") or p.doc_id)).stem if rev else p.doc_id
        fmt = str(rev.get("format") or "") if rev else ""

        boxes: list[dict[str, object]] = []
        flags: set[str] = set()
        unit_ids: list[str] = []
        for span in p.spans:
            uid = str(span.get("unit_id") or "")
            row = unit_row(uid) if uid else None
            if row is None:
                continue
            if uid not in unit_ids:
                unit_ids.append(uid)
            page_no = (
                _as_int(row["position"], 0) + 1 if str(row["kind"]) == "page" else None
            )
            bbox = span.get("bbox")
            if isinstance(bbox, Sequence) and len(bbox) == 4 and all(
                isinstance(v, (int, float)) for v in bbox
            ):
                boxes.append({"unit_id": uid, "page": page_no, "bbox": list(bbox)})
            q = row.get("quality_flags")
            if q:
                with contextlib.suppress(ValueError):
                    flags.update(json.loads(str(q)))

        first_row = unit_row(unit_ids[0]) if unit_ids else None
        first_span = next(
            (s for s in p.spans if isinstance(s, dict) and s.get("unit_id")), None
        )
        section_data: dict[str, object] | None = None
        if first_row is not None and str(first_row.get("kind")) == "section":
            section_data = section_payload(p.rev_id, first_row)

        if first_row is None:
            location: dict[str, object] = {"kind": "unknown"}
        else:
            location = _primary_location(first_row, first_span, section_data)

        # Multi-page chunks: the extra pages are listed for the reader UI.
        pages: list[dict[str, object]] = []
        seen: set[int] = set()
        for uid in unit_ids:
            row = units.get(uid)
            if row is not None and str(row.get("kind")) == "page":
                pos = _as_int(row["position"], 0)
                if pos not in seen:
                    seen.add(pos)
                    pages.append({"page": pos + 1, "label": row.get("ref")})
        if len(pages) > 1:
            location["pages"] = pages

        evidence.append(
            Evidence(
                evidence_id=f"E{i}",
                chunk_id=p.chunk_id,
                doc_id=p.doc_id,
                rev_id=p.rev_id,
                run_id=p.run_id,
                text=p.text,
                title=p.title,
                source_title=source_title,
                format=fmt,
                location=location,
                boxes=tuple(boxes),
                quality_flags=tuple(sorted(flags)),
            )
        )
    return Manifest(tuple(evidence))


# --- model-output parsing -------------------------------------------------------


def parse_citations(text: str) -> list[str]:
    """Citation IDs cited in *text*, in first-appearance order, deduplicated."""
    return list(dict.fromkeys(m.group(1) for m in _CITATION_RE.finditer(text)))


def unknown_citations(text: str, known_ids: Sequence[str]) -> list[str]:
    """Cited IDs that are not part of the manifest (must be rejected)."""
    known = set(known_ids)
    return [c for c in parse_citations(text) if c not in known]


def parse_abstention(text: str) -> str | None:
    """The abstention reason if *text* is a contract abstention, else None.

    Contract: ``ABSTAIN`` (case-insensitive) on the first non-empty line, with
    the one-sentence reason on the following lines.
    """
    lines = [ln.strip() for ln in text.strip().splitlines()]
    if not lines or lines[0].upper() != "ABSTAIN":
        return None
    reason = " ".join(lines[1:]).strip()
    return reason or "evidence insufficient"
