"""M5 citation contract (PRD §9/§12): the strict ``E1..En`` vocabulary, the
ABSTAIN contract, the frozen manifest snapshot (schema 1), and
``build_manifest``'s trusted-metadata derivation — page geometry with bbox
boxes and quality flags, section anchors from checksum-verified artifacts,
and explicit degradation (never a fabricated location) when the unit rows or
artifacts are missing or corrupt.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import asdict
from typing import Any

import pytest

from fixtures import publish_handbuilt
from library_rag.citations import (
    Evidence,
    Manifest,
    build_manifest,
    parse_abstention,
    parse_citations,
    unknown_citations,
)
from library_rag.config import Config
from library_rag.db import Database
from library_rag.embeddings import FakeEmbedder
from library_rag.extraction.store import unit_artifact_path
from library_rag.indexing import FakeQdrant
from library_rag.retrieval import Passage

# --- the E-id vocabulary -----------------------------------------------------------


def test_parse_citations_first_appearance_order_and_dedup() -> None:
    assert parse_citations("See [E3], then [E1], again [E3] and [E2].") == ["E3", "E1", "E2"]
    assert parse_citations("no citations here") == []


def test_parse_citations_rejects_out_of_contract_ids() -> None:
    # Bare IDs, malformed brackets, and >4-digit IDs are not citations.
    assert parse_citations("E1 and [E1 2] and [E99999]") == []
    assert parse_citations("lower [e1]") == []


def test_unknown_citations_flags_only_out_of_vocabulary_ids() -> None:
    assert unknown_citations("ok [E1] bad [E7] ok [E2] bad [E7]", ["E1", "E2"]) == ["E7"]
    assert unknown_citations("ok [E1]", ["E1"]) == []


# --- the ABSTAIN contract ------------------------------------------------------------


def test_parse_abstention_contract() -> None:
    assert parse_abstention("ABSTAIN\nNot covered by the library.") == "Not covered by the library."
    # Case-insensitive, tolerant of leading blank lines.
    assert parse_abstention("\n\nabstain\nthe evidence is insufficient.") == (
        "the evidence is insufficient."
    )
    # Bare abstention gets the documented default reason.
    assert parse_abstention("ABSTAIN") == "evidence insufficient"
    # Anything else on the first line is an answer, not an abstention.
    assert parse_abstention("ABSTAIN because [E1]") is None
    assert parse_abstention("The evidence is insufficient.") is None


# --- the frozen snapshot --------------------------------------------------------------


def _evidence(i: int = 1) -> Evidence:
    return Evidence(
        evidence_id=f"E{i}",
        chunk_id=f"runA:chunk-{i}",
        doc_id="docA",
        rev_id="revA",
        run_id="runA",
        text=f"Passage {i}.",
        title="Chapter One",
        source_title="revA",
        format="pdf",
        location={"kind": "page", "page": i, "label": f"p{i}"},
        boxes=({"unit_id": f"runA:chunk-{i}:u0", "page": i, "bbox": [1.0, 2.0, 3.0, 4.0]},),
        quality_flags=("sparse",),
    )


def test_evidence_from_dict_round_trip() -> None:
    raw = json.loads(json.dumps(asdict(_evidence(2))))
    assert Evidence.from_dict(raw) == _evidence(2)


def test_evidence_from_dict_tolerates_missing_optional_fields() -> None:
    d = asdict(_evidence(1))
    d.pop("title")
    d["boxes"] = []
    d["quality_flags"] = []
    e = Evidence.from_dict(d)
    assert e.title is None
    assert e.boxes == ()
    assert e.quality_flags == ()


def test_manifest_json_round_trip() -> None:
    m = Manifest((_evidence(1), _evidence(2)))
    raw = m.to_json()
    assert '"schema":1' in raw
    assert Manifest.from_json(raw) == m


def test_manifest_from_json_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="unsupported evidence manifest"):
        Manifest.from_json(json.dumps({"schema": 999, "evidence": []}))


def test_manifest_from_json_rejects_bad_evidence_field() -> None:
    with pytest.raises(ValueError, match="'evidence' must be a list"):
        Manifest.from_json(json.dumps({"schema": 1, "evidence": "nope"}))


# --- build_manifest --------------------------------------------------------------------


Library = tuple[Database, Config, FakeQdrant, FakeEmbedder, list[str]]


@pytest.fixture()
def library(state_db: Database, base_config: Config) -> Library:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    _, _, chunk_ids = publish_handbuilt(
        state_db,
        base_config,
        q,
        doc_id="docA",
        rev_id="revA",
        run_id="runA",
        texts=[f"Zebra passage number {i}." for i in range(4)],
        title="Book A",
    )
    return state_db, base_config, q, emb, chunk_ids


def _passage(
    chunk_id: str = "runA:chunk-0",
    *,
    spans: tuple[dict[str, Any], ...] = (),
) -> Passage:
    return Passage(
        chunk_id=chunk_id,
        doc_id="docA",
        rev_id="revA",
        run_id="runA",
        gen_id="genA",
        pub_id="pubA",
        title="Chapter One",
        text="The zebra sleeps on the plain.",
        spans=spans,
        score=1.0,
        dense_rank=1,
        sparse_rank=None,
    )


def _insert_unit(
    db: Database,
    unit_id: str,
    *,
    kind: str,
    position: int,
    ref: str | None = None,
    flags: str | None = None,
    sha: str = "0" * 64,
) -> None:
    db.execute(
        """
        INSERT INTO source_units (
            unit_id, run_id, rev_id, kind, position, ref, char_count, rotation,
            width, height, quality_flags, artifact_relpath, artifact_sha256,
            created_at
        ) VALUES (?, 'runA', 'revA', ?, ?, ?, 10, 0, 595.0, 842.0, ?, 'x.json.gz', ?, ?)
        """,
        (unit_id, kind, position, ref, flags, sha, time.time()),
    )


def test_build_manifest_without_units_gets_unknown_location(library: Library) -> None:
    db, cfg, _, _, _ = library
    m = build_manifest(db, cfg, (_passage(),))
    e = m.evidence[0]
    assert e.evidence_id == "E1"
    assert e.location == {"kind": "unknown"}
    assert e.source_title == "revA"  # stem of the registered first_path
    assert e.format == "pdf"
    assert e.boxes == ()
    assert e.quality_flags == ()


def test_build_manifest_page_geometry_flags_and_multipage(library: Library) -> None:
    db, cfg, _, _, _ = library
    _insert_unit(db, "runA:chunk-0:u0", kind="page", position=3, ref="p4", flags='["sparse"]')
    _insert_unit(db, "runA:chunk-0:u1", kind="page", position=5)
    p = _passage(
        spans=(
            {
                "unit_id": "runA:chunk-0:u0",
                "source_start": 0,
                "source_end": 5,
                "bbox": [1.0, 2.0, 3.0, 4.0],
            },
            {
                "unit_id": "runA:chunk-0:u1",
                "source_start": 5,
                "source_end": 10,
                "bbox": [9.0, 8.0, 7.0, 6.0],
            },
        )
    )
    m = build_manifest(db, cfg, (p,))
    e = m.evidence[0]
    assert e.location["kind"] == "page"
    assert e.location["page"] == 4  # 1-based page number
    assert e.location["label"] == "p4"
    # A chunk spanning several pages lists them for the reader UI.
    assert e.location["pages"] == [
        {"page": 4, "label": "p4"},
        {"page": 6, "label": None},
    ]
    assert e.boxes == (
        {"unit_id": "runA:chunk-0:u0", "page": 4, "bbox": [1.0, 2.0, 3.0, 4.0]},
        {"unit_id": "runA:chunk-0:u1", "page": 6, "bbox": [9.0, 8.0, 7.0, 6.0]},
    )
    assert e.quality_flags == ("sparse",)


def test_build_manifest_page_span_without_bbox_yields_no_box(library: Library) -> None:
    db, cfg, _, _, _ = library
    _insert_unit(db, "runA:chunk-0:u0", kind="page", position=0)
    p = _passage(
        spans=(
            {"unit_id": "runA:chunk-0:u0", "source_start": 0, "source_end": 5, "bbox": None},
        )
    )
    m = build_manifest(db, cfg, (p,))
    assert m.evidence[0].location == {"kind": "page", "page": 1, "label": None}
    assert m.evidence[0].boxes == ()


def _write_section_artifact(cfg: Config, rev_id: str, unit_id: str, payload: dict[str, Any]) -> str:
    """Write a real gzipped unit artifact; returns its sha256 hex digest."""
    body = gzip.compress(
        json.dumps({"schema": 1, **payload}, sort_keys=True, separators=(",", ":")).encode()
    )
    path = unit_artifact_path(cfg.paths.artifact_root, rev_id, unit_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def test_build_manifest_section_anchor_from_verified_artifact(library: Library) -> None:
    db, cfg, _, _, _ = library
    sha = _write_section_artifact(
        cfg,
        "revA",
        "runA:chunk-0:u0",
        {"title": "Chapter One", "paragraphs": [{"text": "Hello.", "anchor": "ch1-p0"}]},
    )
    _insert_unit(
        db,
        "runA:chunk-0:u0",
        kind="section",
        position=0,
        ref="ch1.xhtml",
        sha=sha,
    )
    p = _passage(
        spans=(
            {
                "unit_id": "runA:chunk-0:u0",
                "source_start": 0,
                "source_end": 6,
                "bbox": None,
            },
        )
    )
    m = build_manifest(db, cfg, (p,))
    e = m.evidence[0]
    assert e.location == {
        "kind": "section",
        "ref": "ch1.xhtml",
        "anchor": "ch1-p0",
        "title": "Chapter One",
    }
    assert e.boxes == ()


def test_build_manifest_corrupt_artifact_degrades_instead_of_fabricating(library: Library) -> None:
    db, cfg, _, _, _ = library
    # Row checksum does not match the file on disk: the loader must refuse it.
    sha = _write_section_artifact(
        cfg,
        "revA",
        "runA:chunk-0:u0",
        {"title": "Chapter One", "paragraphs": [{"text": "Hello.", "anchor": "ch1-p0"}]},
    )
    assert sha  # sanity: the artifact was written
    _insert_unit(
        db,
        "runA:chunk-0:u0",
        kind="section",
        position=0,
        ref="ch1.xhtml",
        sha="f" * 64,  # deliberately wrong
    )
    p = _passage(
        spans=(
            {
                "unit_id": "runA:chunk-0:u0",
                "source_start": 0,
                "source_end": 6,
                "bbox": None,
            },
        )
    )
    m = build_manifest(db, cfg, (p,))
    e = m.evidence[0]
    assert e.location == {
        "kind": "section",
        "ref": "ch1.xhtml",
        "anchor": None,
        "title": None,
    }


def test_build_manifest_missing_revision_degrades_metadata(library: Library) -> None:
    db, cfg, _, _, _ = library
    # A passage whose revision row was never registered (or was purged).
    p = Passage(
        chunk_id="runZ:chunk-0",
        doc_id="docZ",
        rev_id="revGhost",
        run_id="runZ",
        gen_id="genZ",
        pub_id="pubZ",
        title=None,
        text="Orphan passage.",
        spans=(),
        score=1.0,
        dense_rank=1,
        sparse_rank=None,
    )
    m = build_manifest(db, cfg, (p,))
    e = m.evidence[0]
    assert e.source_title == "docZ"  # falls back to the doc id
    assert e.format == ""
    assert e.location == {"kind": "unknown"}
