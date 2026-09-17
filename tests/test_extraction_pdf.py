"""PDF extraction (PRD §8B): one unit per physical page, quality flags,
resume after artifact loss, and the permanent failure taxonomy.

Together with test_reader.py this covers the M2 gate case "citations open
correct physical pages": here the units and their geometry, there the
zero-based -> one-based PDF.js mapping and ranged original delivery.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fixtures import ctx_for_rev, ingest_and_register, make_encrypted_pdf, make_pdf
from library_rag.catalog import Format
from library_rag.config import Config, ExtractionSettings, PdfLimits
from library_rag.db import Database
from library_rag.extraction import (
    ExtractionFailure,
    extract_pdf,
    is_permanent,
    load_unit_artifact,
    unit_artifact_path,
)
from library_rag.identity import extraction_key, unit_id_for

# One sentence: 48 non-ws chars (above the sparse threshold of 40) and short
# enough to fit one 11pt line on A4, so nothing is clipped at the page edge
# and the char count is exact.
_NORMAL = "The quick brown fox jumps over the lazy dog near the river."


def _units(db: Database, run_id: str) -> list[Any]:
    return db.query(
        """
        SELECT position, kind, ref, char_count, quality_flags
        FROM source_units WHERE run_id = ? ORDER BY position
        """,
        (run_id,),
    )


def test_extract_pdf_pages_and_flags(state_db: Database, base_config: Config) -> None:
    src = base_config.paths.scratch_root / "fox.pdf"
    make_pdf(src, [_NORMAL, "abc", None])
    rev_id = ingest_and_register(state_db, base_config, src, Format.PDF)
    ctx = ctx_for_rev(state_db, base_config, rev_id)

    assert extract_pdf(ctx) == 3
    row = state_db.query_one("SELECT state, unit_count FROM extraction_runs WHERE run_id = ?",
                             (ctx.run_id,))
    assert row is not None
    assert row["state"] == "succeeded"
    assert int(row["unit_count"]) == 3

    units = _units(state_db, ctx.run_id)
    assert [u["position"] for u in units] == [0, 1, 2]
    assert all(u["kind"] == "page" for u in units)
    assert [u["ref"] for u in units] == ["1", "2", "3"]
    assert [u["char_count"] for u in units] == [48, 3, 0]
    assert units[0]["quality_flags"] is None
    assert json.loads(units[1]["quality_flags"]) == ["sparse"]
    assert json.loads(units[2]["quality_flags"]) == ["no_text"]

    # The committed artifact verifies against its stored checksum and carries
    # text + geometry (the page unit the reader serves citations from).
    unit_id = unit_id_for(ctx.run_id, "page", 0)
    stored = state_db.query_one(
        "SELECT artifact_sha256 FROM source_units WHERE unit_id = ?", (unit_id,)
    )
    assert stored is not None
    payload = load_unit_artifact(
        base_config.paths.artifact_root, rev_id, unit_id, stored["artifact_sha256"]
    )
    assert payload["text"].startswith("The quick brown fox")
    assert payload["label"] == "1"
    assert payload["blocks"]


def test_extract_pdf_resume_after_artifact_loss(state_db: Database, base_config: Config) -> None:
    src = base_config.paths.scratch_root / "resume.pdf"
    make_pdf(src, [_NORMAL, "second page text", "third page text"])
    rev_id = ingest_and_register(state_db, base_config, src, Format.PDF)
    ctx = ctx_for_rev(state_db, base_config, rev_id)
    assert extract_pdf(ctx) == 3

    # Simulate a crash after the run was reset but before completion: mark the
    # run running again and lose one unit's artifact from disk.
    state_db.execute("UPDATE extraction_runs SET state = 'running' WHERE run_id = ?", (ctx.run_id,))
    lost_id = unit_id_for(ctx.run_id, "page", 1)
    lost = unit_artifact_path(base_config.paths.artifact_root, rev_id, lost_id)
    assert lost.is_file()
    lost.unlink()

    # The run id is deterministic (source sha + parser + settings), so a fresh
    # context resumes the same run and re-extracts only the lost unit.
    assert extract_pdf(ctx_for_rev(state_db, base_config, rev_id)) == 3
    assert lost.is_file()
    row = state_db.query_one("SELECT state FROM extraction_runs WHERE run_id = ?", (ctx.run_id,))
    assert row is not None and row["state"] == "succeeded"
    assert len(_units(state_db, ctx.run_id)) == 3  # idempotent: no duplicate rows


def test_extract_pdf_encrypted_is_permanent(state_db: Database, base_config: Config) -> None:
    src = base_config.paths.scratch_root / "locked.pdf"
    make_encrypted_pdf(src, [_NORMAL])
    rev_id = ingest_and_register(state_db, base_config, src, Format.PDF)
    ctx = ctx_for_rev(state_db, base_config, rev_id)
    with pytest.raises(ExtractionFailure) as ei:
        extract_pdf(ctx)
    assert ei.value.category == "encrypted"
    assert is_permanent(ei.value.category)


def test_extract_pdf_corrupt_is_permanent(state_db: Database, base_config: Config) -> None:
    src = base_config.paths.scratch_root / "corrupt.pdf"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"not a pdf")
    rev_id = ingest_and_register(state_db, base_config, src, Format.PDF)
    ctx = ctx_for_rev(state_db, base_config, rev_id)
    with pytest.raises(ExtractionFailure) as ei:
        extract_pdf(ctx)
    assert ei.value.category == "corrupt"
    assert is_permanent(ei.value.category)


def test_extract_pdf_missing_source_is_permanent(state_db: Database, base_config: Config) -> None:
    src = base_config.paths.scratch_root / "gone.pdf"
    make_pdf(src, [_NORMAL])
    rev_id = ingest_and_register(state_db, base_config, src, Format.PDF)
    ctx = ctx_for_rev(state_db, base_config, rev_id)
    ctx.source_path.unlink()
    with pytest.raises(ExtractionFailure) as ei:
        extract_pdf(ctx)
    assert ei.value.category == "missing_source"
    assert is_permanent(ei.value.category)


def test_extraction_key_is_deterministic() -> None:
    base = extraction_key("a" * 64, "pymupdf-1.0.0", "settings-1")
    assert base == extraction_key("a" * 64, "pymupdf-1.0.0", "settings-1")
    # A different parser version or settings snapshot means a fresh run.
    assert base != extraction_key("a" * 64, "pymupdf-9.9.9", "settings-1")
    other_settings = ExtractionSettings(pdf=PdfLimits(sparse_chars=10)).settings_sha()
    assert other_settings != "settings-1"
    assert base != extraction_key("a" * 64, "pymupdf-1.0.0", other_settings)
