"""Extraction-run lifecycle and unit-artifact storage (PRD §5, §7).

Page/section artifacts are compressed JSON with a schema version, committed
with the atomic rename protocol in :mod:`library_rag.artifacts`, and checksummed
(SHA-256 of the compressed bytes) into ``source_units.artifact_sha256``. The
checksum lets a reader verify an artifact before trusting it and lets a
re-extraction detect (and replace) output from a crashed run.

Resume model (PRD §7 "expensive work resumes without repeating completed
units"): a unit is *done* when its ``source_units`` row exists **and** the
artifact verifies against the stored checksum. The artifact is committed before
its row, so every crash point leaves either (nothing) or (a verified unit) or
(a stray artifact with no row — harmless, re-extracted idempotently).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..db import Database

__all__ = [
    "SCHEMA_VERSION",
    "ExtractorCtx",
    "commit_unit_artifact",
    "existing_verified_unit",
    "fail_run",
    "finish_run",
    "insert_unit",
    "load_unit_artifact",
    "start_run",
    "unit_artifact_path",
]

SCHEMA_VERSION = 1


@dataclass
class ExtractorCtx:
    """Everything an extractor needs, injected by the worker."""

    db: Database
    artifact_root: Path
    rev: dict[str, Any]  # the source_revisions row being extracted
    run_id: str
    settings: Any  # library_rag.config.ExtractionSettings
    source_path: Path  # the archived original to read
    # Called between units so the worker can heartbeat the lease.
    on_progress: Callable[[], None] | None = None


def unit_artifact_path(artifact_root: Path, rev_id: str, unit_id: str) -> Path:
    """``<artifact_root>/extract/<rev_id>/<unit_id>.json.gz``."""
    return artifact_root / "extract" / rev_id / f"{unit_id}.json.gz"


def commit_unit_artifact(ctx: ExtractorCtx, unit_id: str, payload: dict[str, Any]) -> str:
    """Durely write a unit artifact; return the SHA-256 of the stored bytes."""
    from ..artifacts import commit_bytes  # local import: keeps the graph tidy

    envelope = {"schema": SCHEMA_VERSION, **payload}
    data = gzip.compress(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    dest = unit_artifact_path(ctx.artifact_root, ctx.rev["rev_id"], unit_id)
    commit_bytes(dest, data)
    return hashlib.sha256(data).hexdigest()


def load_unit_artifact(
    artifact_root: Path, rev_id: str, unit_id: str, expect_sha256: str | None
) -> dict[str, Any]:
    """Read and verify a unit artifact; raise ValueError on checksum mismatch."""
    path = unit_artifact_path(artifact_root, rev_id, unit_id)
    data = path.read_bytes()
    if expect_sha256 is not None and hashlib.sha256(data).hexdigest() != expect_sha256:
        raise ValueError(f"artifact checksum mismatch for unit {unit_id}")
    envelope = json.loads(gzip.decompress(data).decode("utf-8"))
    if not isinstance(envelope, dict) or envelope.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"unsupported artifact schema for unit {unit_id}")
    return envelope


def existing_verified_unit(
    ctx: ExtractorCtx, kind: str, position: int, unit_id: str
) -> bool:
    """True if *position* already has a verified unit (resume skip)."""
    row = ctx.db.query_one(
        "SELECT artifact_sha256 FROM source_units WHERE unit_id = ?", (unit_id,)
    )
    if row is None:
        return False
    path = unit_artifact_path(ctx.artifact_root, ctx.rev["rev_id"], unit_id)
    if not path.exists():
        return False
    try:
        load_unit_artifact(ctx.artifact_root, ctx.rev["rev_id"], unit_id, row["artifact_sha256"])
    except (ValueError, OSError):
        return False  # corrupt/missing artifact: re-extract this unit
    return True


def start_run(
    ctx: ExtractorCtx, parser_version: str, settings_sha: str, now: float | None = None
) -> bool:
    """Register the run; return False if it already succeeded (nothing to do).

    A previously-failed or interrupted run is resumed (state back to
    ``running``); a succeeded one is left alone — its units are the answer.
    """
    ts = time.time() if now is None else now
    with ctx.db.transaction():
        row = ctx.db.query_one(
            "SELECT state FROM extraction_runs WHERE run_id = ?", (ctx.run_id,)
        )
        if row is None:
            ctx.db.execute(
                """
                INSERT INTO extraction_runs
                    (run_id, rev_id, doc_id, parser_version, settings_sha, state,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
                """,
                (ctx.run_id, ctx.rev["rev_id"], ctx.rev["doc_id"],
                 parser_version, settings_sha, ts, ts),
            )
            return True
        if row["state"] == "succeeded":
            return False
        ctx.db.execute(
            """
            UPDATE extraction_runs SET state = 'running', error_category = NULL,
                   error_detail = NULL, updated_at = ?
            WHERE run_id = ?
            """,
            (ts, ctx.run_id),
        )
        return True


def fail_run(
    db: Database, run_id: str, category: str, detail: str, now: float | None = None
) -> None:
    """Mark a run failed (best-effort observability; the job carries the error)."""
    ts = time.time() if now is None else now
    with db.transaction():
        db.execute(
            "UPDATE extraction_runs SET state = 'failed', error_category = ?, "
            "error_detail = ?, updated_at = ? WHERE run_id = ?",
            (category, detail[:2000], ts, run_id),
        )


def finish_run(ctx: ExtractorCtx, unit_count: int, now: float | None = None) -> None:
    ts = time.time() if now is None else now
    with ctx.db.transaction():
        ctx.db.execute(
            "UPDATE extraction_runs SET state = 'succeeded', unit_count = ?, updated_at = ? "
            "WHERE run_id = ?",
            (unit_count, ts, ctx.run_id),
        )


def insert_unit(
    ctx: ExtractorCtx,
    *,
    unit_id: str,
    kind: str,
    position: int,
    ref: str | None,
    char_count: int,
    rotation: int | None,
    width: float | None,
    height: float | None,
    quality_flags: list[str] | None,
    artifact_relpath: str,
    artifact_sha256: str,
    now: float | None = None,
) -> None:
    """Record one unit (idempotent: re-extract of the same unit updates in place)."""
    ts = time.time() if now is None else now
    flags = json.dumps(quality_flags) if quality_flags is not None else None
    with ctx.db.transaction():
        ctx.db.execute(
            """
            INSERT INTO source_units
                (unit_id, run_id, rev_id, kind, position, ref, char_count, rotation,
                 width, height, quality_flags, artifact_relpath, artifact_sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(unit_id) DO UPDATE SET
                ref = excluded.ref,
                char_count = excluded.char_count,
                rotation = excluded.rotation,
                width = excluded.width,
                height = excluded.height,
                quality_flags = excluded.quality_flags,
                artifact_relpath = excluded.artifact_relpath,
                artifact_sha256 = excluded.artifact_sha256
            """,
            (unit_id, ctx.run_id, ctx.rev["rev_id"], kind, position, ref, char_count,
             rotation, width, height, flags, artifact_relpath, artifact_sha256, ts),
        )
