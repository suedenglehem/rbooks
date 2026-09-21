"""Generation migration planning and execution (M7 slice 7, PRD §14).

A *generation migration* is the controlled transition of the index to a new
configuration — a chunker change, an embedding model/dtype/dimensions
change — that invalidates part or all of the currently published evidence.
Instead of doing that ad hoc, this module:

1. **detects drift** (read-only, safe with the worker running):
   * *rechunk* — succeeded runs whose stored chunk fingerprint differs from
     the current configuration's fingerprint (the same test the worker's
     reconcile pass uses), split into all runs and runs of *active*
     revisions (only the latter actually re-enter the index);
   * *reembed* — active-revision runs with chunks whose embedding checkpoint
     count no longer matches the current embedding configuration (mirrors
     the worker's reconcile pass);
   * *supersede* — the active publications of those revisions, whose point
     sets the new generation will make collectible by ``gc``.
2. **estimates capacity** (PRD line 190: two generations must fit, or a
   documented maintenance window is required). The estimate is proportional
   to the *measured* store size — ``storage * migrating_points /
   active_points`` — rather than the raw chunks*dimensions*dtype formula,
   because the store also holds the not-yet-superseded generations and the
   measured ratio already accounts for what Qdrant actually stores per
   point.
3. **executes** (``execute=True``) by enqueuing exactly the jobs the worker's
   startup reconcile pass would enqueue — idempotent, and safe with a running
   worker (the job queue is the serialization boundary; the worker simply
   interleaves). Execution is refused without ``accept_maintenance_window``
   when the capacity check says the new generation will not fit alongside
   the old one.

The supersede step needs no work of its own: when the re-generated
publication activates, the B4 switch marks the old publication
``superseded`` (stamping ``superseded_at``) atomically, and ``gc --execute``
reclaims its point set after the grace window.

No Qdrant client is needed at any point — this module only reads the state
DB and the storage directory's bytes on disk, so it works while the worker
holds the local lock.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, ConfigError
from .db import Database
from .embeddings import embedding_sha
from .worker import _reconcile_chunks, _reconcile_index, chunk_fingerprint_for_run

__all__ = ["MigrationError", "MigrationReport", "run_migration"]


class MigrationError(RuntimeError):
    """Raised when an executing migration is refused (maintenance window)."""


@dataclass
class MigrationReport:
    """What a generation migration would do, and whether it fits.

    ``migrating_points``/``active_points`` are sums of ``expected_points``
    over the affected / all active publications. ``fits`` is ``None`` when
    the capacity cannot be verified (unknown store size or free space);
    ``None`` counts as *not* fitting, so a migration of unknown capacity
    always needs the explicit maintenance-window acceptance.
    """

    rechunk_runs: int = 0
    rechunk_active_runs: int = 0
    reembed_runs: int = 0
    publications_to_supersede: int = 0
    migrating_points: int = 0
    active_points: int = 0
    qdrant_storage_bytes: int | None = None
    free_bytes: int | None = None
    second_generation_bytes: int | None = 0
    fits: bool | None = True
    maintenance_window_required: bool = False
    executed: bool = False
    enqueued_jobs: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _qdrant_store_bytes(cfg: Config) -> tuple[int | None, int | None]:
    """(total bytes in the Qdrant storage dir, free bytes on that volume).

    Either value is ``None`` when unknown: the directory does not exist
    (remote mode, or nothing indexed yet) or the volume stat failed.
    """
    root = Path(cfg.services.qdrant_path) if cfg.services.qdrant_path else cfg.paths.qdrant_root
    if not root.is_dir():
        return None, None
    total = 0
    try:
        for f in root.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        return None, None
    try:
        return total, int(shutil.disk_usage(root).free)
    except OSError:
        return total, None


def run_migration(
    db: Database,
    cfg: Config,
    *,
    execute: bool = False,
    accept_maintenance_window: bool = False,
) -> MigrationReport:
    """Plan (default) or execute a generation migration.

    Planning is read-only and safe with the worker running. Execution
    enqueues via the canonical reconcile pass and is refused without
    ``accept_maintenance_window`` when the new generation does not fit
    under two generations — see the module docstring for the procedure
    (stop the worker, ``gc --execute``, re-check, retry with the flag).
    """
    report = MigrationReport(executed=execute)

    # 1. Rechunk drift: stored fingerprint vs the current configuration.
    rechunk_revs: set[str] = set()
    for row in db.query(
        "SELECT run_id, rev_id, chunk_fingerprint FROM extraction_runs WHERE state = 'succeeded'"
    ):
        if row["chunk_fingerprint"] == chunk_fingerprint_for_run(db, row["run_id"], cfg):
            continue
        report.rechunk_runs += 1
        rev = db.query_one(
            "SELECT is_active FROM source_revisions WHERE rev_id = ?", (row["rev_id"],)
        )
        if rev is not None and rev["is_active"]:
            report.rechunk_active_runs += 1
            rechunk_revs.add(row["rev_id"])

    # 2. Reembed drift: checkpoint count vs the current embedding settings.
    reembed_revs: set[str] = set()
    try:
        emb_sha = embedding_sha(cfg)
    except ConfigError:
        emb_sha = None
    if emb_sha is not None:
        batch_size = max(1, cfg.embedding.batch_size)
        for row in db.query(
            "SELECT run_id, rev_id FROM extraction_runs WHERE state = 'succeeded'"
        ):
            rev = db.query_one(
                "SELECT is_active FROM source_revisions WHERE rev_id = ?", (row["rev_id"],)
            )
            if rev is None or not rev["is_active"]:
                continue
            n = db.query_one("SELECT COUNT(*) AS n FROM chunks WHERE run_id = ?", (row["run_id"],))
            n_chunks = int(n["n"] or 0) if n is not None else 0
            if n_chunks == 0:
                continue
            b = db.query_one(
                "SELECT COUNT(*) AS n FROM embedding_batches "
                "WHERE run_id = ? AND embedding_sha = ?",
                (row["run_id"], emb_sha),
            )
            assert b is not None
            expected = (n_chunks + batch_size - 1) // batch_size
            if int(b["n"]) != expected:
                report.reembed_runs += 1
                reembed_revs.add(row["rev_id"])

    # 3. Active publications the new generation will supersede.
    migrating_revs = rechunk_revs | reembed_revs
    if migrating_revs:
        placeholders = ", ".join("?" * len(migrating_revs))
        rows = db.query(
            "SELECT rev_id, expected_points FROM publications "
            f"WHERE state = 'active' AND rev_id IN ({placeholders})",
            tuple(migrating_revs),
        )
        report.publications_to_supersede = len(rows)
        report.migrating_points = sum(int(r["expected_points"]) for r in rows)

    active_row = db.query_one(
        "SELECT COALESCE(SUM(expected_points), 0) AS n FROM publications WHERE state = 'active'"
    )
    report.active_points = int(active_row["n"]) if active_row is not None else 0

    # 4. Capacity: two generations must fit, or a documented window.
    storage, free = _qdrant_store_bytes(cfg)
    report.qdrant_storage_bytes = storage
    report.free_bytes = free
    if report.migrating_points == 0:
        report.fits = True
    elif storage is None or free is None:
        report.fits = None
        report.notes.append(
            "Qdrant store size or free space unknown; capacity cannot be "
            "verified — treat the migration as requiring a maintenance window"
        )
    else:
        second_gen = storage * report.migrating_points // max(1, report.active_points)
        report.second_generation_bytes = second_gen
        report.fits = (storage + second_gen) <= free
    report.maintenance_window_required = report.fits is not True

    # 5. Execute: enqueue via the canonical reconcile pass (idempotent).
    if execute:
        if report.maintenance_window_required and not accept_maintenance_window:
            raise MigrationError(
                "the new generation does not fit alongside the current one "
                f"(active points {report.active_points}, migrating "
                f"{report.migrating_points}, store {storage} B, free {free} B); "
                "stop the worker and run `library-rag gc --execute` to reclaim "
                "superseded point sets, re-check with `library-rag migrate`, "
                "then retry with --accept-maintenance-window"
            )
        report.enqueued_jobs = _reconcile_chunks(db, cfg) + _reconcile_index(db, cfg)

    return report
