"""Garbage collection with reference checks (PRD §14, M7 slice 2).

The stores are append-only during ingestion: the content-addressed archive
(:mod:`library_rag.archive`), the artifact tree (extract units, embedding
checkpoints), and retained job logs are never deleted by the pipeline itself
(success logs are flushed by the worker, failure logs are only pruned by
count). This module reclaims what no catalog row references:

* **archive objects** — ``<archive_root>/<2hex>/<sha256>`` with no
  ``source_revisions`` row for that SHA (rows are never deleted; superseded
  revisions still pin their object);
* **extract artifacts** — ``<artifact_root>/extract/...`` with no
  ``source_units`` row;
* **embedding checkpoints** — ``<artifact_root>/embeddings/<run>/<sha>/
  batch_NNNNN.bin`` and its ``.json`` sidecar with no ``embedding_batches``
  row (a row references the ``.bin``; the sidecar follows it);
* **stale job logs** — ``<state_root>/job_logs/<job_id>.attempt<N>.log``
  whose job row no longer exists;
* **superseded index points** — Qdrant points whose publication row SQLite
  has marked ``superseded`` and aged past the grace window (see below).
  Points are deleted by filter; the ``publications`` row is kept as history
  (coverage and backup verification still see the lineage).

Safety:

* Refuses to run while any job is in state ``running`` — a live worker is
  mid-pipeline and may be writing store objects.
* **Grace period** (default 10 minutes): files modified within the window are
  skipped. This covers the scan window in which archive bytes are copied
  *before* the revision row is committed (see
  ``library_rag.scan._process_file``). For the points kind the window is
  measured from ``publications.superseded_at``: a row is only ``superseded``
  once the B4 switch has committed atomically (the replacement generation is
  fully active in the same transaction), so the state is definitive and the
  grace is a second belt — it keeps GC out of a freshly-switched doc.
  Rows superseded before the column existed carry NULL and are never
  collected (unknown age).
* Dry-run is the default: candidates are reported, nothing is deleted.
  ``--execute`` deletes, then removes the directories that became empty.
* The points kind needs a Qdrant client (``run_gc(..., qdrant=...)``);
  without one the file kinds still run and the report carries a note. In
  local (embedded) mode a running worker holds the storage lock, so the
  client cannot open while the worker is up — stop it first.

GC never touches source roots, the state DB, or the scratch root. The points
kind's only Qdrant write is a filtered delete of exactly the superseded
publication's point set — it never creates, updates, or reorganizes points
or the storage layout. Saved answers are unaffected: citations resolve from
the frozen evidence manifest in the ``answers`` table, not from Qdrant.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .db import Database
from .embeddings import checkpoint_manifest_path
from .indexing import FieldCond, IndexFilter, QdrantOps

__all__ = ["GcCandidate", "GcError", "GcReport", "run_gc"]

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LOG_NAME = re.compile(r"^(\d+)\.attempt(\d+)\.log$")


class GcError(RuntimeError):
    """Raised when GC cannot run safely (e.g. a worker is running)."""


@dataclass(frozen=True)
class GcCandidate:
    """One collectible store object.

    ``relpath`` is relative to the kind root; for the ``points`` kind it is
    the superseded publication id (there is no file — the delete is a
    Qdrant filter).
    """

    kind: str
    relpath: str
    size_bytes: int
    reason: str


@dataclass
class GcReport:
    candidates: list[GcCandidate] = field(default_factory=list)
    executed: bool = False
    grace_seconds: float = 600.0
    deleted: int = 0
    bytes_reclaimed: int = 0
    directories_removed: int = 0
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "grace_seconds": self.grace_seconds,
            "candidates": [asdict(c) for c in self.candidates],
            "deleted": self.deleted,
            "bytes_reclaimed": self.bytes_reclaimed,
            "directories_removed": self.directories_removed,
            "errors": self.errors,
            "notes": self.notes,
        }


def _kind_root(cfg: Config, kind: str) -> Path:
    if kind == "archive":
        return cfg.paths.archive_root
    if kind == "artifact":
        return cfg.paths.artifact_root
    return cfg.paths.state_root / "job_logs"


def _running_jobs(db: Database) -> int:
    row = db.query_one("SELECT COUNT(*) AS n FROM jobs WHERE state = 'running'")
    return int(row["n"]) if row is not None else 0


def _modified_within_grace(path: Path, now: float, grace_seconds: float) -> bool:
    try:
        return (now - path.stat().st_mtime) <= grace_seconds
    except OSError:
        return True  # vanished under us: skip, do not report


def _add_artifact_ref(refs: set[str], relpath: str) -> None:
    refs.add(relpath)
    if relpath.endswith(".bin"):
        refs.add(checkpoint_manifest_path(Path(relpath)).as_posix())


_Evaluator = Callable[[str], str | None]


def _archive_candidate_reasons(db: Database) -> _Evaluator:
    """Reason-evaluator for the archive tree (SHA is the filename)."""
    referenced = {r["sha256"] for r in db.query("SELECT DISTINCT sha256 FROM source_revisions")}

    def evaluate(rel: str) -> str | None:
        parts = Path(rel).parts
        sha = Path(rel).name
        if len(parts) != 2 or parts[0] != sha[:2] or _HEX64.fullmatch(sha) is None:
            return "malformed archive path"
        if sha in referenced:
            return None
        return "no source_revisions reference"

    return evaluate


def _artifact_candidate_reasons(db: Database) -> _Evaluator:
    """Return a reason-evaluator for the artifact tree."""
    refs: set[str] = set()
    for r in db.query("SELECT artifact_relpath FROM source_units"):
        _add_artifact_ref(refs, r["artifact_relpath"])
    for r in db.query("SELECT artifact_relpath FROM embedding_batches"):
        _add_artifact_ref(refs, r["artifact_relpath"])

    def evaluate(rel: str) -> str | None:
        return None if rel in refs else "no catalog reference"

    return evaluate


def _job_log_candidate_reasons(db: Database) -> _Evaluator:
    """Reason-evaluator for the job-log directory."""
    live = {int(r["job_id"]) for r in db.query("SELECT job_id FROM jobs")}

    def evaluate(rel: str) -> str | None:
        m = _LOG_NAME.fullmatch(rel)
        if m is None:
            return None  # not a pipeline log; never collect unrecognized names
        return None if int(m.group(1)) in live else "job row gone"

    return evaluate


def _measured_point_bytes(cfg: Config, qdrant: QdrantOps) -> float:
    """Bytes per point, measured: total storage bytes / total points.

    Informational — it only sizes the points candidates in the report.
    Falls back to 0.0 when the storage layout is unknown (remote mode,
    directory absent) or the collection is empty, so a wrong estimate can
    never make a delete decision (deletes are driven by the publications
    rows, not by size).
    """
    root = (
        Path(cfg.services.qdrant_path) if cfg.services.qdrant_path else cfg.paths.qdrant_root
    )
    total = 0
    if root.is_dir():
        try:
            for f in root.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except OSError:
            return 0.0
    try:
        points = qdrant.count(IndexFilter.all())
    except Exception:
        return 0.0
    if points <= 0:
        return 0.0
    return total / points


def _prune_empty_dirs(root: Path) -> int:
    """Remove directories under *root* that have become empty (deepest first).

    Each bottom-up pass re-checks the *live* directory contents, because a
    pass that removes a child makes its parent empty only afterwards (the
    walk's dirnames list is taken before children are pruned). The loop
    repeats until a pass removes nothing.
    """
    n = 0
    if not root.is_dir():
        return n
    changed = True
    while changed:
        changed = False
        for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
            d = Path(dirpath)
            if d == root:
                continue
            if any(d.iterdir()):
                continue
            try:
                d.rmdir()
                n += 1
                changed = True
            except OSError:
                pass
    return n


def run_gc(
    db: Database,
    cfg: Config,
    *,
    execute: bool = False,
    grace_seconds: float = 600.0,
    now: float | None = None,
    qdrant: QdrantOps | None = None,
) -> GcReport:
    """Collect (and, with ``execute=True``, delete) unreferenced store objects.

    See the module docstring for the reference rules and the safety guards.
    ``qdrant`` enables the ``points`` kind (superseded publications past the
    grace window); without it the file kinds still run and the report notes
    that the points kind was skipped. A :class:`GcError` is raised when a
    worker is running; per-file and per-point delete failures are recorded
    in the report instead of raising.
    """
    if grace_seconds < 0:
        raise GcError("grace_seconds must be >= 0")
    running = _running_jobs(db)
    if running:
        raise GcError(
            f"{running} job(s) are running; stop the worker before GC "
            "(pause, or wait for the in-flight job to finish)"
        )
    ts = time.time() if now is None else now

    candidates: list[GcCandidate] = []
    for kind, root, evaluate in (
        ("archive", _kind_root(cfg, "archive"), _archive_candidate_reasons(db)),
        ("artifact", _kind_root(cfg, "artifact"), _artifact_candidate_reasons(db)),
        ("job_log", _kind_root(cfg, "job_log"), _job_log_candidate_reasons(db)),
    ):
        if not root.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            base = Path(dirpath)
            for name in sorted(filenames):
                p = base / name
                if _modified_within_grace(p, ts, grace_seconds):
                    continue
                rel = p.relative_to(root).as_posix()
                reason = evaluate(rel)
                if reason is None:
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                candidates.append(
                    GcCandidate(kind=kind, relpath=rel, size_bytes=size, reason=reason)
                )

    report = GcReport(candidates=candidates, executed=execute, grace_seconds=grace_seconds)
    if qdrant is None:
        report.notes.append(
            "points: skipped (no Qdrant client; in local mode stop the worker first)"
        )
    else:
        per_point = _measured_point_bytes(cfg, qdrant)
        for r in db.query(
            """
            SELECT pub_id, expected_points, superseded_at
            FROM publications
            WHERE state = 'superseded' AND superseded_at IS NOT NULL
              AND superseded_at <= ?
            ORDER BY pub_id
            """,
            (ts - grace_seconds,),
        ):
            candidates.append(
                GcCandidate(
                    kind="points",
                    relpath=r["pub_id"],
                    size_bytes=int(int(r["expected_points"]) * per_point),
                    reason=f"publication superseded {int(ts - r['superseded_at'])}s ago",
                )
            )
    if not execute:
        return report

    for c in candidates:
        if c.kind == "points":
            assert qdrant is not None
            try:
                qdrant.delete(IndexFilter.all(FieldCond("pub_id", "eq", c.relpath)))
                report.deleted += 1
                report.bytes_reclaimed += c.size_bytes
            except Exception as exc:
                report.errors.append(f"points: {c.relpath}: {exc}")
            continue
        p = _kind_root(cfg, c.kind) / c.relpath
        try:
            if p.is_file():
                p.unlink()
                report.deleted += 1
                report.bytes_reclaimed += c.size_bytes
            else:
                report.errors.append(f"{c.kind}: {c.relpath} (vanished before delete)")
        except OSError as exc:
            report.errors.append(f"{c.kind}: {c.relpath}: {exc}")
    report.directories_removed = (
        _prune_empty_dirs(_kind_root(cfg, "archive"))
        + _prune_empty_dirs(_kind_root(cfg, "artifact"))
    )
    return report
