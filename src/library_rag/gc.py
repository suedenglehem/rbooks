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
  whose job row no longer exists.

Safety:

* Refuses to run while any job is in state ``running`` — a live worker is
  mid-pipeline and may be writing store objects.
* **Grace period** (default 10 minutes): files modified within the window are
  skipped. This covers the scan window in which archive bytes are copied
  *before* the revision row is committed (see
  ``library_rag.scan._process_file``), so a concurrent scan cannot lose
  freshly-archived books.
* Dry-run is the default: candidates are reported, nothing is deleted.
  ``--execute`` deletes, then removes the directories that became empty.

GC never touches source roots, the state DB, Qdrant storage, or the scratch
root.
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

__all__ = ["GcCandidate", "GcError", "GcReport", "run_gc"]

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LOG_NAME = re.compile(r"^(\d+)\.attempt(\d+)\.log$")


class GcError(RuntimeError):
    """Raised when GC cannot run safely (e.g. a worker is running)."""


@dataclass(frozen=True)
class GcCandidate:
    """One collectible store object. ``relpath`` is relative to the kind root."""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "grace_seconds": self.grace_seconds,
            "candidates": [asdict(c) for c in self.candidates],
            "deleted": self.deleted,
            "bytes_reclaimed": self.bytes_reclaimed,
            "directories_removed": self.directories_removed,
            "errors": self.errors,
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
) -> GcReport:
    """Collect (and, with ``execute=True``, delete) unreferenced store objects.

    See the module docstring for the reference rules and the safety guards.
    A :class:`GcError` is raised when a worker is running; per-file delete
    failures are recorded in the report instead of raising.
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
    if not execute:
        return report

    for c in candidates:
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
