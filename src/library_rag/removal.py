"""Explicit book removal (PRD §14, M7 slice 3).

Removes one document (a book) from the library. A target is either a source
path or a ``doc_id``; both resolve to the same deterministic identity, so
removal is repeatable (a second run reports "not found").

Order of operations for ``execute``:

1. **Index points** — every point whose payload ``doc_id`` matches, across
   *all* generations (active and superseded), deleted via the Qdrant filter.
   Deleted *before* the catalog: if the process dies afterwards, the catalog
   still names the book (the operator re-runs removal, which is idempotent),
   where deleting the catalog first would leave invisible orphan points that
   nothing else can reach (GC does not touch Qdrant storage).
2. **Catalog rows** — one FK-safe transaction, children first: ``scan_state``
   (so a re-scan of a re-added file registers fresh instead of fast-checking
   against a dead revision), ``path_aliases``, ``publications``,
   ``index_generations``, ``chunks``, ``source_units``, ``embedding_batches``,
   ``extraction_runs``, ``source_revisions``, ``documents``. Open jobs
   (pending / retryable_failed) for this document's revisions or runs are
   cancelled in the same transaction.
3. **Archive object** — after the transaction, the revision's archive object
   is unlinked only if no ``source_revisions`` row still references its SHA
   (content-anchored identity makes a shared SHA impossible, but the check is
   what makes the delete safe). Emptied directories are left for ``gc``.

Deliberately *not* touched:

* **Source files** — the system never deletes or modifies sources. If the
  source reappears, the next scan re-registers the book from scratch (same
  bytes → same deterministic doc/rev ids).
* **Artifact files** (extract units, embedding checkpoints) — they become
  unreferenced and are reclaimed by ``gc --execute`` with its grace window.
* **Answers** — kept on purpose: their frozen evidence manifests are
  self-contained (PRD §12).
* **``sparse_corpus_stats``** — the epoch record survives; the next publish
  self-heals it via recompute.

Safety:

* Refuses while any job is ``running`` (a live worker may be mid-pipeline
  for this document) — same guard as :mod:`library_rag.gc`.
* Refuses when the document has index points but the Qdrant storage cannot be
  opened (a running worker holds the local storage lock); catalog-only
  removal is allowed when the document has no points.
* Dry-run is the default: the report carries before-counts and the archive
  objects that would be deleted. ``execute`` performs, then **verifies**:
  every catalog table filtered to the document (and revision/run set) must be
  empty, and the point count must be zero, or a :class:`RemovalError` is
  raised with the leftovers named.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .archive import archive_path_for
from .config import Config
from .db import Database
from .identity import normalize_path
from .indexing import FieldCond, IndexFilter, QdrantOps

__all__ = ["RemovalError", "RemoveReport", "remove_document", "resolve_target"]

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


class RemovalError(RuntimeError):
    """Raised when a document cannot be resolved or removed safely."""


@dataclass
class RemoveReport:
    """Before-counts (dry-run) or deletion results (execute) for one document."""

    target: str
    doc_id: str
    executed: bool
    points: int = 0
    publications: int = 0
    generations: int = 0
    extraction_runs: int = 0
    source_units: int = 0
    chunks: int = 0
    embedding_batches: int = 0
    revisions: int = 0
    aliases: int = 0
    scan_state_rows: int = 0
    jobs_cancelled: int = 0
    archive_objects: list[dict[str, Any]] = field(default_factory=list)
    archive_kept: list[str] = field(default_factory=list)
    verified: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_target(db: Database, target: str) -> str:
    """Resolve *target* (a ``doc_id`` or a source path) to a ``doc_id``.

    UUID-shaped targets are doc ids; anything else is treated as a path and
    resolved through ``path_aliases``. Raises :class:`RemovalError` when the
    target names no known document.
    """
    t = target.strip()
    if _UUID_RE.fullmatch(t):
        doc_id = t.lower()
        row = db.query_one("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
        if row is None:
            raise RemovalError(f"no document with doc_id {doc_id}")
        return doc_id
    norm = normalize_path(t)
    row = db.query_one("SELECT doc_id FROM path_aliases WHERE path = ?", (norm,))
    if row is None:
        raise RemovalError(f"no document registered for path {target}")
    return str(row["doc_id"])


def _count(db: Database, sql: str, params: Sequence[Any] = ()) -> int:
    row = db.query_one(sql, params)
    return int(row["n"]) if row is not None else 0


def _in(values: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """``(placeholder list, params)`` for ``col IN (...)`` (empty-safe callers)."""
    return f"({', '.join('?' * len(values))})", tuple(values)


def _running_jobs(db: Database) -> int:
    return _count(db, "SELECT COUNT(*) AS n FROM jobs WHERE state = 'running'")


def _doc_filter(doc_id: str) -> IndexFilter:
    return IndexFilter.all(FieldCond("doc_id", "eq", doc_id))


def _gather(
    db: Database, doc_id: str
) -> tuple[list[str], list[str], dict[str, int]]:
    """Return (rev_ids, run_ids, before-counts) for *doc_id*."""
    revs = [str(r["rev_id"]) for r in db.query("SELECT rev_id FROM source_revisions WHERE doc_id = ?", (doc_id,))]
    runs: list[str] = []
    if revs:
        ph, params = _in(revs)
        runs = [str(r["run_id"]) for r in db.query(f"SELECT run_id FROM extraction_runs WHERE rev_id IN {ph}", params)]
    counts: dict[str, int] = {
        "publications": _count(db, "SELECT COUNT(*) AS n FROM publications WHERE doc_id = ?", (doc_id,)),
        "revisions": len(revs),
        "aliases": _count(db, "SELECT COUNT(*) AS n FROM path_aliases WHERE doc_id = ?", (doc_id,)),
        "scan_state_rows": 0,
        "extraction_runs": len(runs),
        "source_units": 0,
        "chunks": 0,
        "embedding_batches": 0,
        "generations": 0,
    }
    if revs:
        ph, params = _in(revs)
        counts["scan_state_rows"] = _count(db, f"SELECT COUNT(*) AS n FROM scan_state WHERE rev_id IN {ph}", params)
        counts["source_units"] = _count(db, f"SELECT COUNT(*) AS n FROM source_units WHERE rev_id IN {ph}", params)
        counts["chunks"] = _count(db, f"SELECT COUNT(*) AS n FROM chunks WHERE rev_id IN {ph}", params)
        counts["generations"] = _count(db, f"SELECT COUNT(*) AS n FROM index_generations WHERE rev_id IN {ph}", params)
    if runs:
        ph, params = _in(runs)
        counts["embedding_batches"] = _count(db, f"SELECT COUNT(*) AS n FROM embedding_batches WHERE run_id IN {ph}", params)
    return revs, runs, counts


def _archive_plan(
    db: Database, revs_rows: list[Any], own_refs: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split this document's archive objects into (to_delete, kept).

    An object is deletable only if the recount of ``source_revisions`` rows
    for its SHA shows no reference beyond the document's own *own_refs* rows
    (own_refs is 1 in a dry run, 0 after the deletion transaction).
    """
    to_delete: list[dict[str, Any]] = []
    kept: list[str] = []
    for rev in sorted(revs_rows, key=lambda r: str(r["sha256"])):
        sha = str(rev["sha256"])
        refs = _count(db, "SELECT COUNT(*) AS n FROM source_revisions WHERE sha256 = ?", (sha,))
        if refs > own_refs:
            kept.append(sha)
        else:
            to_delete.append({"sha256": sha, "size_bytes": int(rev["size_bytes"])})
    return to_delete, kept


def remove_document(
    db: Database,
    cfg: Config,
    qdrant: QdrantOps | None,
    doc_id: str,
    *,
    execute: bool = False,
) -> RemoveReport:
    """Remove *doc_id*: index points, catalog rows, then its archive object.

    See the module docstring for the full contract. ``qdrant=None`` is
    accepted only when the document has no index points. Raises
    :class:`RemovalError` on unsafe preconditions or failed verification.
    """
    running = _running_jobs(db)
    if running:
        raise RemovalError(
            f"{running} job(s) are running; stop the worker before removal "
            "(pause, or wait for the in-flight job to finish)"
        )

    report = RemoveReport(target=doc_id, doc_id=doc_id, executed=execute)
    revs, runs, counts = _gather(db, doc_id)
    for key, value in counts.items():
        setattr(report, key, value)

    doc_filter = _doc_filter(doc_id)
    if qdrant is not None and qdrant.collection_exists():
        report.points = qdrant.count(doc_filter)
    if qdrant is None and report.publications > 0:
        raise RemovalError(
            "document has index points but Qdrant storage cannot be opened "
            "(a running worker may hold the local storage lock); stop the "
            "worker and retry"
        )

    revs_rows = list(
        db.query("SELECT sha256, size_bytes FROM source_revisions WHERE doc_id = ?", (doc_id,))
    )
    # Dry-run plan: which archive objects *would* be deleted. Identity is
    # content-anchored, so every revision SHA is referenced by exactly this
    # document; the recount after the transaction is what keeps it safe.
    if not execute:
        plan_delete, plan_kept = _archive_plan(db, revs_rows, own_refs=1)
        report.archive_objects = plan_delete
        report.archive_kept = plan_kept
        return report

    # 1. Points first (all generations, active and superseded), verified
    #    before the catalog is touched: a crash here leaves the catalog
    #    intact and the removal simply re-runs.
    if report.points > 0:
        assert qdrant is not None  # publications>0 forced a live client above
        qdrant.delete(doc_filter)
        leftover = qdrant.count(doc_filter)
        if leftover:
            raise RemovalError(
                f"point deletion verification failed: {leftover} point(s) remain for {doc_id}"
            )

    # 2. Catalog rows, one transaction, children before parents (FK ON).
    ts = time.time()
    job_ids: set[str] = set(revs) | set(runs)
    with db.transaction():
        if job_ids:
            ph, params = _in(sorted(job_ids))
            cur = db.execute(
                "UPDATE jobs SET state = 'cancelled', updated_at = ? "
                f"WHERE state IN ('pending','retryable_failed') AND input_id IN {ph}",
                (ts, *params),
            )
            report.jobs_cancelled = cur.rowcount
        if revs:
            ph, params = _in(revs)
            db.execute(f"DELETE FROM scan_state WHERE rev_id IN {ph}", params)
        # path_aliases and publications before index_generations: publications
        # holds an FK to gen_id (and path_aliases to rev_id/doc_id), so both
        # children must be gone first.
        db.execute("DELETE FROM path_aliases WHERE doc_id = ?", (doc_id,))
        db.execute("DELETE FROM publications WHERE doc_id = ?", (doc_id,))
        if revs:
            ph, params = _in(revs)
            db.execute(f"DELETE FROM index_generations WHERE rev_id IN {ph}", params)
            db.execute(f"DELETE FROM chunks WHERE rev_id IN {ph}", params)
            db.execute(f"DELETE FROM source_units WHERE rev_id IN {ph}", params)
        if runs:
            ph, params = _in(runs)
            db.execute(f"DELETE FROM embedding_batches WHERE run_id IN {ph}", params)
        # extraction_runs after every row holding an FK to it (index_generations,
        # chunks, source_units, embedding_batches) and before source_revisions,
        # which it references by rev_id: children before parents.
        if revs:
            ph, params = _in(revs)
            db.execute(f"DELETE FROM extraction_runs WHERE rev_id IN {ph}", params)
        db.execute("DELETE FROM source_revisions WHERE doc_id = ?", (doc_id,))
        db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    # 3. Archive objects: recount references now that the rows are gone.
    to_delete, kept = _archive_plan(db, revs_rows, own_refs=0)
    report.archive_kept = kept
    for obj in to_delete:
        p = archive_path_for(cfg.paths.archive_root, str(obj["sha256"]))
        try:
            if p.is_file():
                p.unlink()
            else:
                report.errors.append(f"archive {obj['sha256']}: missing on disk")
        except OSError as exc:
            report.errors.append(f"archive {obj['sha256']}: {exc}")
            continue
        report.archive_objects.append(obj)

    # 4. Verification: every table filtered to the document must be empty.
    leftovers: dict[str, int] = {
        "documents": _count(db, "SELECT COUNT(*) AS n FROM documents WHERE doc_id = ?", (doc_id,)),
        "path_aliases": _count(db, "SELECT COUNT(*) AS n FROM path_aliases WHERE doc_id = ?", (doc_id,)),
        "publications": _count(db, "SELECT COUNT(*) AS n FROM publications WHERE doc_id = ?", (doc_id,)),
    }
    if revs:
        ph, params = _in(revs)
        leftovers["source_revisions"] = _count(
            db, f"SELECT COUNT(*) AS n FROM source_revisions WHERE rev_id IN {ph}", params
        )
        leftovers["extraction_runs"] = _count(
            db, f"SELECT COUNT(*) AS n FROM extraction_runs WHERE rev_id IN {ph}", params
        )
        leftovers["source_units"] = _count(
            db, f"SELECT COUNT(*) AS n FROM source_units WHERE rev_id IN {ph}", params
        )
        leftovers["chunks"] = _count(
            db, f"SELECT COUNT(*) AS n FROM chunks WHERE rev_id IN {ph}", params
        )
        leftovers["scan_state"] = _count(
            db, f"SELECT COUNT(*) AS n FROM scan_state WHERE rev_id IN {ph}", params
        )
        leftovers["index_generations"] = _count(
            db, f"SELECT COUNT(*) AS n FROM index_generations WHERE rev_id IN {ph}", params
        )
    if runs:
        ph, params = _in(runs)
        leftovers["embedding_batches"] = _count(
            db, f"SELECT COUNT(*) AS n FROM embedding_batches WHERE run_id IN {ph}", params
        )
    if qdrant is not None and qdrant.collection_exists():
        leftover_points = qdrant.count(doc_filter)
        if leftover_points:
            leftovers["qdrant_points"] = leftover_points
    bad = {k: v for k, v in leftovers.items() if v}
    if bad:
        raise RemovalError(f"removal verification failed, rows remain: {bad}")
    report.verified = True
    return report
