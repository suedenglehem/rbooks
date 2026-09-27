"""The manual-selection ledger (M10).

The operator ticked books in the Browse view and submitted them for immediate
processing (``POST /browse/process``). The ledger is the *final* set of
manually selected books, not a submission log: one row per normalized path,
and a resubmission refreshes the row (latest outcome wins), so the Rag page
shows exactly the books the operator picked and nothing else.

On top of the ledger rows, :func:`manual_selection_status` joins live
pipeline state so each book carries its current position:

* ``queued``      -- registered, extract job waiting for a worker
* ``processing``  -- a pipeline job is running (the stage is reported), or
                     extraction finished and the downstream stages are due
* ``published``   -- extraction succeeded and the revision has an active
                     publication
* ``failed``      -- a pipeline job failed (stage + detail are reported)
* ``parked``      -- a job is parked by the book budget
                     (``error_detail == book_budget_parked``)
* ``superseded``  -- the revision was replaced by a newer one
* ``unknown``     -- the file never registered (missing/invalid/deferred),
                     so there is no pipeline to join

The join uses the worker's ``jobs.input_id`` mapping: extract/resume are
keyed by ``rev_id``; ocr/chunk/embed/publish by the revision's most recent
``extraction_runs.run_id``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import Database
from .jobs import BOOK_BUDGET_PARKED

__all__ = [
    "ManualSelection",
    "manual_selection_status",
    "record_manual_selections",
    "record_submission",
]

# Which job stage is keyed by which identity in ``jobs.input_id`` (worker.py).
_REV_KEYED_STAGES = ("extract", "resume")
_RUN_KEYED_STAGES = ("ocr", "chunk", "embed", "publish")
_ALL_STAGES = _REV_KEYED_STAGES + _RUN_KEYED_STAGES

# Per (stage, input) aggregation priority when a book has several jobs in the
# same stage (e.g. one job per OCR page): the most urgent state wins, so a
# running page outranks ten succeeded ones and a failure outranks anything.
_STATE_RANK = {
    "running": 4,
    "pending": 3,
    "retryable_failed": 2,
    "permanent_failed": 2,
    "cancelled": 2,
    "succeeded": 1,
}


@dataclass(frozen=True)
class ManualSelection:
    """One book the operator submitted from the Browse view.

    ``sha256``/``rev_id``/``doc_id`` are the links to what the file
    registered; they are None for files that never registered (missing,
    invalid, deferred) and are preserved across resubmissions that carry
    none (see :func:`record_manual_selections`).
    """

    path: str
    outcome: str
    sha256: str | None = None
    rev_id: str | None = None
    doc_id: str | None = None


def record_manual_selections(
    db: Database, selections: Sequence[ManualSelection], now: float | None = None
) -> None:
    """Upsert ledger rows (identity: normalized path).

    A resubmission refreshes ``outcome`` and ``submitted_at``; the link
    columns keep their previous value when the new submission carries none,
    so a file that vanished after registration still points at its book.
    """
    if not selections:
        return
    ts = time.time() if now is None else now
    with db.transaction():
        for sel in selections:
            db.execute(
                """
                INSERT INTO manual_selections (path, outcome, sha256, rev_id, doc_id, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    outcome      = excluded.outcome,
                    sha256       = COALESCE(excluded.sha256, manual_selections.sha256),
                    rev_id       = COALESCE(excluded.rev_id, manual_selections.rev_id),
                    doc_id       = COALESCE(excluded.doc_id, manual_selections.doc_id),
                    submitted_at = excluded.submitted_at
                """,
                (sel.path, sel.outcome, sel.sha256, sel.rev_id, sel.doc_id, ts),
            )


def record_submission(
    db: Database, outcomes: Mapping[str, object], now: float | None = None
) -> None:
    """Record the outcome of one ``process_explicit_paths`` call.

    *outcomes* maps normalized absolute path → outcome (the
    ``ProcessStatus`` values). Links are resolved from ``scan_state``
    (path → sha256/rev_id, written by the scan for every registered file)
    and ``source_revisions`` (rev_id → doc_id).
    """
    if not outcomes:
        return
    marks = ", ".join("?" * len(outcomes))
    state_rows = db.query(
        f"SELECT path, sha256, rev_id FROM scan_state WHERE path IN ({marks})",
        tuple(outcomes),
    )
    links = {r["path"]: (r["sha256"], r["rev_id"]) for r in state_rows}
    rev_ids = sorted({rev for _sha, rev in links.values() if rev is not None})
    docs: dict[str, str] = {}
    if rev_ids:
        rm = ", ".join("?" * len(rev_ids))
        docs = {
            r["rev_id"]: r["doc_id"]
            for r in db.query(
                f"SELECT rev_id, doc_id FROM source_revisions WHERE rev_id IN ({rm})",
                tuple(rev_ids),
            )
        }
    selections = []
    for path, status in outcomes.items():
        sha, rev = links.get(path, (None, None))
        selections.append(
            ManualSelection(
                path=path,
                outcome=str(status),
                sha256=sha,
                rev_id=rev,
                doc_id=docs.get(rev) if rev is not None else None,
            )
        )
    record_manual_selections(db, selections, now=now)


def manual_selection_status(db: Database, *, limit: int = 200) -> list[dict[str, Any]]:
    """Ledger rows (newest first) joined with live pipeline status.

    The payload is JSON-friendly for the Rag page: one dict per selected
    book with the ledger facts (path, title, outcome, submitted_at, links)
    and the derived ``status``/``stage``/``error``.
    """
    rows = db.query(
        """
        SELECT path, outcome, sha256, rev_id, doc_id, submitted_at
        FROM manual_selections
        ORDER BY submitted_at DESC, path ASC
        LIMIT ?
        """,
        (limit,),
    )
    if not rows:
        return []

    rev_ids = sorted({r["rev_id"] for r in rows if r["rev_id"] is not None})
    marks = ", ".join("?" * len(rev_ids))
    revs: dict[str, dict[str, Any]] = {}
    latest_run: dict[str, str] = {}
    active_pub_revs: set[str] = set()
    if rev_ids:
        for r in db.query(
            f"""
            SELECT rev_id, doc_id, sha256, size_bytes, format, first_path, is_active
            FROM source_revisions WHERE rev_id IN ({marks})
            """,
            tuple(rev_ids),
        ):
            revs[r["rev_id"]] = dict(r)
        # Most recent run per revision wins (ASC order, later rows overwrite).
        for r in db.query(
            f"""
            SELECT rev_id, run_id FROM extraction_runs
            WHERE rev_id IN ({marks}) ORDER BY created_at ASC, run_id ASC
            """,
            tuple(rev_ids),
        ):
            latest_run[r["rev_id"]] = r["run_id"]
        active_pub_revs = {
            r["rev_id"]
            for r in db.query(
                f"""
                SELECT rev_id FROM publications
                WHERE rev_id IN ({marks}) AND state = 'active'
                """,
                tuple(rev_ids),
            )
        }

    jobs: dict[tuple[str, str], dict[str, Any]] = {}
    inputs = sorted(set(rev_ids) | set(latest_run.values()))
    if inputs:
        im = ", ".join("?" * len(inputs))
        sm = ", ".join("?" * len(_ALL_STAGES))
        for r in db.query(
            f"""
            SELECT job_id, stage, input_id, state, error_category, error_detail
            FROM jobs WHERE input_id IN ({im}) AND stage IN ({sm})
            """,
            tuple(inputs) + _ALL_STAGES,
        ):
            key = (r["stage"], r["input_id"])
            cur = jobs.get(key)
            if cur is None or _STATE_RANK.get(r["state"], 0) > _STATE_RANK.get(cur["state"], 0):
                jobs[key] = dict(r)

    out: list[dict[str, Any]] = []
    for r in rows:
        rev = r["rev_id"]
        revinfo = revs.get(rev) if rev is not None else None
        if revinfo is None:
            # Never registered (missing/invalid/deferred) — nothing to join.
            status, stage, error = "unknown", None, None
            title = Path(r["path"]).stem
            fmt: str | None = None
            size: int | None = None
        else:
            title = Path(str(revinfo["first_path"])).stem
            fmt = revinfo["format"]
            size = revinfo["size_bytes"]
            run_id = latest_run.get(rev) if rev is not None else None
            status, stage, error = _derive_status(
                str(rev), revinfo, run_id, jobs, active_pub_revs
            )
        out.append(
            {
                "path": r["path"],
                "title": title,
                "outcome": r["outcome"],
                "submitted_at": r["submitted_at"],
                "rev_id": rev,
                "doc_id": r["doc_id"],
                "format": fmt,
                "size_bytes": size,
                "status": status,
                "stage": stage,
                "error": error,
            }
        )
    return out


def _derive_status(
    rev_id: str,
    revinfo: Mapping[str, Any],
    run_id: str | None,
    jobs: Mapping[tuple[str, str], Mapping[str, Any]],
    active_pub_revs: set[str],
) -> tuple[str, str | None, str | None]:
    """(status, stage, error) for one registered revision.

    Precedence: a failed job (the most recent one) wins over anything else —
    that is the current blocker; then an in-flight job; then a pending one;
    then the terminal states of a fully succeeded extract.
    """
    keys = {stage: (stage, rev_id) for stage in _REV_KEYED_STAGES}
    if run_id is not None:
        keys.update({stage: (stage, run_id) for stage in _RUN_KEYED_STAGES})
    by_stage = {stage: jobs.get(key) for stage, key in keys.items()}
    present = {stage: j for stage, j in by_stage.items() if j is not None}

    failed = [
        (int(j["job_id"]), stage, j)
        for stage, j in present.items()
        if j["state"] in ("retryable_failed", "permanent_failed", "cancelled")
    ]
    if failed:
        _jid, stage, j = max(failed, key=lambda t: t[0])
        if j["error_detail"] == BOOK_BUDGET_PARKED:
            return "parked", stage, j["error_category"]
        return "failed", stage, j["error_detail"] or j["error_category"]

    running = [stage for stage, j in present.items() if j["state"] == "running"]
    if running:
        return "processing", running[0], None
    if any(j["state"] == "pending" for j in present.values()):
        return "queued", None, None

    extract = present.get("extract")
    if extract is not None and extract["state"] == "succeeded":
        if not revinfo["is_active"]:
            return "superseded", None, None
        if rev_id in active_pub_revs:
            return "published", None, None
        # Extract done, downstream stages not yet observed: still on its way.
        return "processing", None, None
    return "unknown", None, None
