"""Worker loop: claim durable jobs and run their pipeline stage (PRD §7).

M2 implemented the ``extract`` stage; M3 adds ``ocr`` (the selective Tesseract
pass over routed pages, PRD §8C) and ``chunk`` (normalization + token-budget
chunking, PRD §8E). Contract:

* one job at a time per worker; the lease is heartbeated between units
  (inject :meth:`Jobs.heartbeat` as the stage's ``on_progress``), so a
  long book cannot expire its own lease;
* ``StaleLeaseError`` from any fenced write means another worker owns the job
  now — stop touching it and let the new owner finish (at-least-once work,
  idempotent effects);
* ``ExtractionFailure`` categories classify permanence
  (encrypted/corrupt/ocr_unavailable/... fail permanently until explicit
  requeue), anything else is transient and gets backoff/retry;
* a job whose prerequisites are not ready yet (a chunk job waiting for the
  OCR jobs of its run) is *deferred* (:meth:`Jobs.defer`) — a fenced,
  non-consuming retry that does not burn attempts;
* OCR is a no-op when the unit already carries OCR output for the current
  engine settings, so a crash between the artifact commit and the job commit
  reclaims a job that skips Tesseract entirely;
* on worker start, :func:`_reconcile_chunks` re-enqueues the chunk job of any
  succeeded run whose stored fingerprint no longer matches (the M2→M3
  migration, a lost chunk job row, or a downstream state wipe).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from typing import Any

import pymupdf

from .archive import archive_path_for
from .catalog import Format
from .chunking import UnitInput, chunk_units
from .config import Config
from .db import Database
from .extraction import (
    ExtractionFailure,
    ExtractorCtx,
    commit_unit_artifact,
    extract_epub,
    extract_pdf,
    fail_run,
    is_permanent,
    load_unit_artifact,
    ocr_page,
    refresh_unit_artifact,
    set_unit_ocr,
)
from .extraction.epub import parser_version as epub_parser_version
from .extraction.pdf import parser_version as pdf_parser_version
from .identity import extraction_key, make_task_key, unit_id_for, units_fingerprint
from .jobs import Claimed, Jobs, StaleLeaseError
from .normalization import normalize_unit, unit_removed_ranges
from .scan import STAGE_CHUNK, STAGE_EXTRACT, STAGE_OCR

__all__ = [
    "DEFAULT_LEASE_TTL",
    "build_ctx",
    "chunk_fingerprint_for_run",
    "run_worker",
    "units_fingerprint_for_run",
    "worker_name",
]

DEFAULT_LEASE_TTL = 300.0
# How long a not-ready job (chunk waiting on OCR) waits before being re-claimed.
_DEFER_DELAY = 15.0
# Job states that still count as "in flight" for prerequisite checks.
_OPEN_STATES = "('pending', 'running', 'retryable_failed')"
# Job states that are terminal: the pipeline may move on past them.
_SETTLED_STATES = "('succeeded', 'permanent_failed', 'cancelled')"


def worker_name() -> str:
    return f"extract-{os.uname().nodename}-{os.getpid()}"


def units_fingerprint_for_run(db: Database, run_id: str) -> str:
    """Content fingerprint of a run's units: (position, artifact sha) pairs."""
    rows = db.query(
        "SELECT position, artifact_sha256 FROM source_units WHERE run_id = ?", (run_id,)
    )
    return units_fingerprint([(int(r["position"]), r["artifact_sha256"]) for r in rows])


def chunk_fingerprint_for_run(db: Database, run_id: str, cfg: Config) -> str:
    """Pipeline fingerprint stored in ``extraction_runs.chunk_fingerprint``.

    The units' content plus the settings that define what chunking will do to
    it: any OCR/normalization artifact rewrite or a normalization/chunking
    settings change yields a new fingerprint, so the chunk job re-runs exactly
    when its inputs changed (PRD §6/§8E).
    """
    units_fp = units_fingerprint_for_run(db, run_id)
    canonical = (
        f"{units_fp}:{cfg.extraction.normalization.settings_sha()}:"
        f"{cfg.chunking.settings_sha()}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rekey_chunk_job(db: Database, job: Claimed, run_id: str, fingerprint: str) -> None:
    """Point the job row at the fingerprint of the state it actually chunked.

    The enqueue-time key can carry a *pre*-OCR fingerprint: OCR commits new
    unit artifacts, changing the units' fingerprint after the chunk job was
    queued. Without the re-key, every later replay's idempotent enqueue
    (fingerprint recomputed post-OCR) would miss the settled row and insert a
    fresh duplicate job. The guard never violates UNIQUE(task_key): when
    another row already holds the target key the update is skipped.
    """
    key = make_task_key(STAGE_CHUNK, run_id, fingerprint)
    db.execute(
        "UPDATE jobs SET task_key = ? WHERE job_id = ? AND task_key != ? "
        "AND NOT EXISTS (SELECT 1 FROM jobs j2 "
        " WHERE j2.task_key = ? AND j2.job_id != ?)",
        (key, job.job_id, key, key, job.job_id),
    )


def build_ctx(
    db: Database,
    cfg: Config,
    rev: Any,
    on_progress: Callable[[], None] | None,
    jobs: Jobs | None = None,
) -> ExtractorCtx:
    """Assemble the extractor context for a revision row (run id is keyed on
    source hash + parser version + stage settings, PRD §6).

    When *jobs* is given the M3 pipeline hooks are wired: each page unit
    routed to OCR enqueues its (idempotent) OCR job, and extraction
    completion enqueues the run's chunk job.
    """
    fmt = Format(rev["format"])
    parser = pdf_parser_version() if fmt is Format.PDF else epub_parser_version()
    run_id = extraction_key(rev["sha256"], parser, cfg.extraction.settings_sha())
    enqueue_ocr: Callable[[int], None] | None = None
    on_extract_done: Callable[[], None] | None = None
    if jobs is not None:
        queue = jobs
        rev_sha = rev["sha256"]

        def enqueue_ocr(position: int) -> None:
            queue.enqueue(
                make_task_key(STAGE_OCR, run_id, rev_sha, f"page:{position}"),
                STAGE_OCR,
                input_id=run_id,
                input_version=rev_sha,
                range_spec=f"page:{position}",
            )

        def on_extract_done() -> None:
            fp = chunk_fingerprint_for_run(db, run_id, cfg)
            queue.enqueue(
                make_task_key(STAGE_CHUNK, run_id, fp),
                STAGE_CHUNK,
                input_id=run_id,
                input_version=rev_sha,
            )

    return ExtractorCtx(
        db=db,
        artifact_root=cfg.paths.artifact_root,
        rev=dict(rev),
        run_id=run_id,
        settings=cfg.extraction,
        source_path=archive_path_for(cfg.paths.archive_root, rev["sha256"]),
        on_progress=on_progress,
        enqueue_ocr=enqueue_ocr,
        on_extract_done=on_extract_done,
    )


def _run_extract(db: Database, cfg: Config, jobs: Jobs, job: Claimed, lease_ttl: float) -> bool:
    """Run one claimed extraction job.

    Returns True when the job reached a terminal state (succeeded, or failed
    and recorded); False when this worker lost the lease and must stop.
    """
    rev = db.query_one("SELECT * FROM source_revisions WHERE rev_id = ?", (job.input_id,))
    if rev is None:
        jobs.fail(job, "missing_source", f"revision {job.input_id} not in catalog", transient=False)
        return True
    ctx = build_ctx(
        db, cfg, rev, on_progress=lambda: jobs.heartbeat(job, lease_ttl), jobs=jobs
    )
    extract = extract_pdf if Format(rev["format"]) is Format.PDF else extract_epub
    try:
        units = extract(ctx)
    except StaleLeaseError:
        return False  # another worker owns this job now; touch nothing
    except ExtractionFailure as exc:
        jobs.fail(job, exc.category, exc.detail, transient=not is_permanent(exc.category))
        fail_run(db, ctx.run_id, exc.category, exc.detail)
        return True
    except Exception as exc:
        jobs.fail(job, "worker_error", f"{type(exc).__name__}: {exc}", transient=True)
        return True
    jobs.succeed(job, json.dumps({"run_id": ctx.run_id, "units": units}, sort_keys=True))
    return True


def _run_ocr(db: Database, cfg: Config, jobs: Jobs, job: Claimed, lease_ttl: float) -> bool:
    """Run one claimed OCR job (one page unit)."""
    spec = job.range_spec or ""
    if not spec.startswith("page:"):
        jobs.fail(job, "invalid_range", f"OCR job range {spec!r} is not a page", transient=False)
        return True
    try:
        page = int(spec.removeprefix("page:"))
    except ValueError:
        jobs.fail(job, "invalid_range", f"OCR job range {spec!r} is not a page", transient=False)
        return True
    run_id = job.input_id or ""
    run = db.query_one("SELECT * FROM extraction_runs WHERE run_id = ?", (run_id,))
    if run is None:
        jobs.fail(job, "missing_run", f"extraction run {run_id} not found", transient=False)
        return True
    if run["state"] == "running":
        jobs.defer(job, _DEFER_DELAY)  # extraction of the run is still in flight
        return True
    # A failed run keeps its already-extracted units; OCRing those pages is
    # still useful, so only "running" defers here.
    rev = db.query_one("SELECT * FROM source_revisions WHERE rev_id = ?", (run["rev_id"],))
    if rev is None:
        jobs.fail(job, "missing_source", f"revision {run['rev_id']} not in catalog", transient=False)
        return True
    ctx = build_ctx(db, cfg, rev, on_progress=lambda: jobs.heartbeat(job, lease_ttl))
    unit_id = unit_id_for(run_id, "page", page)
    unit_row = db.query_one(
        "SELECT * FROM source_units WHERE unit_id = ? AND run_id = ?", (unit_id, run_id)
    )
    if unit_row is None:
        jobs.fail(
            job, "missing_unit", f"page {page} has no unit in run {run_id}", transient=False
        )
        return True
    try:
        payload = load_unit_artifact(
            cfg.paths.artifact_root, rev["rev_id"], unit_id, unit_row["artifact_sha256"]
        )
    except (ValueError, OSError) as exc:
        jobs.fail(job, "missing_unit", f"unit artifact unreadable: {exc}", transient=False)
        return True

    ocr_sha = cfg.extraction.ocr.settings_sha()
    # Crash between the artifact commit and the job commit (or a double
    # claim): the unit already has OCR output for these settings -> skip
    # the expensive Tesseract pass entirely.
    if unit_row["ocr_state"] == "done" and payload.get("ocr", {}).get("settings_sha") == ocr_sha:
        jobs.succeed(job, json.dumps({"unit_id": unit_id, "noop": True}, sort_keys=True))
        return True

    try:
        jobs.heartbeat(job, lease_ttl)  # the Tesseract call can outlast a lease
        doc = pymupdf.open(str(ctx.source_path))  # type: ignore[no-untyped-call]
        with doc:
            result = ocr_page(
                doc, page, cfg.extraction.ocr, cfg.paths.scratch_root / "ocr" / run_id
            )
        payload["ocr"] = {**result, "settings_sha": ocr_sha}
        # Artifact first, then the row — the standard commit ordering, so a
        # crash leaves a consistent (row points at the new artifact) state.
        new_sha = commit_unit_artifact(ctx, unit_id, payload)
        char_count = len("".join(result["text"].split()))
        set_unit_ocr(
            ctx,
            unit_id=unit_id,
            artifact_relpath=f"extract/{rev['rev_id']}/{unit_id}.json.gz",
            artifact_sha256=new_sha,
            char_count=char_count,
            state="done",
        )
        jobs.succeed(
            job, json.dumps({"unit_id": unit_id, "chars": char_count}, sort_keys=True)
        )
    except StaleLeaseError:
        return False
    except ExtractionFailure as exc:
        if is_permanent(exc.category):
            # Record the failed attempt on the unit (keeps the pre-OCR
            # artifact) so the chunk pass falls back to the native text.
            set_unit_ocr(
                ctx,
                unit_id=unit_id,
                artifact_relpath=str(unit_row["artifact_relpath"]),
                artifact_sha256=str(unit_row["artifact_sha256"]),
                char_count=int(unit_row["char_count"]),
                state="failed",
            )
        jobs.fail(job, exc.category, exc.detail, transient=not is_permanent(exc.category))
        return True
    except Exception as exc:
        jobs.fail(job, "worker_error", f"{type(exc).__name__}: {exc}", transient=True)
        return True
    return True


def _effective_text(row: Any, payload: dict[str, Any]) -> str:
    """The unit's searchable source text: OCR output when present, else the
    native layer (page text, or a section's paragraphs joined)."""
    if row["kind"] == "page":
        if row["ocr_state"] == "done" and payload.get("ocr"):
            return str(payload["ocr"].get("text", ""))
        return str(payload.get("text", ""))
    paragraphs = payload.get("paragraphs") or []
    return "\n\n".join(str(p.get("text", "")) for p in paragraphs)


def _spans_json(spans: list[Any]) -> str:
    return json.dumps(
        [
            {
                "unit_id": s.unit_id,
                "source_start": s.source_start,
                "source_end": s.source_end,
                "bbox": s.bbox,
            }
            for s in spans
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def _run_chunk(db: Database, cfg: Config, jobs: Jobs, job: Claimed, lease_ttl: float) -> bool:
    """Run one claimed chunk job: normalize the run's units (idempotent
    rewrite into the unit artifacts), chunk the token stream, replace the
    chunks table rows, and record the pipeline fingerprint."""
    run_id = job.input_id or ""
    run = db.query_one("SELECT * FROM extraction_runs WHERE run_id = ?", (run_id,))
    if run is None:
        jobs.fail(job, "missing_run", f"extraction run {run_id} not found", transient=False)
        return True
    if run["state"] == "running":
        jobs.defer(job, _DEFER_DELAY)
        return True
    if run["state"] == "failed":
        open_extract = db.query_one(
            "SELECT 1 AS x FROM jobs WHERE stage = ? AND input_id = ? AND state IN "
            + _OPEN_STATES,
            (STAGE_EXTRACT, run["rev_id"]),
        )
        if open_extract is not None:
            jobs.defer(job, _DEFER_DELAY)  # a retried extract may still finish it
            return True
        jobs.fail(job, "run_failed", f"extraction run {run_id} failed", transient=False)
        return True
    open_ocr = db.query_one(
        "SELECT 1 AS x FROM jobs WHERE stage = ? AND input_id = ? AND state NOT IN "
        + _SETTLED_STATES,
        (STAGE_OCR, run_id),
    )
    if open_ocr is not None:
        jobs.defer(job, _DEFER_DELAY)  # wait until every OCR job settled
        return True
    rev = db.query_one("SELECT * FROM source_revisions WHERE rev_id = ?", (run["rev_id"],))
    if rev is None:
        jobs.fail(job, "missing_source", f"revision {run['rev_id']} not in catalog", transient=False)
        return True

    current = chunk_fingerprint_for_run(db, run_id, cfg)
    if run["chunk_fingerprint"] == current:
        n_chunks = db.query_one(
            "SELECT COUNT(*) AS n FROM chunks WHERE run_id = ?", (run_id,)
        )
        n_units = db.query_one(
            "SELECT COUNT(*) AS n FROM source_units WHERE run_id = ?", (run_id,)
        )
        assert n_chunks is not None and n_units is not None
        if int(n_chunks["n"] or 0) > 0 or int(n_units["n"] or 0) == 0:
            _rekey_chunk_job(db, job, run_id, current)
            jobs.succeed(
                job,
                json.dumps(
                    {"run_id": run_id, "chunks": int(n_chunks["n"] or 0), "noop": True},
                    sort_keys=True,
                ),
            )
            return True
        # Fingerprint matches but the chunks vanished: fall through and rebuild.

    ctx = build_ctx(db, cfg, rev, on_progress=lambda: jobs.heartbeat(job, lease_ttl))
    try:
        unit_rows = db.query(
            "SELECT unit_id, kind, position, ocr_state, artifact_sha256 "
            "FROM source_units WHERE run_id = ? ORDER BY position",
            (run_id,),
        )
        if not unit_rows:
            ts = time.time()
            with db.transaction():
                db.execute("DELETE FROM chunks WHERE run_id = ?", (run_id,))
                db.execute(
                    "UPDATE extraction_runs SET chunk_fingerprint = ?, updated_at = ? "
                    "WHERE run_id = ?",
                    (current, ts, run_id),
                )
            _rekey_chunk_job(db, job, run_id, current)
            jobs.succeed(
                job, json.dumps({"run_id": run_id, "chunks": 0, "fingerprint": current}, sort_keys=True)
            )
            return True

        payloads: list[dict[str, Any]] = []
        for row in unit_rows:
            try:
                payloads.append(
                    load_unit_artifact(
                        cfg.paths.artifact_root, rev["rev_id"], row["unit_id"],
                        row["artifact_sha256"],
                    )
                )
            except (ValueError, OSError) as exc:
                jobs.fail(job, "missing_unit", f"unit artifact unreadable: {exc}", transient=False)
                return True

        texts = [_effective_text(row, p) for row, p in zip(unit_rows, payloads, strict=True)]
        norm = cfg.extraction.normalization
        norm_sha = norm.settings_sha()
        if any(p.get("normalization", {}).get("settings_sha") != norm_sha for p in payloads):
            cross_unit = any(row["kind"] == "page" for row in unit_rows)
            removed_list = unit_removed_ranges(texts, norm, cross_unit=cross_unit)
            for i, (row, payload) in enumerate(zip(unit_rows, payloads, strict=True)):
                if payload.get("normalization", {}).get("settings_sha") == norm_sha:
                    continue
                normalized = normalize_unit(texts[i], removed_list[i], norm)
                payload["normalization"] = {
                    "settings_sha": norm_sha,
                    "searchable": normalized.searchable,
                    "spans": normalized.spans,
                }
                new_sha = commit_unit_artifact(ctx, row["unit_id"], payload)
                refresh_unit_artifact(
                    ctx,
                    unit_id=row["unit_id"],
                    artifact_relpath=f"extract/{rev['rev_id']}/{row['unit_id']}.json.gz",
                    artifact_sha256=new_sha,
                )
            # The artifacts just changed: recompute BEFORE comparing/record.
            current = chunk_fingerprint_for_run(db, run_id, cfg)

        inputs = [
            UnitInput(
                unit_id=row["unit_id"],
                searchable=str(payload["normalization"]["searchable"]),
                spans=[list(map(int, s)) for s in payload["normalization"]["spans"]],
                title=payload.get("title"),
                word_boxes=(
                    tuple(
                        (str(w["text"]), [float(c) for c in w["box"]])
                        for w in (payload["ocr"].get("words") or [])
                    )
                    if row["ocr_state"] == "done" and payload.get("ocr")
                    else ()
                ),
            )
            for row, payload in zip(unit_rows, payloads, strict=True)
        ]
        results = chunk_units(inputs, cfg.chunking, run_id)

        ts = time.time()
        with db.transaction():
            db.execute("DELETE FROM chunks WHERE run_id = ?", (run_id,))
            for position, result in enumerate(results):
                prev_id = results[position - 1].chunk_id if position > 0 else None
                next_id = results[position + 1].chunk_id if position + 1 < len(results) else None
                db.execute(
                    """
                    INSERT INTO chunks
                        (chunk_id, run_id, rev_id, position, text, token_count, title,
                         spans, prev_chunk_id, next_chunk_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.chunk_id, run_id, rev["rev_id"], position, result.text,
                        result.token_count, result.title, _spans_json(result.spans),
                        prev_id, next_id, ts,
                    ),
                )
            db.execute(
                "UPDATE extraction_runs SET chunk_fingerprint = ?, updated_at = ? "
                "WHERE run_id = ?",
                (current, ts, run_id),
            )
        _rekey_chunk_job(db, job, run_id, current)
        jobs.succeed(
            job,
            json.dumps(
                {"run_id": run_id, "chunks": len(results), "fingerprint": current},
                sort_keys=True,
            ),
        )
    except StaleLeaseError:
        return False
    except Exception as exc:
        jobs.fail(job, "worker_error", f"{type(exc).__name__}: {exc}", transient=True)
        return True
    return True


def _reconcile_chunks(db: Database, cfg: Config) -> None:
    """Re-enqueue the chunk job of every succeeded run whose stored
    fingerprint does not match (or was never set — runs extracted under M2).

    This is the durable answer to two lost-enqueue windows: the M2→M3
    migration (succeeded runs with no chunk job at all) and a chunk job row
    lost after its effects. Idempotent: enqueue is INSERT OR IGNORE on the
    task key, and the chunk job itself no-ops when the fingerprint matches.
    Runs with in-flight OCR jobs are skipped (their chunk job will defer).
    """
    jobs = Jobs(db)
    runs = db.query(
        "SELECT run_id, rev_id, chunk_fingerprint "
        "FROM extraction_runs WHERE state = 'succeeded'"
    )
    for row in runs:
        run_id = row["run_id"]
        if db.query_one(
            "SELECT 1 AS x FROM jobs WHERE stage = ? AND input_id = ? AND state NOT IN "
            + _SETTLED_STATES,
            (STAGE_OCR, run_id),
        ) is not None:
            continue
        current = chunk_fingerprint_for_run(db, run_id, cfg)
        if row["chunk_fingerprint"] == current:
            continue
        rev = db.query_one(
            "SELECT sha256 FROM source_revisions WHERE rev_id = ?", (row["rev_id"],)
        )
        if rev is None:
            continue
        jobs.enqueue(
            make_task_key(STAGE_CHUNK, run_id, current),
            STAGE_CHUNK,
            input_id=run_id,
            input_version=rev["sha256"],
        )


def run_worker(
    db: Database,
    cfg: Config,
    *,
    once: bool = False,
    lease_ttl: float = DEFAULT_LEASE_TTL,
    poll_delay: float = 1.0,
    stop_event: Callable[[], bool] | None = None,
    reconcile_on_start: bool = True,
) -> int:
    """Claim and run jobs until the queue is drained (*once*) or *stop_event*
    becomes true. Returns the number of jobs brought to a handled state
    (terminal, or deferred — deferrals keep the *once* drain loop from
    busy-spinning on a not-ready job).

    Stopping mid-loop is graceful by construction: at most the current job's
    current unit is in flight; everything already committed is durable and
    resumable (PRD §7 SIGTERM).
    """
    name = worker_name()
    jobs = Jobs(db)
    if reconcile_on_start:
        _reconcile_chunks(db, cfg)
    handlers = {
        STAGE_EXTRACT: _run_extract,
        STAGE_OCR: _run_ocr,
        STAGE_CHUNK: _run_chunk,
    }
    completed = 0
    while True:
        if stop_event is not None and stop_event():
            break
        job = jobs.claim(name, lease_ttl)
        if job is None:
            if once:
                break
            if poll_delay > 0:
                _sleep_interruptible(poll_delay, stop_event)
            continue
        handler = handlers.get(job.stage)
        if handler is None:
            jobs.fail(job, "unknown_stage", f"stage {job.stage!r} has no handler", transient=False)
            completed += 1  # permanent_failed is a terminal state
            continue
        if handler(db, cfg, jobs, job, lease_ttl):
            completed += 1
    return completed


def _sleep_interruptible(delay: float, stop_event: Callable[[], bool] | None) -> None:
    end = time.monotonic() + delay
    while time.monotonic() < end:
        if stop_event is not None and stop_event():
            return
        time.sleep(min(0.2, max(0.0, end - time.monotonic())))
