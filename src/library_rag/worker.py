"""Worker loop: claim durable jobs and run their pipeline stage (PRD §7).

M2 implemented the ``extract`` stage; M3 added ``ocr`` (the selective
Tesseract pass over routed pages, PRD §8C) and ``chunk`` (normalization +
token-budget chunking, PRD §8E); M4 adds ``embed`` (checkpointed dense
encoding with bounded OOM halving, PRD §5/§8F) and ``publish`` (staged
upsert + verified visibility switch under the publish lock, PRD §8F).
Contract:

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
* embeddings are checkpointed batch-by-batch *before* the batch row is
  committed, so a re-run reuses every surviving batch and a corrupt
  checkpoint is detected, dropped, and re-encoded rather than replayed;
* publication is staged: points are upserted inactive and verified, the
  SQLite row switches under the publish lock, and a crash at any boundary is
  repaired by :func:`reconcile_publications` (called at worker start and at
  search start) — never by trusting the Qdrant flags;
* on worker start, :func:`_reconcile_chunks` re-enqueues the chunk job of any
  succeeded run whose stored fingerprint no longer matches (the M2→M3
  migration, a lost chunk job row, or a downstream state wipe) and
  :func:`_reconcile_index` closes the M4 gaps (a run whose checkpoints are
  missing for the current embedding configuration gets an embed job; a
  settled revision with no staged/active publication gets a publish job).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import pymupdf

from .archive import archive_path_for
from .catalog import Format
from .chunking import UnitInput, chunk_units
from .config import Config, ConfigError
from .db import Database
from .embeddings import (
    CheckpointCorruptError,
    Embedder,
    EmbeddingError,
    EmbeddingOOMError,
    ModelUnavailableError,
    checkpoint_path,
    corpus_stats_version_part,
    embedding_sha,
    encode_batch_oom,
    make_embedder,
    read_checkpoint,
    write_checkpoint,
)
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
from .identity import extraction_key, generation_id, make_task_key, unit_id_for, units_fingerprint
from .indexing import (
    PublicationError,
    QdrantOps,
    RealQdrantOps,
    acquire_publish_lock,
    publication_is_current,
    publish_generation,
    reconcile_publications,
    release_publish_lock,
)
from .jobs import Claimed, Jobs, StaleLeaseError
from .normalization import normalize_unit, unit_removed_ranges
from .scan import STAGE_CHUNK, STAGE_EMBED, STAGE_EXTRACT, STAGE_OCR, STAGE_PUBLISH

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


def _enqueue_embed_job(jobs: Jobs, run_id: str, fingerprint: str) -> None:
    """Idempotently enqueue the run's embedding job (the chunk stage's handoff).

    The task key carries the chunk fingerprint, so a re-chunk (new
    fingerprint) mints a new job and a replay of the same chunk job re-inserts
    nothing (INSERT OR IGNORE on the task key).
    """
    jobs.enqueue(
        make_task_key(STAGE_EMBED, run_id, fingerprint),
        STAGE_EMBED,
        input_id=run_id,
        input_version=fingerprint,
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
    completion enqueues the run's chunk job. The pilot page cap
    (``cfg.pilot.page_cap``, M6) is threaded into the run key and the
    extractor context so capped runs are isolated from full runs.
    """
    fmt = Format(rev["format"])
    parser = pdf_parser_version() if fmt is Format.PDF else epub_parser_version()
    # M6 pilot: cfg.pilot.page_cap (None in production) is hashed into the
    # extraction key, so capped and full runs of the same bytes are distinct.
    page_cap = cfg.pilot.page_cap
    run_id = extraction_key(rev["sha256"], parser, cfg.extraction.settings_sha(), page_cap)
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
        page_cap=page_cap,
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
            if int(n_chunks["n"] or 0) > 0:
                _enqueue_embed_job(jobs, run_id, current)
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
        if results:
            _enqueue_embed_job(jobs, run_id, current)
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


def _run_embed(
    db: Database,
    cfg: Config,
    jobs: Jobs,
    job: Claimed,
    lease_ttl: float,
    embedder: Embedder | None = None,
) -> bool:
    """Checkpoint the run's dense vectors (PRD §5/§8F), then hand off to publish.

    Vectors are persisted to a non-pickle artifact *before* the batch row is
    committed, so a crash never leaves a manifest row pointing at missing or
    unverified bytes; a corrupt artifact (row present, bytes failing
    validation) deletes the row and re-encodes that batch (PRD §14). GPU OOM
    halves the batch within the bounded retry policy; an unencodable chunk
    fails the job explicitly rather than looping forever. On success the
    publication job is enqueued — the stage that stages, verifies, and
    activates the index generation (PRD §8F).
    """
    run = db.query_one(
        "SELECT run_id, rev_id, state, chunk_fingerprint FROM extraction_runs WHERE run_id = ?",
        (job.input_id or "",),
    )
    if run is None:
        jobs.fail(job, "missing_run", f"run {job.input_id} not in catalog", transient=False)
        return True
    if run["state"] == "running":
        jobs.defer(job)
        return True
    if run["state"] == "failed":
        # An open extract job for the same revision will retry; otherwise the
        # run is terminally failed and embedding it would be wasted work.
        if db.query_one(
            "SELECT 1 AS x FROM jobs WHERE stage = ? AND input_id = ? AND state IN "
            + _OPEN_STATES,
            (STAGE_EXTRACT, run["rev_id"]),
        ) is None:
            jobs.fail(job, "run_failed", "extraction run failed; not embedding", transient=False)
            return True
        jobs.defer(job)
        return True
    if db.query_one(
        "SELECT 1 AS x FROM jobs WHERE stage IN (?, ?) AND input_id = ? AND state NOT IN "
        + _SETTLED_STATES,
        (STAGE_OCR, STAGE_CHUNK, job.input_id or ""),
    ) is not None:
        jobs.defer(job)
        return True
    run_id = run["run_id"]
    rev = db.query_one(
        "SELECT rev_id, doc_id, is_active FROM source_revisions WHERE rev_id = ?",
        (run["rev_id"],),
    )
    if rev is None:
        jobs.fail(job, "missing_source", f"revision {run['rev_id']} not in catalog", transient=False)
        return True
    if not rev["is_active"]:
        # The revision was replaced: its vectors must never enter the index.
        jobs.succeed(
            job, json.dumps({"run_id": run_id, "chunks": 0, "noop": True}, sort_keys=True)
        )
        return True
    current = chunk_fingerprint_for_run(db, run_id, cfg)
    if run["chunk_fingerprint"] != current:
        # Chunks are being rebuilt; the chunk job re-enqueues this stage.
        jobs.defer(job)
        return True
    try:
        emb = embedder if embedder is not None else make_embedder(cfg)
        emb_sha = embedding_sha(cfg)
    except ConfigError as exc:
        jobs.fail(job, "embedding_not_configured", str(exc), transient=False)
        return True

    chunk_rows = db.query(
        "SELECT chunk_id, text FROM chunks WHERE run_id = ? ORDER BY position", (run_id,)
    )
    if not chunk_rows:
        jobs.succeed(
            job, json.dumps({"run_id": run_id, "chunks": 0, "noop": True}, sort_keys=True)
        )
        return True

    batch_size = max(1, cfg.embedding.batch_size)
    n_batches = (len(chunk_rows) + batch_size - 1) // batch_size
    try:
        for i in range(0, len(chunk_rows), batch_size):
            batch_index = i // batch_size
            batch = chunk_rows[i : i + batch_size]
            chunk_ids = [r["chunk_id"] for r in batch]
            cp = checkpoint_path(cfg.paths.artifact_root, run_id, emb_sha, batch_index)
            row = db.query_one(
                "SELECT batch_id FROM embedding_batches "
                "WHERE run_id = ? AND embedding_sha = ? AND batch_index = ?",
                (run_id, emb_sha, batch_index),
            )
            if row is not None:
                try:
                    read_checkpoint(cp, chunk_ids)
                    continue  # checkpoint verified: this batch is done
                except CheckpointCorruptError:
                    # Corrupt artifact: drop the manifest row so the batch
                    # re-encodes and re-commits (PRD §14 corrupt-artifact case).
                    db.execute(
                        "DELETE FROM embedding_batches WHERE batch_id = ?", (row["batch_id"],)
                    )
            vectors = encode_batch_oom(
                emb,
                [r["text"] for r in batch],
                start_size=batch_size,
                max_halvings=cfg.embedding.oom_max_halvings,
            )
            if len(vectors) != len(chunk_ids):
                jobs.fail(
                    job,
                    "worker_error",
                    f"encoder returned {len(vectors)} vectors for {len(chunk_ids)} chunks",
                    transient=False,
                )
                return True
            vector_sha = write_checkpoint(
                cp,
                vectors,
                emb.dimensions,
                {
                    "run_id": run_id,
                    "embedding_sha": emb_sha,
                    "model_revision": emb.model_revision,
                    "batch_index": batch_index,
                    "chunk_ids": chunk_ids,
                },
            )
            relpath = cp.relative_to(cfg.paths.artifact_root).as_posix()
            with db.transaction():
                db.execute(
                    """
                    INSERT INTO embedding_batches
                        (batch_id, run_id, model_revision, embedding_sha, batch_index,
                         chunk_ids, vector_sha256, artifact_relpath, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, embedding_sha, batch_index) DO NOTHING
                    """,
                    (
                        f"{run_id}:{emb_sha}:{batch_index:05d}",
                        run_id,
                        emb.model_revision,
                        emb_sha,
                        batch_index,
                        json.dumps(chunk_ids),
                        vector_sha,
                        relpath,
                        time.time(),
                    ),
                )
            jobs.heartbeat(job, lease_ttl)
    except StaleLeaseError:
        return False
    except EmbeddingOOMError as exc:
        jobs.fail(job, "embedding_oom", str(exc), transient=True)
        return True
    except ModelUnavailableError as exc:
        jobs.fail(job, "model_unavailable", str(exc), transient=True)
        return True
    except EmbeddingError as exc:
        jobs.fail(job, "embedding_error", str(exc), transient=False)
        return True
    except Exception as exc:
        jobs.fail(job, "worker_error", f"{type(exc).__name__}: {exc}", transient=True)
        return True

    # The version is an idempotency hint only (M6 fix B1): the recorded corpus
    # epoch, or "init" before the first publish commits one. The publish handler
    # reuses (or, for a genuinely new revision, computes) the real epoch under
    # the publish lock, and the epoch fan-out converges any stale-epoch
    # republish — so enqueuing stays O(1) instead of a full corpus tokenization.
    version = f"{emb_sha}:{corpus_stats_version_part(db)}"
    jobs.enqueue(
        make_task_key(STAGE_PUBLISH, rev["rev_id"], version),
        STAGE_PUBLISH,
        input_id=rev["rev_id"],
        input_version=version,
    )
    jobs.succeed(
        job,
        json.dumps(
            {
                "run_id": run_id,
                "chunks": len(chunk_rows),
                "batches": n_batches,
                "embedding_sha": emb_sha,
            },
            sort_keys=True,
        ),
    )
    return True


def _run_publish(
    db: Database,
    cfg: Config,
    jobs: Jobs,
    job: Claimed,
    lease_ttl: float,
    qdrant: QdrantOps | None = None,
) -> bool:
    """Stage and activate the revision's index generation under the publish
    lock (PRD §8F).

    ``input_version`` is the ``<embedding_sha>:<stats_sha>`` the enqueue side
    computed; it is an idempotency key only — the handler resolves the real
    statistics epoch from the committed corpus-epoch record (computing only for
    a genuinely new revision, M6 fix B1), so a job that raced a corpus-stats
    change simply no-ops if its generation is already current.
    """
    rev_id = job.input_id or ""
    rev = db.query_one(
        "SELECT rev_id, is_active FROM source_revisions WHERE rev_id = ?", (rev_id,)
    )
    if rev is None:
        jobs.fail(job, "missing_source", f"revision {rev_id} not in catalog", transient=False)
        return True
    if not rev["is_active"]:
        jobs.succeed(job, json.dumps({"rev_id": rev_id, "result": "noop"}, sort_keys=True))
        return True
    run = db.query_one(
        "SELECT run_id FROM extraction_runs WHERE rev_id = ? AND state = 'succeeded' "
        "ORDER BY created_at DESC",
        (rev_id,),
    )
    if run is None:
        jobs.defer(job)  # extraction not settled yet
        return True
    run_id = run["run_id"]
    emb_sha_part, _, stats_sha_part = (job.input_version or "").partition(":")
    if emb_sha_part and stats_sha_part:
        state = publication_is_current(
            db, rev_id=rev_id, gen_id=generation_id(run_id, emb_sha_part, stats_sha_part)
        )
        if state in ("active", "superseded"):
            jobs.succeed(job, json.dumps({"rev_id": rev_id, "result": "noop"}, sort_keys=True))
            return True

    q = qdrant if qdrant is not None else RealQdrantOps(cfg)
    if not q.ping():
        jobs.fail(job, "qdrant_unavailable", "Qdrant unreachable; will retry", transient=True)
        return True
    owner = f"publish-{os.uname().nodename}-{os.getpid()}"
    if not acquire_publish_lock(db, owner):
        jobs.defer(job)  # another publisher is mid-switch
        return True
    try:
        try:
            # A fresh state (e.g. the pilot sandbox) has no collection yet;
            # under the publish lock creation is serialized with the switch.
            q.ensure_collection(dimensions=cfg.embedding.dimensions)
            result = publish_generation(
                db,
                cfg,
                q,
                rev_id=rev_id,
                run_id=run_id,
                on_progress=lambda: jobs.heartbeat(job, lease_ttl),
            )
        except StaleLeaseError:
            return False
        except PublicationError as exc:
            jobs.fail(job, "publication_error", str(exc), transient=True)
            return True
        if result == "published":
            _enqueue_epoch_republishes(db, cfg, jobs, rev_id)
        jobs.succeed(
            job, json.dumps({"rev_id": rev_id, "result": result}, sort_keys=True)
        )
    finally:
        release_publish_lock(db, owner)
    return True


def _enqueue_epoch_republishes(db: Database, cfg: Config, jobs: Jobs, rev_id: str) -> None:
    """Converge the other books onto the new statistics epoch.

    Publishing under a fresh corpus-stats epoch shifts the sparse (BM25) space
    for *every* publication, so each other active publication must be
    re-published (re-sparse-encoded) under the new epoch. Only revisions whose
    generation already carries the current embedding epoch are eligible here;
    a different model epoch belongs to the re-embedding path, not this one.
    Task keys are idempotent and the handler resolves the epoch from the
    committed record, so these converge even if the corpus keeps changing.
    """
    emb_sha = embedding_sha(cfg)
    row = db.query_one(
        """
        SELECT g.sparse_stats_sha AS stats
        FROM publications p
        JOIN index_generations g ON g.gen_id = p.gen_id
        WHERE p.rev_id = ? AND p.state = 'active'
        """,
        (rev_id,),
    )
    if row is None:
        return
    stats_sha = row["stats"]
    others = db.query(
        """
        SELECT p.rev_id AS rev_id, g.embedding_sha AS emb
        FROM publications p
        JOIN index_generations g ON g.gen_id = p.gen_id
        WHERE p.state = 'active' AND p.rev_id != ? AND g.sparse_stats_sha != ?
        """,
        (rev_id, stats_sha),
    )
    for other in others:
        if other["emb"] != emb_sha:
            continue  # different model epoch: re-embedding path's job
        version = f"{emb_sha}:{stats_sha}"
        jobs.enqueue(
            make_task_key(STAGE_PUBLISH, other["rev_id"], version),
            STAGE_PUBLISH,
            input_id=other["rev_id"],
            input_version=version,
        )


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


def _reconcile_index(db: Database, cfg: Config) -> None:
    """Close the M4 pipeline gaps on worker start (durable, idempotent).

    Two lost-enqueue windows survive a crash:

    * a run whose embedding checkpoints are incomplete for the *current*
      embedding configuration (never embedded, or a model/dtype/dimensions
      change invalidated them) gets its embed job; a batch_size change
      self-heals the same way — the expected batch count no longer matches;
    * a settled, active revision that never reached a staged or active
      publication (a publish job row lost after the embed job succeeded)
      gets its publish job.

    Runs with an in-flight ocr/chunk/embed job are skipped (their own
    handoff will re-enqueue), as are replaced revisions — their evidence must
    not enter the index.
    """
    jobs = Jobs(db)
    try:
        emb_sha = embedding_sha(cfg)
    except ConfigError:
        emb_sha = None
    stats_sha: str | None = None
    runs = db.query(
        "SELECT run_id, rev_id, chunk_fingerprint FROM extraction_runs WHERE state = 'succeeded'"
    )
    for row in runs:
        run_id = row["run_id"]
        if db.query_one(
            "SELECT 1 AS x FROM jobs WHERE stage IN (?, ?, ?) AND input_id = ? "
            "AND state NOT IN " + _SETTLED_STATES,
            (STAGE_OCR, STAGE_CHUNK, STAGE_EMBED, run_id),
        ) is not None:
            continue
        rev = db.query_one(
            "SELECT rev_id, is_active FROM source_revisions WHERE rev_id = ?", (row["rev_id"],)
        )
        if rev is None or not rev["is_active"]:
            continue
        current = chunk_fingerprint_for_run(db, run_id, cfg)
        if row["chunk_fingerprint"] != current:
            continue  # the chunk job re-runs and re-enqueues from there
        n_chunks = db.query_one(
            "SELECT COUNT(*) AS n FROM chunks WHERE run_id = ?", (run_id,)
        )
        assert n_chunks is not None
        n_chunks_count = int(n_chunks["n"] or 0)
        if n_chunks_count == 0:
            continue
        if emb_sha is not None:
            batch_size = max(1, cfg.embedding.batch_size)
            n_batches = db.query_one(
                "SELECT COUNT(*) AS n FROM embedding_batches "
                "WHERE run_id = ? AND embedding_sha = ?",
                (run_id, emb_sha),
            )
            assert n_batches is not None
            expected = (n_chunks_count + batch_size - 1) // batch_size
            if int(n_batches["n"] or 0) != expected:
                _enqueue_embed_job(jobs, run_id, current)
        # Publication gap: no staged/active publication for this revision.
        if db.query_one(
            "SELECT 1 AS x FROM publications WHERE rev_id = ? AND state IN ('staged', 'active')",
            (row["rev_id"],),
        ) is not None:
            continue
        if emb_sha is None:
            continue  # cannot compute the version until embedding is configured
        if stats_sha is None:
            stats_sha = corpus_stats_version_part(db)  # O(1) hint; handler resolves the real epoch
        version = f"{emb_sha}:{stats_sha}"
        jobs.enqueue(
            make_task_key(STAGE_PUBLISH, row["rev_id"], version),
            STAGE_PUBLISH,
            input_id=row["rev_id"],
            input_version=version,
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
    qdrant: QdrantOps | None = None,
    embedder: Embedder | None = None,
) -> int:
    """Claim and run jobs until the queue is drained (*once*) or *stop_event*
    becomes true. Returns the number of jobs brought to a handled state
    (terminal, or deferred — deferrals keep the *once* drain loop from
    busy-spinning on a not-ready job).

    Stopping mid-loop is graceful by construction: at most the current job's
    current unit is in flight; everything already committed is durable and
    resumable (PRD §7 SIGTERM). *qdrant*/*embedder* are injectable seams for
    tests (and the CLI); production callers pass None and the worker builds
    them per job.
    """
    name = worker_name()
    jobs = Jobs(db)
    if reconcile_on_start:
        _reconcile_chunks(db, cfg)
        _reconcile_index(db, cfg)
        q = qdrant if qdrant is not None else RealQdrantOps(cfg)
        try:
            if q.ping():
                reconcile_publications(db, cfg, q)
        except Exception:  # best-effort: a broken index must not block the queue
            pass
    handlers: dict[str, Callable[..., bool]] = {
        STAGE_EXTRACT: _run_extract,
        STAGE_OCR: _run_ocr,
        STAGE_CHUNK: _run_chunk,
        STAGE_EMBED: partial(_run_embed, embedder=embedder),
        STAGE_PUBLISH: partial(_run_publish, qdrant=qdrant),
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
