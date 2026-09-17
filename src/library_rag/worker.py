"""Worker loop: claim durable jobs and run their pipeline stage (PRD §7).

M2 implements the ``extract`` stage. Contract:

* one job at a time per worker; the lease is heartbeated between units
  (inject :meth:`Jobs.heartbeat` as the extractor's ``on_progress``), so a
  long book cannot expire its own lease;
* ``StaleLeaseError`` from any fenced write means another worker owns the job
  now — stop touching it and let the new owner finish (at-least-once work,
  idempotent effects);
* ``ExtractionFailure`` categories classify permanence
  (encrypted/corrupt/... fail permanently until explicit requeue), anything
  else is transient and gets backoff/retry;
* ``run_worker`` never swallows ``StaleLeaseError`` out of ``succeed``/``fail``
  silently: such a job is simply dropped from this worker's view.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

from .archive import archive_path_for
from .catalog import Format
from .config import Config
from .db import Database
from .extraction import (
    ExtractionFailure,
    ExtractorCtx,
    extract_epub,
    extract_pdf,
    fail_run,
    is_permanent,
)
from .extraction.epub import parser_version as epub_parser_version
from .extraction.pdf import parser_version as pdf_parser_version
from .identity import extraction_key
from .jobs import Claimed, Jobs, StaleLeaseError
from .scan import STAGE_EXTRACT

__all__ = ["DEFAULT_LEASE_TTL", "build_ctx", "run_worker", "worker_name"]

DEFAULT_LEASE_TTL = 300.0


def worker_name() -> str:
    return f"extract-{os.uname().nodename}-{os.getpid()}"


def build_ctx(db: Database, cfg: Config, rev: Any, on_progress: Callable[[], None] | None) -> ExtractorCtx:
    """Assemble the extractor context for a revision row (run id is keyed on
    source hash + parser version + stage settings, PRD §6)."""
    fmt = Format(rev["format"])
    parser = pdf_parser_version() if fmt is Format.PDF else epub_parser_version()
    return ExtractorCtx(
        db=db,
        artifact_root=cfg.paths.artifact_root,
        rev=dict(rev),
        run_id=extraction_key(rev["sha256"], parser, cfg.extraction.settings_sha()),
        settings=cfg.extraction,
        source_path=archive_path_for(cfg.paths.archive_root, rev["sha256"]),
        on_progress=on_progress,
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
    ctx = build_ctx(db, cfg, rev, on_progress=lambda: jobs.heartbeat(job, lease_ttl))
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


def run_worker(
    db: Database,
    cfg: Config,
    *,
    once: bool = False,
    lease_ttl: float = DEFAULT_LEASE_TTL,
    poll_delay: float = 1.0,
    stop_event: Callable[[], bool] | None = None,
) -> int:
    """Claim and run jobs until the queue is drained (*once*) or *stop_event*
    becomes true. Returns the number of jobs brought to a terminal state.

    Stopping mid-loop is graceful by construction: at most the current job's
    current unit is in flight; everything already committed is durable and
    resumable (PRD §7 SIGTERM).
    """
    name = worker_name()
    jobs = Jobs(db)
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
        if job.stage != STAGE_EXTRACT:
            jobs.fail(job, "unknown_stage", f"stage {job.stage!r} has no handler yet", transient=False)
            completed += 1  # permanent_failed is a terminal state
            continue
        if _run_extract(db, cfg, jobs, job, lease_ttl):
            completed += 1
    return completed


def _sleep_interruptible(delay: float, stop_event: Callable[[], bool] | None) -> None:
    end = time.monotonic() + delay
    while time.monotonic() < end:
        if stop_event is not None and stop_event():
            return
        time.sleep(min(0.2, max(0.0, end - time.monotonic())))
