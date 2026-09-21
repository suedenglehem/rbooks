"""Durable job execution: leases, fencing, pause, retry (PRD §7).

A job moves through ``pending -> running -> {succeeded, retryable_failed,
permanent_failed, cancelled}``. Claims are made atomically in a short
``BEGIN IMMEDIATE`` transaction. Every job carries a **lease token** that fences
stale workers: the fencing token is embedded in the ``WHERE`` clause of every
state-changing write, so only the worker that currently holds the token can
heartbeat, commit, or fail the job. A slow worker whose lease has expired finds
its token replaced and its writes rejected with :class:`StaleLeaseError`.

Semantics are at-least-once with idempotent effects (PRD §7): a killed worker is
recovered by reclaiming its expired lease and re-running the (idempotent) work,
never by assuming it "succeeded."
"""

from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .db import Database
from .locks import pid_alive
from .versioning import software_version

__all__ = [
    "LEGACY_VERSION",
    "VERSION_ACK_KEY",
    "Claimed",
    "Jobs",
    "StaleLeaseError",
    "VersionGateError",
    "backoff_delay",
    "normalize_version",
]


class StaleLeaseError(RuntimeError):
    """Raised when a write is rejected because the caller's lease token is stale."""


class VersionGateError(RuntimeError):
    """Raised at worker start when the queue holds jobs created by another
    software version and the operator has not explicitly confirmed running them.

    Carries the human-facing *message* and the *foreign versions* (with counts)
    so the caller can render an actionable refusal."""

    def __init__(self, message: str, foreign: dict[str, int]) -> None:
        super().__init__(message)
        self.foreign = foreign


# Sentinel for jobs enqueued before version signatures existed (the
# ``created_by_version`` column is NULL on them). Treated as foreign: a NULL
# job was necessarily enqueued by pre-signature code, i.e. a different version
# than the one now about to run it.
LEGACY_VERSION = "legacy"

# Job states that are still executable by a (re)started worker. Terminal
# states (succeeded / permanent_failed / cancelled) are never re-executed, so
# they are excluded from the version gate.
_NOT_TERMINAL = ("'pending'", "'running'", "'retryable_failed'")

# meta key holding the operator's recorded confirmation of foreign versions.
VERSION_ACK_KEY = "version_gate_ack"

# Trailing pid in a lease_owner string (worker_name() = "<stage>-<host>-<pid>").
# Used by reclaim_dead_lease_owners to tell a provably-dead owner from a live one.
_OWNER_PID = re.compile(r"-(\d+)$")


def normalize_version(version: str | None) -> str:
    """Map a stored ``created_by_version`` (NULL for pre-signature jobs) to a
    displayable token; NULL becomes :data:`LEGACY_VERSION`."""
    return version if version is not None else LEGACY_VERSION


@dataclass(frozen=True)
class Claimed:
    """A job the calling worker now holds, together with its fencing token."""

    job_id: int
    task_key: str
    stage: str
    input_id: str | None
    input_version: str | None
    range_spec: str | None
    attempts: int
    max_attempts: int
    token: str


def backoff_delay(attempts: int, base: float = 1.0, cap: float = 300.0, jitter: float = 0.2) -> float:
    """Exponential backoff (base * 2 ** (attempts-1)) capped, with multiplicative jitter."""
    exp: float = min(cap, base * (2 ** max(0, attempts - 1)))
    if jitter > 0:
        exp = exp * (1.0 + random.uniform(0.0, jitter))
    return exp


class Jobs:
    def __init__(self, db: Database, *, version: str | None = None) -> None:
        self._db = db
        # The software version stamped on every job this instance enqueues.
        # Defaults to the running code's content hash (test seam: pass a
        # literal to simulate a different installed version).
        self._version = version if version is not None else software_version()

    # --- enqueue -----------------------------------------------------------
    def enqueue(
        self,
        task_key: str,
        stage: str,
        input_id: str | None = None,
        input_version: str | None = None,
        range_spec: str | None = None,
        max_attempts: int = 3,
        now: float | None = None,
    ) -> None:
        """Add a pending job, idempotently on *task_key*.

        The row is stamped with the software version that enqueued it
        (``created_by_version``); the worker-start version gate uses this to
        detect work created by a different code version (M7 slice 9).
        ``INSERT OR IGNORE`` means a re-enqueue of an existing *task_key*
        keeps the original creator's version — the signature records who
        *created* the job, not who last tried to.
        """
        ts = time.time() if now is None else now
        with self._db.transaction():
            self._db.execute(
                """
                INSERT OR IGNORE INTO jobs
                    (task_key, stage, input_id, input_version, range_spec, state,
                     max_attempts, created_by_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                """,
                (
                    task_key,
                    stage,
                    input_id,
                    input_version,
                    range_spec,
                    max_attempts,
                    self._version,
                    ts,
                    ts,
                ),
            )

    # --- claim -------------------------------------------------------------
    def claim(self, worker: str, ttl: float, now: float | None = None) -> Claimed | None:
        """Claim the next eligible job for *worker*, or None if none.

        Reclaims expired leases first, then claims one pending (or due
        retryable) job, minting a fresh fencing token and recording the attempt.
        Refuses to claim new work while paused.
        """
        ts = time.time() if now is None else now
        with self._db.transaction():
            if self._is_paused():
                return None
            # Reclaim jobs whose lease expired (worker died / SIGKILL).
            self._db.execute(
                """
                UPDATE jobs SET state = 'pending', lease_owner = NULL, lease_token = NULL,
                       lease_expires_at = NULL, updated_at = ?
                WHERE state = 'running' AND lease_expires_at <= ?
                """,
                (ts, ts),
            )
            row = self._db.query_one(
                """
                SELECT * FROM jobs
                WHERE state = 'pending'
                   OR (state = 'retryable_failed' AND next_attempt_at <= ?)
                ORDER BY job_id LIMIT 1
                """,
                (ts,),
            )
            if row is None:
                return None
            token = uuid.uuid4().hex
            attempts = int(row["attempts"]) + 1
            self._db.execute(
                """
                UPDATE jobs SET state = 'running', lease_owner = ?, lease_token = ?,
                       lease_expires_at = ?, last_heartbeat_at = ?, attempts = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (worker, token, ts + ttl, ts, attempts, ts, row["job_id"]),
            )
            return Claimed(
                job_id=int(row["job_id"]),
                task_key=row["task_key"],
                stage=row["stage"],
                input_id=row["input_id"],
                input_version=row["input_version"],
                range_spec=row["range_spec"],
                attempts=attempts,
                max_attempts=int(row["max_attempts"]),
                token=token,
            )

    # --- heartbeat / commit / fail (all fenced) ---------------------------
    def heartbeat(self, job: Claimed, ttl: float, now: float | None = None) -> None:
        ts = time.time() if now is None else now
        cur = self._db.execute(
            """
            UPDATE jobs SET last_heartbeat_at = ?, lease_expires_at = ?
            WHERE job_id = ? AND lease_token = ?
            """,
            (ts, ts + ttl, job.job_id, job.token),
        )
        if cur.rowcount == 0:
            raise StaleLeaseError(f"job {job.job_id}: stale token on heartbeat")

    def succeed(self, job: Claimed, output_manifest: str, now: float | None = None) -> None:
        """Fenced commit: only the current token may mark the job succeeded."""
        ts = time.time() if now is None else now
        cur = self._db.execute(
            """
            UPDATE jobs SET state = 'succeeded', output_manifest = ?,
                   lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL, updated_at = ?
            WHERE job_id = ? AND lease_token = ?
            """,
            (output_manifest, ts, job.job_id, job.token),
        )
        if cur.rowcount == 0:
            raise StaleLeaseError(f"job {job.job_id}: stale token on commit")

    def fail(
        self,
        job: Claimed,
        error_category: str,
        error_detail: str,
        *,
        transient: bool = True,
        now: float | None = None,
        backoff: float = 1.0,
    ) -> None:
        """Fenced failure. Transient and not exhausted -> retryable; else permanent."""
        ts = time.time() if now is None else now
        cur = self._db.execute(
            """
            UPDATE jobs SET error_category = ?, error_detail = ?, updated_at = ?
            WHERE job_id = ? AND lease_token = ?
            """,
            (error_category, error_detail, ts, job.job_id, job.token),
        )
        if cur.rowcount == 0:
            raise StaleLeaseError(f"job {job.job_id}: stale token on fail")

        exhausted = job.attempts >= job.max_attempts
        if (not transient) or exhausted:
            self._db.execute(
                """
                UPDATE jobs SET state = 'permanent_failed', lease_owner = NULL,
                       lease_token = NULL, lease_expires_at = NULL
                WHERE job_id = ?
                """,
                (job.job_id,),
            )
        else:
            delay = backoff_delay(job.attempts, base=backoff)
            self._db.execute(
                """
                UPDATE jobs SET state = 'retryable_failed', lease_owner = NULL,
                       lease_token = NULL, lease_expires_at = NULL, next_attempt_at = ?
                WHERE job_id = ?
                """,
                (ts + delay, job.job_id),
            )

    def defer(self, job: Claimed, delay: float = 15.0, now: float | None = None) -> None:
        """Fenced "wait and retry" for a job whose prerequisites are not ready yet.

        Unlike :meth:`fail`, deferral is *not* an error: it clears any recorded
        error and does not consume an attempt, so a job can be deferred
        indefinitely (e.g. a chunk job waiting for the OCR jobs it depends on).
        The job returns to ``retryable_failed`` — deliberately NOT ``pending`` —
        because :meth:`claim` ignores ``next_attempt_at`` for pending jobs, and a
        pending defer would spin the worker in a hot loop.
        """
        ts = time.time() if now is None else now
        cur = self._db.execute(
            """
            UPDATE jobs SET state = 'retryable_failed', next_attempt_at = ?,
                   error_category = NULL, error_detail = NULL,
                   lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                   updated_at = ?
            WHERE job_id = ? AND lease_token = ?
            """,
            (ts + delay, ts, job.job_id, job.token),
        )
        if cur.rowcount == 0:
            raise StaleLeaseError(f"job {job.job_id}: stale token on defer")

    def reclaim_dead_lease_owners(self, now: float | None = None) -> int:
        """Flip ``running`` jobs whose owner process is provably dead back to
        ``pending`` immediately (M7 slice 10, ``ingest --force``).

        ``lease_owner`` is ``worker_name()`` = ``extract-<host>-<pid>``. If that
        pid no longer exists, the lease is already orphaned, so there is no
        reason to wait out the 300 s TTL. ``claim()`` still reclaims expired
        leases on its own; this just makes the recovery instant after a crash
        or SIGKILL. Jobs owned by a *live* pid (even a stale-token one) are
        left alone — a live owner may still commit.
        """
        ts = time.time() if now is None else now
        reclaimed = 0
        rows = self._db.query(
            "SELECT job_id, lease_owner FROM jobs WHERE state = 'running'"
        )
        for row in rows:
            owner = row["lease_owner"] or ""
            match = _OWNER_PID.search(owner)
            if match is None:
                continue  # owner format we do not recognize; let the TTL handle it
            if not pid_alive(int(match.group(1))):
                cur = self._db.execute(
                    """
                    UPDATE jobs SET state = 'pending', lease_owner = NULL,
                           lease_token = NULL, lease_expires_at = NULL, updated_at = ?
                    WHERE job_id = ? AND state = 'running'
                    """,
                    (ts, row["job_id"]),
                )
                reclaimed += cur.rowcount
        return reclaimed

    # --- pause / resume ----------------------------------------------------
    def pause(self, reason: str = "", now: float | None = None) -> None:
        ts = time.time() if now is None else now
        with self._db.transaction():
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES ('paused', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (json.dumps({"reason": reason, "at": ts}),),
            )

    def resume(self) -> None:
        with self._db.transaction():
            self._db.execute("DELETE FROM meta WHERE key = 'paused'")

    def is_paused(self) -> bool:
        return self._is_paused()

    def _is_paused(self) -> bool:
        row = self._db.query_one("SELECT 1 AS x FROM meta WHERE key = 'paused'")
        return row is not None

    # --- retry -------------------------------------------------------------
    def retry(self, include_permanent: bool = False, now: float | None = None) -> int:
        """Requeue retryable jobs (and, if requested, permanently-failed ones)."""
        ts = time.time() if now is None else now
        requeued = 0
        with self._db.transaction():
            cur = self._db.execute(
                """
                UPDATE jobs SET state = 'pending', next_attempt_at = NULL, updated_at = ?
                WHERE state = 'retryable_failed'
                """,
                (ts,),
            )
            requeued += cur.rowcount
            if include_permanent:
                cur = self._db.execute(
                    """
                    UPDATE jobs SET state = 'pending', attempts = 0, next_attempt_at = NULL,
                           error_category = NULL, error_detail = NULL, updated_at = ?
                    WHERE state = 'permanent_failed'
                    """,
                    (ts,),
                )
                requeued += cur.rowcount
        return requeued

    # --- inspection --------------------------------------------------------
    def counts(self) -> dict[str, int]:
        rows = self._db.query("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state")
        return {r["state"]: int(r["n"]) for r in rows}

    def version_mismatch_counts(self, current: str) -> dict[str, int]:
        """Counts of not-yet-terminal jobs NOT created by *current* version.

        Maps each normalized creator version (NULL -> :data:`LEGACY_VERSION`)
        to the number of jobs in ``pending`` / ``running`` / ``retryable_failed``
        state it enqueued. An empty result means the queue holds only work
        created by the running code version.
        """
        states = ", ".join(_NOT_TERMINAL)
        rows = self._db.query(
            f"""
            SELECT created_by_version, COUNT(*) AS n FROM jobs
            WHERE state IN ({states})
              AND (created_by_version IS NULL OR created_by_version != ?)
            GROUP BY created_by_version
            """,
            (current,),
        )
        return {normalize_version(r["created_by_version"]): int(r["n"]) for r in rows}

    def get(self, job_id: int) -> dict[str, Any] | None:
        row = self._db.query_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        return dict(row) if row is not None else None
