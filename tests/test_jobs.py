"""Durable job execution: lease fencing, crash replay, pause, backoff.

These cover the M1 recovery gate:
* a killed worker's job is replayed and produces exactly one logical output;
* a stale worker (expired lease / replaced token) cannot commit;
* pause blocks new claims while active work finishes;
* transient failures back off and retry up to a bound, then fail permanently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from library_rag.artifacts import commit_bytes
from library_rag.db import Database
from library_rag.identity import make_task_key
from library_rag.jobs import BOOK_BUDGET_PARKED, Jobs, StaleLeaseError

MANIFEST = '{"logical_output": true}'


def _work(artifact_root: Path, task_key: str) -> str:
    """Idempotent work: durably commit an artifact keyed by the task identity."""
    dest = artifact_root / "extract" / f"{task_key}.json"
    commit_bytes(dest, b'{"done": true}')
    return MANIFEST


def _count_artifacts(artifact_root: Path) -> int:
    d = artifact_root / "extract"
    if not d.exists():
        return 0
    return len([p for p in d.iterdir() if p.is_file()])


def test_killed_worker_replay_produces_one_logical_output(
    state_db: Database, roots: dict[str, Path]
) -> None:
    jobs = Jobs(state_db)
    task_key = make_task_key("extract", "doc-1", "cfg-v1", "pages:0-9")
    jobs.enqueue(task_key, "extract", input_id="doc-1", input_version="cfg-v1",
                 range_spec="pages:0-9", now=1000.0)

    # Worker 1 claims, does the work, then is SIGKILLed before committing success.
    w1 = jobs.claim("worker-1", ttl=1.0, now=1000.0)
    assert w1 is not None
    assert w1.attempts == 1
    _work(roots["artifact_root"], w1.task_key)
    # (w1 never calls succeed; its lease simply expires.)

    # A restarted coordinator reclaims the expired lease and worker 2 re-runs.
    w2 = jobs.claim("worker-2", ttl=1.0, now=1002.0)  # 2s later, lease expired
    assert w2 is not None
    assert w2.job_id == w1.job_id
    assert w2.attempts == 2
    assert w2.token != w1.token
    _work(roots["artifact_root"], w2.task_key)  # idempotent re-run
    jobs.succeed(w2, MANIFEST, now=1002.0)

    row = jobs.get(w1.job_id)
    assert row is not None
    assert row["state"] == "succeeded"
    assert row["attempts"] == 2
    assert row["output_manifest"] == MANIFEST
    # At-least-once work ran twice, but there is exactly one logical output.
    assert _count_artifacts(roots["artifact_root"]) == 1


def test_stale_worker_cannot_commit_after_reclaim(
    state_db: Database, roots: dict[str, Path]
) -> None:
    jobs = Jobs(state_db)
    task_key = make_task_key("extract", "doc-2", "cfg-v1")
    jobs.enqueue(task_key, "extract", input_id="doc-2", input_version="cfg-v1", now=1000.0)

    w1 = jobs.claim("worker-1", ttl=1.0, now=1000.0)
    assert w1 is not None
    # Lease expires; worker 2 reclaims and commits.
    w2 = jobs.claim("worker-2", ttl=1.0, now=1002.0)
    assert w2 is not None and w2.token != w1.token
    jobs.succeed(w2, MANIFEST, now=1002.0)

    # The stale worker 1, arriving late, cannot commit with its old token.
    with pytest.raises(StaleLeaseError):
        jobs.succeed(w1, MANIFEST, now=1003.0)
    # ...nor can it heartbeat or fail the job.
    with pytest.raises(StaleLeaseError):
        jobs.heartbeat(w1, ttl=1.0, now=1003.0)
    with pytest.raises(StaleLeaseError):
        jobs.fail(w1, "transient", "late failure", now=1003.0)


def test_stale_token_rejected_even_mid_flight(
    state_db: Database, roots: dict[str, Path]
) -> None:
    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("extract", "doc-3", "v1"), "extract",
                 input_id="doc-3", input_version="v1", now=1000.0)
    w1 = jobs.claim("worker-1", ttl=1.0, now=1000.0)
    assert w1 is not None
    w2 = jobs.claim("worker-2", ttl=1.0, now=1002.0)  # reclaims while w1 is "stuck"
    assert w2 is not None and w2.token != w1.token
    # w1's token is already replaced; its commit is fenced out before w2 commits.
    with pytest.raises(StaleLeaseError):
        jobs.succeed(w1, MANIFEST, now=1002.5)


def test_pause_blocks_new_claims_but_allows_finish(
    state_db: Database, roots: dict[str, Path]
) -> None:
    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("extract", "d", "v"), "extract", input_id="d", input_version="v",
                 now=1000.0)
    # Claim one job before pausing.
    claimed = jobs.claim("worker-1", ttl=10.0, now=1000.0)
    assert claimed is not None
    jobs.pause(reason="operator")
    assert jobs.is_paused()
    # While paused, no (new) work is claimed even though a job is pending.
    assert jobs.claim("worker-2", ttl=10.0, now=1001.0) is None
    # But the already-active job can still finish.
    jobs.succeed(claimed, MANIFEST, now=1001.0)
    row = jobs.get(claimed.job_id)
    assert row is not None
    assert row["state"] == "succeeded"
    jobs.resume()
    assert not jobs.is_paused()


def test_transient_failure_backs_off_then_retries(
    state_db: Database, roots: dict[str, Path]
) -> None:
    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("extract", "d", "v"), "extract", input_id="d", input_version="v",
                 max_attempts=3, now=1000.0)

    j = jobs.claim("w", ttl=10.0, now=1000.0)
    assert j is not None
    jobs.fail(j, "transient", "disk hiccup", transient=True, now=1000.0, backoff=5.0)
    row = jobs.get(j.job_id)
    assert row is not None
    assert row["state"] == "retryable_failed"
    assert row["next_attempt_at"] > 1000.5

    # Not claimable before the backoff elapses.
    assert jobs.claim("w", ttl=10.0, now=1000.5) is None
    # Claimable once the backoff has passed.
    due = jobs.claim("w", ttl=10.0, now=row["next_attempt_at"] + 1)
    assert due is not None
    assert due.attempts == 2
    jobs.succeed(due, MANIFEST, now=row["next_attempt_at"] + 1)
    row = jobs.get(j.job_id)
    assert row is not None
    assert row["state"] == "succeeded"


def test_release_parks_without_consuming_attempt(state_db: Database) -> None:
    # Bounded-run budget gate: a claimed-but-unprocessed job is put back with
    # its attempt restored (unlike defer, which leaves the claim's increment).
    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("extract", "d", "v"), "extract", input_id="d", input_version="v",
                 max_attempts=3, now=1000.0)

    j = jobs.claim("w", ttl=10.0, now=1000.0)
    assert j is not None and j.attempts == 1
    jobs.release(j, delay=30.0, now=1000.0)
    row = jobs.get(j.job_id)
    assert row is not None
    assert row["state"] == "retryable_failed"
    assert int(row["attempts"]) == 0  # the claim's increment was restored
    assert row["error_category"] is None
    assert row["error_detail"] == BOOK_BUDGET_PARKED

    # Not claimable before the delay elapses; fresh (one attempt) afterwards.
    assert jobs.claim("w", ttl=10.0, now=1029.0) is None
    due = jobs.claim("w", ttl=10.0, now=1031.0)
    assert due is not None and due.attempts == 1

    # Repeated park/claim cycles (bounded re-runs) must not exhaust the budget:
    for release_at in (1031.0, 1062.0):
        jobs.release(due, delay=30.0, now=release_at)
        due = jobs.claim("w", ttl=10.0, now=release_at + 31.0)
        assert due is not None and due.attempts == 1
    jobs.release(due, delay=30.0, now=1093.0)
    row = jobs.get(j.job_id)
    assert row is not None
    assert int(row["attempts"]) == 0

    # Fenced like every other write: the old token no longer parks it.
    with pytest.raises(StaleLeaseError):
        jobs.release(j, delay=30.0, now=1125.0)


def test_attempts_exhausted_become_permanent(state_db: Database, roots: dict[str, Path]) -> None:
    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("extract", "d", "v"), "extract", input_id="d", input_version="v",
                 max_attempts=2, now=1000.0)
    j1 = jobs.claim("w", ttl=10.0, now=1000.0)
    assert j1 is not None
    jobs.fail(j1, "transient", "err", transient=True, now=1000.0)
    j2 = jobs.claim("w", ttl=10.0, now=1005.0)  # after backoff
    assert j2 is not None and j2.attempts == 2
    jobs.fail(j2, "transient", "err again", transient=True, now=1005.0)
    row = jobs.get(j2.job_id)
    assert row is not None
    assert row["state"] == "permanent_failed"


def test_permanent_error_fails_immediately(state_db: Database, roots: dict[str, Path]) -> None:
    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("extract", "d", "v"), "extract", input_id="d", input_version="v",
                 now=1000.0)
    j = jobs.claim("w", ttl=10.0, now=1000.0)
    assert j is not None
    jobs.fail(j, "corrupt", "encrypted file", transient=False, now=1000.0)
    row = jobs.get(j.job_id)
    assert row is not None
    assert row["state"] == "permanent_failed"
    # An explicit retry can requeue it and reset the attempt count.
    requeued = jobs.retry(include_permanent=True, now=1001.0)
    assert requeued == 1
    j2 = jobs.claim("w", ttl=10.0, now=1001.0)
    assert j2 is not None and j2.attempts == 1


def test_enqueue_is_idempotent_on_task_key(state_db: Database, roots: dict[str, Path]) -> None:
    jobs = Jobs(state_db)
    tk = make_task_key("extract", "d", "v")
    jobs.enqueue(tk, "extract", input_id="d", input_version="v", now=1000.0)
    jobs.enqueue(tk, "extract", input_id="d", input_version="v", now=1000.0)  # no-op
    assert jobs.counts() == {"pending": 1}
