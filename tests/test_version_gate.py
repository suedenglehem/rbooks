"""Job version signature and the cross-version execution gate (M7 slice 9).

After a reboot + upgrade, jobs enqueued by another code version (or by
pre-signature code, i.e. ``created_by_version IS NULL``) must not be executed
without explicit operator confirmation. Covers:
* the content-hash version itself (determinism, change detection, cache exclusion);
* enqueue stamping and the idempotent re-enqueue keeping the creator's version;
* ``version_mismatch_counts`` (foreign / legacy / terminal-state handling);
* ``check_version_gate`` raise / ack / re-ask semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from library_rag.db import Database
from library_rag.identity import make_task_key
from library_rag.jobs import LEGACY_VERSION, Jobs, VersionGateError
from library_rag.versioning import (
    short_version,
    software_version,
    software_version_for,
)
from library_rag.worker import check_version_gate

# --- software_version_for ----------------------------------------------------


def test_software_version_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    (root / "sub").mkdir(parents=True)
    (root / "a.py").write_text("x = 1\n")
    (root / "sub" / "b.py").write_text("y = 2\n")
    h1 = software_version_for(root)
    assert software_version_for(root) == h1
    assert len(h1) == 64  # sha256 hexdigest


def test_software_version_detects_changes(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    h1 = software_version_for(root)
    (root / "a.py").write_text("x = 2\n")
    assert software_version_for(root) != h1
    (root / "a.py").write_text("x = 1\n")
    assert software_version_for(root) == h1  # content-addressed, not path-addressed


def test_software_version_ignores_cache_files(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    (root / "sub").mkdir(parents=True)
    (root / "a.py").write_text("x = 1\n")
    h1 = software_version_for(root)
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00fake\x00")
    (root / "sub" / "b.pyc").write_bytes(b"\x00fake\x00")
    assert software_version_for(root) == h1


def test_short_version_is_stable_prefix() -> None:
    v = software_version()
    assert short_version(v) == v[:12]
    assert len(short_version(v)) == 12


# --- enqueue stamping ---------------------------------------------------------


def test_enqueue_stamps_creator_version(state_db: Database) -> None:
    task_key = make_task_key("extract", "doc-sig-1", "cfg-v1")
    Jobs(state_db, version="v-old").enqueue(task_key, "extract", now=1000.0)
    row = state_db.query_one(
        "SELECT created_by_version FROM jobs WHERE task_key = ?", (task_key,)
    )
    assert row is not None
    assert row["created_by_version"] == "v-old"


def test_enqueue_defaults_to_running_version(state_db: Database) -> None:
    task_key = make_task_key("extract", "doc-sig-2", "cfg-v1")
    Jobs(state_db).enqueue(task_key, "extract", now=1000.0)
    row = state_db.query_one(
        "SELECT created_by_version FROM jobs WHERE task_key = ?", (task_key,)
    )
    assert row is not None
    assert row["created_by_version"] == software_version()


def test_reenqueue_keeps_original_creator_version(state_db: Database) -> None:
    """INSERT OR IGNORE: the signature records who *created* the job, not the
    last process that tried to enqueue it."""
    task_key = make_task_key("extract", "doc-sig-3", "cfg-v1")
    Jobs(state_db, version="v-old").enqueue(task_key, "extract", now=1000.0)
    Jobs(state_db).enqueue(task_key, "extract", now=2000.0)  # current version, same key
    row = state_db.query_one(
        "SELECT created_by_version FROM jobs WHERE task_key = ?", (task_key,)
    )
    assert row is not None
    assert row["created_by_version"] == "v-old"


# --- version_mismatch_counts ---------------------------------------------------


def test_mismatch_counts_same_version_is_empty(state_db: Database) -> None:
    Jobs(state_db).enqueue(make_task_key("extract", "doc-mc-1", "cfg-v1"), "extract",
                           now=1000.0)
    assert Jobs(state_db).version_mismatch_counts(software_version()) == {}


def test_mismatch_counts_foreign_and_legacy(state_db: Database) -> None:
    jobs = Jobs(state_db, version="v-old")
    legacy_key = make_task_key("extract", "doc-mc-3", "cfg-v1")
    jobs.enqueue(make_task_key("extract", "doc-mc-2", "cfg-v1"), "extract", now=1000.0)
    jobs.enqueue(legacy_key, "extract", now=1001.0)
    # A NULL row is necessarily pre-signature: it was enqueued by some code
    # version we never recorded, so it is foreign ("legacy").
    state_db.execute(
        "UPDATE jobs SET created_by_version = NULL WHERE task_key = ?", (legacy_key,)
    )

    counts = Jobs(state_db).version_mismatch_counts(software_version())
    assert counts == {"v-old": 1, LEGACY_VERSION: 1}


def test_mismatch_counts_ignore_terminal_states(state_db: Database) -> None:
    jobs = Jobs(state_db, version="v-old")
    tasks = {
        "succeeded": make_task_key("extract", "doc-mt-1", "cfg-v1"),
        "permanent_failed": make_task_key("extract", "doc-mt-2", "cfg-v1"),
        "cancelled": make_task_key("extract", "doc-mt-3", "cfg-v1"),
    }
    for task_key in tasks.values():
        jobs.enqueue(task_key, "extract", now=1000.0)

    w = jobs.claim("worker-1", ttl=10.0, now=1001.0)
    assert w is not None
    jobs.succeed(w, "{}", now=1002.0)

    w = jobs.claim("worker-1", ttl=10.0, now=1003.0)
    assert w is not None
    jobs.fail(w, "corrupt", "bad file", transient=False, now=1004.0)

    state_db.execute(
        "UPDATE jobs SET state = 'cancelled' WHERE task_key = ?", (tasks["cancelled"],)
    )

    assert Jobs(state_db).version_mismatch_counts(software_version()) == {}


def test_mismatch_counts_pending_and_retryable_count(state_db: Database) -> None:
    jobs = Jobs(state_db, version="v-old")
    jobs.enqueue(make_task_key("extract", "doc-mr-1", "cfg-v1"), "extract", now=1000.0)
    jobs.enqueue(make_task_key("extract", "doc-mr-2", "cfg-v1"), "extract", now=1001.0)

    w = jobs.claim("worker-1", ttl=10.0, now=1002.0)
    assert w is not None
    jobs.fail(w, "timeout", "network blip", transient=True, now=1003.0)

    counts = Jobs(state_db).version_mismatch_counts(software_version())
    assert counts == {"v-old": 2}  # one pending + one retryable_failed


# --- check_version_gate ---------------------------------------------------------


def test_gate_passes_silently_when_no_foreign_jobs(state_db: Database) -> None:
    Jobs(state_db).enqueue(make_task_key("extract", "doc-g-1", "cfg-v1"), "extract",
                           now=1000.0)
    assert check_version_gate(state_db) == {}


def test_gate_raises_on_foreign_without_confirmation(state_db: Database) -> None:
    Jobs(state_db, version="v-old").enqueue(
        make_task_key("extract", "doc-g-2", "cfg-v1"), "extract", now=1000.0
    )
    with pytest.raises(VersionGateError) as ei:
        check_version_gate(state_db)
    assert ei.value.foreign == {"v-old": 1}
    assert "--allow-version-mismatch" in str(ei.value)
    # Nothing may have been recorded or executed by the refusal.
    assert state_db.query_one(
        "SELECT 1 AS x FROM meta WHERE key = 'version_gate_ack'"
    ) is None


def test_gate_allow_writes_ack_and_passes(state_db: Database) -> None:
    jobs = Jobs(state_db, version="v-old")
    jobs.enqueue(make_task_key("extract", "doc-g-3", "cfg-v1"), "extract", now=1000.0)
    jobs.enqueue(make_task_key("extract", "doc-g-4", "cfg-v1"), "extract", now=1001.0)

    counts = check_version_gate(state_db, allow=True)
    assert counts == {"v-old": 2}

    row = state_db.query_one(
        "SELECT value FROM meta WHERE key = 'version_gate_ack'"
    )
    assert row is not None
    ack = json.loads(row["value"])
    assert ack["to"] == software_version()
    assert ack["from"] == ["v-old"]
    assert "at" in ack

    # The confirmation is recorded: a later call without the flag passes.
    assert check_version_gate(state_db) == {"v-old": 2}


def test_gate_ignores_ack_recorded_against_other_version(state_db: Database) -> None:
    """An ack is only valid for the version it was recorded *against*: a
    further upgrade re-triggers the gate even for previously-confirmed
    versions (conservative)."""
    Jobs(state_db, version="v-old").enqueue(
        make_task_key("extract", "doc-g-5", "cfg-v1"), "extract", now=1000.0
    )
    state_db.execute(
        "INSERT INTO meta (key, value) VALUES ('version_gate_ack', ?)",
        (json.dumps({"to": "other-version", "from": ["v-old"], "at": 1.0}),),
    )
    with pytest.raises(VersionGateError) as ei:
        check_version_gate(state_db)
    assert ei.value.foreign == {"v-old": 1}


def test_gate_treats_corrupt_ack_as_absent(state_db: Database) -> None:
    Jobs(state_db, version="v-old").enqueue(
        make_task_key("extract", "doc-g-6", "cfg-v1"), "extract", now=1000.0
    )
    state_db.execute(
        "INSERT INTO meta (key, value) VALUES ('version_gate_ack', 'not-json{')"
    )
    with pytest.raises(VersionGateError):
        check_version_gate(state_db)


def test_gate_counts_legacy_null_jobs_as_foreign(state_db: Database) -> None:
    Jobs(state_db).enqueue(make_task_key("extract", "doc-g-7", "cfg-v1"), "extract",
                           now=1000.0)
    state_db.execute("UPDATE jobs SET created_by_version = NULL")
    with pytest.raises(VersionGateError) as ei:
        check_version_gate(state_db)
    assert ei.value.foreign == {LEGACY_VERSION: 1}
    # ...but the operator can confirm them too.
    assert check_version_gate(state_db, allow=True) == {LEGACY_VERSION: 1}
    assert check_version_gate(state_db) == {LEGACY_VERSION: 1}


def test_gate_reasks_when_new_foreign_version_appears(state_db: Database) -> None:
    """The ack covers exactly the versions confirmed; a *new* foreign version
    (a further upgrade) re-triggers the gate, and the previously-confirmed
    set alone does not cover it."""
    jobs = Jobs(state_db, version="v-old")
    jobs.enqueue(make_task_key("extract", "doc-g-8", "cfg-v1"), "extract", now=1000.0)
    assert check_version_gate(state_db, allow=True) == {"v-old": 1}

    Jobs(state_db, version="v-newer").enqueue(
        make_task_key("extract", "doc-g-9", "cfg-v1"), "extract", now=2000.0
    )
    with pytest.raises(VersionGateError) as ei:
        check_version_gate(state_db)
    assert ei.value.foreign == {"v-old": 1, "v-newer": 1}

    # One confirmation covers both; both are then recorded in the ack.
    assert check_version_gate(state_db, allow=True) == {"v-old": 1, "v-newer": 1}
    row = state_db.query_one("SELECT value FROM meta WHERE key = 'version_gate_ack'")
    ack = json.loads(row["value"])  # type: ignore[index]
    assert ack["from"] == ["v-newer", "v-old"]
