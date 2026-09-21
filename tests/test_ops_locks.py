"""Worker lifecycle ops (M7 slice 10): stop / forced-start building blocks.

Covers the pieces ``library-rag stop`` and ``library-rag ingest --force``
rely on: live-process detection, the ingest-worker identity check (argv,
not substring), pidfile helpers, Qdrant local-lock holder discovery, the
refuse-vs-kill rule of ``force_release_qdrant_lock``, and immediate
reclamation of ``running`` jobs whose owner pid is provably dead.

Process fixtures are short-lived child pythons that hold a ``flock`` on a
temp lock file; every test cleans them up in ``finally``.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from library_rag.config import Config
from library_rag.db import Database
from library_rag.identity import make_task_key
from library_rag.jobs import Jobs
from library_rag.locks import (
    descendant_pids,
    find_flock_holders,
    force_release_qdrant_lock,
    is_ingest_worker_pid,
    pid_alive,
    pid_cmdline,
    pidfile_path,
    read_pidfile,
    remove_pidfile,
    remove_pidfile_if_ours,
    remove_stale_pidfile,
    write_pidfile,
)

FLOCK_CODE = (
    "import fcntl, sys, time\n"
    "fd = open(sys.argv[1], 'a+')\n"
    "fcntl.flock(fd, fcntl.LOCK_EX)\n"
    "print('locked', flush=True)\n"
    "time.sleep(300)\n"
)


def _dead_pid() -> int:
    """A pid beyond pid_max — guaranteed to be unused."""
    return int(Path("/proc/sys/kernel/pid_max").read_text().strip()) + 1


def _wait_for_holders(lock: Path, expected: list[int], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if find_flock_holders(lock) == expected:
            return
        time.sleep(0.05)
    pytest.fail(f"lock holders did not become {expected} (got {find_flock_holders(lock)})")


# --- live-process detection ---------------------------------------------------


def test_pid_alive_self_and_dead() -> None:
    assert pid_alive(os.getpid()) is True
    assert pid_alive(0) is False
    assert pid_alive(-1) is False
    assert pid_alive(_dead_pid()) is False


def test_pid_alive_zombie_counts_as_dead() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.terminate()
    try:
        # Wait until the child is actually a zombie (unreaped by us), then
        # pid_alive must report it dead: a zombie holds no locks.
        deadline = time.monotonic() + 10.0
        while True:
            try:
                lines = (Path("/proc") / str(proc.pid) / "status").read_text().splitlines()
            except OSError:
                pytest.fail("child vanished before becoming a zombie")
            state = next((line for line in lines if line.startswith("State:")), "")
            if state.split(maxsplit=1)[1].strip()[:1] == "Z":
                break
            if time.monotonic() > deadline:
                pytest.fail("child did not become a zombie in time")
            time.sleep(0.05)
        assert pid_alive(proc.pid) is False
    finally:
        proc.wait(timeout=10)
    assert pid_alive(proc.pid) is False


def test_pid_cmdline_self() -> None:
    assert "python" in pid_cmdline(os.getpid())
    assert pid_cmdline(_dead_pid()) == ""


def test_is_ingest_worker_pid_rejects_this_process() -> None:
    # This pytest process's argv has "ingest" as at most a substring of a
    # test name, never as its own word — and never alongside "library-rag".
    assert is_ingest_worker_pid(os.getpid()) is False


# --- pidfile helpers ------------------------------------------------------------


def test_pidfile_roundtrip(base_config: Config) -> None:
    paths = base_config.paths
    paths.scratch_root.mkdir(parents=True, exist_ok=True)
    assert read_pidfile(paths) is None
    write_pidfile(paths, 12345)
    assert read_pidfile(paths) == 12345
    pidfile_path(paths).write_text("not-a-pid\n")
    assert read_pidfile(paths) is None
    pidfile_path(paths).write_text("6789\n")
    remove_pidfile(paths)
    assert read_pidfile(paths) is None


def test_remove_stale_pidfile(base_config: Config) -> None:
    paths = base_config.paths
    paths.scratch_root.mkdir(parents=True, exist_ok=True)
    assert remove_stale_pidfile(paths) is False  # no file at all

    write_pidfile(paths, _dead_pid())
    assert remove_stale_pidfile(paths) is True  # dead pid -> stale
    assert read_pidfile(paths) is None

    write_pidfile(paths, os.getpid())  # live, but NOT an ingest worker
    assert remove_stale_pidfile(paths) is False  # pid-reuse guard: keep it
    assert read_pidfile(paths) == os.getpid()


def test_remove_pidfile_if_ours(base_config: Config) -> None:
    paths = base_config.paths
    paths.scratch_root.mkdir(parents=True, exist_ok=True)
    write_pidfile(paths, 11111)
    remove_pidfile_if_ours(paths, 22222)
    assert read_pidfile(paths) == 11111  # someone else's file: untouched
    remove_pidfile_if_ours(paths, 11111)
    assert read_pidfile(paths) is None


# --- descendant resolution ------------------------------------------------------


def test_descendant_pids_finds_grandchild() -> None:
    code = (
        "import subprocess, sys\n"
        "p = subprocess.Popen(['sleep', '300'])\n"
        "print(p.pid, flush=True)\n"
        "p.wait()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True
    )
    grandchild = int(proc.stdout.readline())
    try:
        out = descendant_pids(proc.pid)
        assert proc.pid not in out  # excludes the root itself
        assert grandchild in out
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        with contextlib.suppress(ProcessLookupError):
            os.kill(grandchild, signal.SIGKILL)  # it would outlive its parent


# --- flock holder discovery -------------------------------------------------------


def test_find_flock_holders(tmp_path: Path) -> None:
    lock = tmp_path / "qlk" / ".lock"
    lock.parent.mkdir()
    lock.touch()
    proc = subprocess.Popen(
        [sys.executable, "-c", FLOCK_CODE, str(lock)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        proc.stdout.readline()  # 'locked'
        _wait_for_holders(lock, [proc.pid])
        assert is_ingest_worker_pid(proc.pid) is False  # plain python -c
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    assert find_flock_holders(lock) == []


def test_force_release_refuses_foreign_holder(base_config: Config, tmp_path: Path) -> None:
    lock_dir = tmp_path / "qdrant"
    lock_dir.mkdir()
    (lock_dir / ".lock").touch()
    proc = subprocess.Popen(
        [sys.executable, "-c", FLOCK_CODE, str(lock_dir / ".lock")],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        proc.stdout.readline()
        _wait_for_holders(lock_dir / ".lock", [proc.pid])
        cfg = base_config.model_copy(
            update={"services": base_config.services.model_copy(update={"qdrant_path": str(lock_dir)})}
        )
        with pytest.raises(RuntimeError, match="refusing to kill"):
            force_release_qdrant_lock(cfg)
        assert pid_alive(proc.pid)  # the foreign holder must survive
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_force_release_kills_stuck_worker(base_config: Config, tmp_path: Path) -> None:
    # A script literally named "library-rag", invoked as `library-rag ingest
    # <lock-dir>` — its argv matches the worker identity check the way a
    # real (or nohup'd uv-run) worker does.
    worker = tmp_path / "library-rag"
    worker.write_text(
        "#!/usr/bin/env python3\n"
        "import fcntl, sys, time\n"
        "fd = open(sys.argv[2] + '/.lock', 'a+')\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "print('locked', flush=True)\n"
        "time.sleep(300)\n"
    )
    worker.chmod(0o755)
    lock_dir = tmp_path / "qdrant"
    lock_dir.mkdir()
    (lock_dir / ".lock").touch()
    proc = subprocess.Popen([str(worker), "ingest", str(lock_dir)], stdout=subprocess.PIPE, text=True)
    try:
        proc.stdout.readline()
        _wait_for_holders(lock_dir / ".lock", [proc.pid])
        assert is_ingest_worker_pid(proc.pid) is True
        cfg = base_config.model_copy(
            update={"services": base_config.services.model_copy(update={"qdrant_path": str(lock_dir)})}
        )
        killed = force_release_qdrant_lock(cfg, kill_timeout=10.0)
        assert killed == [proc.pid]
        assert pid_alive(proc.pid) is False
        assert find_flock_holders(lock_dir / ".lock") == []
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_force_release_no_lock_file(base_config: Config, tmp_path: Path) -> None:
    cfg = base_config.model_copy(
        update={"services": base_config.services.model_copy(update={"qdrant_path": str(tmp_path / "nope")})}
    )
    assert force_release_qdrant_lock(cfg) == []  # directory does not exist


# --- dead-lease reclamation --------------------------------------------------------


def test_reclaim_dead_lease_owners(state_db: Database) -> None:
    dead = _dead_pid()
    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("extract", "doc-dl-1", "cfg-v1"), "extract", now=1000.0)
    w = jobs.claim(f"extract-fakehost-{dead}", ttl=300.0, now=1001.0)
    assert w is not None
    assert w.attempts == 1

    reclaimed = jobs.reclaim_dead_lease_owners(now=1002.0)
    assert reclaimed == 1
    row = state_db.query_one("SELECT state, lease_owner FROM jobs WHERE job_id = ?", (w.job_id,))
    assert row is not None
    assert row["state"] == "pending"
    assert row["lease_owner"] is None
    # A second pass finds nothing left to reclaim.
    assert jobs.reclaim_dead_lease_owners(now=1003.0) == 0


def test_reclaim_leaves_live_owner_alone(state_db: Database) -> None:
    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("extract", "doc-dl-2", "cfg-v1"), "extract", now=1000.0)
    w = jobs.claim(f"extract-self-{os.getpid()}", ttl=300.0, now=1001.0)
    assert w is not None
    # This process is alive: even if its token were stale, it may still commit.
    assert jobs.reclaim_dead_lease_owners(now=1002.0) == 0
    row = state_db.query_one("SELECT state FROM jobs WHERE job_id = ?", (w.job_id,))
    assert row is not None
    assert row["state"] == "running"


def test_reclaim_ignores_unrecognized_owner_format(state_db: Database) -> None:
    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("extract", "doc-dl-3", "cfg-v1"), "extract", now=1000.0)
    w = jobs.claim("mystery-owner", ttl=300.0, now=1001.0)
    assert w is not None
    # No `-<pid>` suffix: we cannot prove the owner dead, so the TTL handles it.
    assert jobs.reclaim_dead_lease_owners(now=1002.0) == 0
