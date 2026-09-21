"""Worker lifecycle: pidfile, live-process detection, stale lock cleanup.

M7 slice 10. Two operator flows:

* ``library-rag stop`` — graceful shutdown of a running ingest worker
  (SIGTERM, escalating to SIGKILL after ``--timeout``);
* ``library-rag ingest --force`` — forced start: kill a stuck ingest worker
  still holding the Qdrant local lock, drop a stale pidfile, and flip
  ``running`` jobs whose owner process is provably dead back to ``pending``
  without waiting out the 300 s lease TTL.

The Qdrant local (file) client holds an exclusive ``flock`` on
``<qdrant_root>/.lock`` for its lifetime (the same lock ``backup`` probes
before snapshotting); the kernel releases it automatically when the process
dies, so a lock with *no live holder* is never a problem. What ``--force``
handles is the inverse: a **live** process that holds the lock but no longer
makes progress (hung), plus the bookkeeping a dead worker leaves behind.

Safety: we only ever signal a pid whose ``/proc/<pid>/cmdline`` identifies it
as a ``library-rag ingest`` worker. Anything else holding the lock (e.g. a
``serve`` app, or an operator's own tool) is refused, not killed.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import time
from pathlib import Path

from .config import Config

__all__ = [
    "PIDFILE_NAME",
    "QDRANT_LOCK_NAME",
    "descendant_pids",
    "find_flock_holders",
    "force_release_qdrant_lock",
    "is_ingest_worker_pid",
    "kill_pids",
    "pid_alive",
    "pid_cmdline",
    "pidfile_path",
    "read_pidfile",
    "remove_pidfile",
    "remove_pidfile_if_ours",
    "remove_stale_pidfile",
    "write_pidfile",
]

PIDFILE_NAME = "ingest_worker.pid"
# Qdrant local (file) client's exclusive flock filename — the one place in the
# package this constant belongs; backup.py imports it from here (locks must
# not depend on backup, or jobs.py's pid_alive import cycles back through it).
QDRANT_LOCK_NAME = ".lock"


# --- live process detection ----------------------------------------------------


def pid_alive(pid: int) -> bool:
    """True when *pid* refers to a live (not zombie) process, even another user's.

    A zombie has exited but not been reaped by its parent. It holds no locks
    and does no work, so it counts as dead — otherwise a long-lived parent
    (a supervisor, a test harness, an init-less daemon) would keep a crashed
    worker looking alive to ``stop`` / ``--force`` / lease reclamation.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return True  # vanished between the kill probe and the read
    for line in status.splitlines():
        if line.startswith("State:"):
            # "State:\tZombie (zombie)" / "State:\tRunning (running)" ...
            return line.split(maxsplit=1)[1].strip()[:1] != "Z"
    return True


def pid_cmdline(pid: int) -> str:
    """The process's command line as one string (NULs -> spaces); '' if gone."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()


def is_ingest_worker_pid(pid: int) -> bool:
    """True when *pid*'s command line is a ``library-rag ingest`` worker.

    Matches on argv, not substrings: the word ``ingest`` must be its own
    argument (so a config file merely named ``ingest-*.yaml`` does not
    match), and ``library-rag`` must appear in the command. This is the
    identity check that lets ``stop``/``--force`` distinguish the worker
    from a pid-reused process or any other library-rag command (``serve``,
    ``discover``, ...).
    """
    cmdline = pid_cmdline(pid)
    parts = cmdline.split()
    return "library-rag" in cmdline and "ingest" in parts


def _ppid_map() -> dict[int, list[int]]:
    """pid -> children, from a single /proc scan."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            status = Path("/proc", entry, "status").read_text()
        except OSError:
            continue  # process vanished mid-scan
        for line in status.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split(maxsplit=1)[1])
                children.setdefault(ppid, []).append(int(entry))
                break
    return children


def descendant_pids(root: int) -> list[int]:
    """All descendant pids of *root* (BFS over /proc; excludes *root* itself)."""
    children = _ppid_map()
    out: list[int] = []
    seen = {root}
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in seen:
                seen.add(child)
                out.append(child)
                stack.append(child)
    return out


# --- pidfile -------------------------------------------------------------------


def pidfile_path(paths: object) -> Path:
    """``<scratch_root>/ingest_worker.pid`` — written by the worker itself."""
    return Path(paths.scratch_root) / PIDFILE_NAME  # type: ignore[attr-defined]


def write_pidfile(paths: object, pid: int) -> None:
    pidfile_path(paths).write_text(f"{pid}\n")


def read_pidfile(paths: object) -> int | None:
    try:
        return int(pidfile_path(paths).read_text().strip())
    except (OSError, ValueError):
        return None


def remove_pidfile(paths: object) -> None:
    with contextlib.suppress(OSError):
        pidfile_path(paths).unlink()


def remove_pidfile_if_ours(paths: object, pid: int) -> None:
    """Remove the pidfile only if it names *pid* (never clobber another's)."""
    if read_pidfile(paths) == pid:
        remove_pidfile(paths)


def remove_stale_pidfile(paths: object) -> bool:
    """Drop the pidfile only when its pid is provably dead.

    Returns True when a file was removed. A *live* pid — even one that is not
    an ingest worker (i.e. the pid was reused after a crash) — keeps the
    file: it is not provably stale, and deleting evidence would mask the
    refusal ``stop`` makes for such a pid. The next worker start overwrites
    the file with its own pid regardless.
    """
    pid = read_pidfile(paths)
    if pid is None:
        return False
    if not pid_alive(pid):
        remove_pidfile(paths)
        return True
    return False


# --- Qdrant local lock -----------------------------------------------------------


def _qdrant_lock_path(cfg: Config) -> Path | None:
    if cfg.services.qdrant_path is None:
        return None
    return Path(cfg.services.qdrant_path) / QDRANT_LOCK_NAME


def _lock_is_held(lock: Path) -> bool:
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def find_flock_holders(lock: Path) -> list[int]:
    """Pids of live processes with an open fd on *lock* (sorted, unique).

    Scans ``/proc/<pid>/fd`` — the kernel keeps a lock's owner list private
    to itself, so the fd table is the only portable source.
    """
    if not lock.exists():
        return []
    real = os.path.realpath(lock)
    holders: set[int] = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        fd_dir = Path("/proc", entry, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd_dir / fd)
            except OSError:
                continue
            if os.path.realpath(target) == real:
                holders.add(int(entry))
    return sorted(holders)


def kill_pids(
    pids: list[int], timeout: float = 30.0, kill_wait: float = 5.0
) -> tuple[list[int], list[int]]:
    """SIGTERM *pids*; escalate to SIGKILL any survivor after *timeout* s.

    Returns ``(signaled, escalated)`` — the pids that received SIGTERM and
    the subset that needed SIGKILL.
    """
    alive = [p for p in pids if pid_alive(p)]
    for p in alive:
        with contextlib.suppress(ProcessLookupError):
            os.kill(p, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while any(pid_alive(p) for p in alive) and time.monotonic() < deadline:
        time.sleep(0.2)
    escalated = [p for p in alive if pid_alive(p)]
    for p in escalated:
        with contextlib.suppress(ProcessLookupError):
            os.kill(p, signal.SIGKILL)
    deadline = time.monotonic() + kill_wait
    while any(pid_alive(p) for p in escalated) and time.monotonic() < deadline:
        time.sleep(0.1)
    return alive, escalated


def force_release_qdrant_lock(cfg: Config, kill_timeout: float = 30.0) -> list[int]:
    """Ensure nothing blocks a new worker from taking the Qdrant local lock.

    Probe non-blocking; when held, resolve the holder(s) via /proc and:

    * refuse (``RuntimeError``) if any holder is **not** an ingest worker —
      that is a legitimate concurrent user (``serve``, a backup, ...);
    * otherwise SIGTERM (then SIGKILL) the worker(s), wait for the kernel to
      release the lock, and verify the probe passes.

    Returns the pids that were killed (empty when the lock was free or the
    storage directory does not exist yet).
    """
    lock = _qdrant_lock_path(cfg)
    if lock is None or not lock.exists():
        return []
    if not _lock_is_held(lock):
        return []
    holders = find_flock_holders(lock)
    if not holders:
        raise RuntimeError(
            f"Qdrant lock {lock} is held but no holder process was found — "
            "investigate manually (a wedged lock can only be cleared by rebooting)"
        )
    foreign = [p for p in holders if not is_ingest_worker_pid(p)]
    if foreign:
        detail = "; ".join(
            f"pid {p} ({pid_cmdline(p) or 'unknown'})" for p in foreign
        )
        raise RuntimeError(
            f"the Qdrant lock is held by a process that is not an ingest "
            f"worker — refusing to kill it: {detail}. Stop it deliberately, "
            "then retry."
        )
    _signaled, escalated = kill_pids(holders, kill_timeout)
    # The kernel releases the flock as soon as the last holder's fd closes
    # (process exit); verify rather than assume.
    if _lock_is_held(lock):
        raise RuntimeError(
            f"Qdrant lock {lock} is still held after killing {holders} — "
            "investigate manually"
        )
    if escalated:
        print(f"force: escalated to SIGKILL for pid(s) {escalated}")
    return holders
