"""Crash-injection hooks.

The PRD (M1) requires crash-injection points at two boundaries:

* before/after the **atomic rename** that makes an artifact durable, and
* before/after a **database commit**.

These hooks are *disabled outside tests*: the process-global hook table is empty
in production, so :func:`fire` is a no-op. A test arms a hook with
:func:`set_hook` (usually raising :class:`SimulatedCrash`) to model a process
dying at exactly that instant, then asserts the durable state is consistent
(partial artifact never committed, uncommitted transaction rolled back).

Model of a simulated crash:

* ``BEFORE_DB_COMMIT`` fires just before ``conn.commit()``. Raising here means
  the commit never lands, so the surrounding ``transaction`` context manager
  rolls back — matching a SIGKILL before the commit record is durable.
* ``AFTER_DB_COMMIT`` fires just after. The commit is already durable; raising
  here models a crash after the write succeeded, so the data stays.
* ``BEFORE_ARTIFACT_RENAME`` / ``AFTER_ARTIFACT_RENAME`` bracket the atomic
  rename. Raising before the rename means the temp file is left behind (a
  durable output to reconcile) and the destination is untouched.
"""

from __future__ import annotations

import enum
from collections.abc import Callable

__all__ = ["CrashPhase", "SimulatedCrash", "clear_hooks", "fire", "is_armed", "set_hook"]


class CrashPhase(enum.StrEnum):
    """The injection points recognized by the durable-write and commit paths."""

    BEFORE_ARTIFACT_RENAME = "before_artifact_rename"
    AFTER_ARTIFACT_RENAME = "after_artifact_rename"
    BEFORE_DB_COMMIT = "before_db_commit"
    AFTER_DB_COMMIT = "after_db_commit"


class SimulatedCrash(RuntimeError):
    """Raised by an armed hook to model a process death at that exact point."""


# Process-global, intentionally empty in production. Only tests register hooks,
# and only for the duration of a single test (see the ``crash_hooks`` fixture).
_hooks: dict[CrashPhase, Callable[[], object]] = {}


def set_hook(phase: CrashPhase, fn: Callable[[], object] | None) -> None:
    """Arm (or, with ``None``, disarm) the hook for *phase*."""
    if fn is None:
        _hooks.pop(phase, None)
    else:
        _hooks[phase] = fn


def clear_hooks() -> None:
    """Disarm every hook. Called by the test fixture's teardown."""
    _hooks.clear()


def is_armed(phase: CrashPhase) -> bool:
    return phase in _hooks


def fire(phase: CrashPhase) -> None:
    """Fire the hook for *phase* if one is armed; a no-op in production."""
    fn = _hooks.get(phase)
    if fn is not None:
        fn()
