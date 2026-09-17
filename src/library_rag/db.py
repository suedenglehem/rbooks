"""SQLite connection setup and a thin transactional wrapper.

The database uses the durable-state settings from PRD §6: WAL journaling,
``foreign_keys=ON``, a ``busy_timeout`` for concurrent coordinators/workers, and
``synchronous=FULL``. Transactions are short and explicit (autocommit is the
default; :meth:`Database.transaction` issues ``BEGIN IMMEDIATE`` for a write
section) and bracket the commit with the database-commit crash hooks.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from .crash import CrashPhase, fire

__all__ = ["Database", "db_path_for"]


def db_path_for(state_root: Path) -> Path:
    """Where the state database lives: under the (SSD) state root."""
    return state_root / "library.db"


class Database:
    """A configured SQLite connection with an explicit-transaction helper.

    The connection runs in autocommit mode (``isolation_level=None``) so reads
    do not hold locks; write sections opt in to a single short
    ``BEGIN IMMEDIATE`` transaction via :meth:`transaction`.

    The connection is shared across threads (``check_same_thread=False``):
    the FastAPI reader runs sync endpoints in a thread pool. A re-entrant lock
    serializes statements and whole transactions, so a cross-thread interleave
    can never split one connection's statement sequence (transaction bodies
    call :meth:`execute`, hence re-entrant, not plain).
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock | None = None) -> None:
        self._conn = conn
        self._lock = lock if lock is not None else threading.RLock()

    @classmethod
    def connect(cls, path: Path) -> Database:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # PRAGMA settings are connection-scoped; set them once here.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=FULL;")
        return cls(conn)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return cast("sqlite3.Row | None", self._conn.execute(sql, tuple(params)).fetchone())

    @contextmanager
    def transaction(self) -> Iterator[Database]:
        """Run *body* inside a single short ``BEGIN IMMEDIATE`` transaction.

        The *body* (the ``with`` block) runs inside the open transaction. The
        commit is bracketed by crash hooks: if a ``BEFORE_DB_COMMIT`` hook fires
        and raises — or the body itself raises — the change is rolled back (an
        uncommitted transaction is not durable); if an ``AFTER_DB_COMMIT`` hook
        raises, the commit has already landed and the data stays.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE;")
            committed = False
            try:
                yield self
                fire(CrashPhase.BEFORE_DB_COMMIT)
                self._conn.commit()
                committed = True
            except BaseException:
                if not committed:
                    self._conn.execute("ROLLBACK;")
                raise
            fire(CrashPhase.AFTER_DB_COMMIT)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
