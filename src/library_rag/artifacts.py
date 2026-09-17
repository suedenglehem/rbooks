"""The durable artifact-commit protocol (PRD §7).

A write is durable only after this sequence:

1. write to a temporary file **on the same filesystem** as the destination,
2. flush and ``fsync`` the file,
3. atomically ``rename`` it onto the destination (POSIX rename is atomic),
4. ``fsync`` the containing directory so the rename itself is durable.

Crash injection points sit around the rename (see :mod:`library_rag.crash`).
If a crash lands before the rename, the temp file is left behind — a durable
output that reconciliation later removes or reuses — and the destination is
untouched, so readers never observe a partial artifact.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path

from .crash import CrashPhase, fire

__all__ = ["atomic_rename", "commit_bytes", "fsync_dir", "fsync_file", "make_tmp_path"]


def make_tmp_path(dest: Path) -> Path:
    """A temp path in the same directory (same filesystem) as *dest*."""
    return dest.with_name(f".{dest.name}.tmp-{uuid.uuid4().hex}")


def fsync_file(path: Path) -> None:
    """``fsync`` an existing file by reopening it read-only."""
    with path.open("rb") as fh:
        os.fsync(fh.fileno())


def fsync_dir(directory: Path) -> None:
    """``fsync`` a directory so a rename within it is durable."""
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_rename(src: Path, dest: Path) -> None:
    """Atomically move *src* onto *dest*, firing the rename crash hooks.

    After the rename the containing directory is fsync'd so the new directory
    entry is durable.
    """
    fire(CrashPhase.BEFORE_ARTIFACT_RENAME)
    os.rename(src, dest)
    fire(CrashPhase.AFTER_ARTIFACT_RENAME)
    fsync_dir(dest.parent)


def commit_bytes(dest: Path, data: bytes) -> None:
    """Durely write *data* to *dest* using the temp+fsync+rename+fsync-dir protocol."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = make_tmp_path(dest)
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    atomic_rename(tmp, dest)


def commit_stream(dest: Path, writer: Callable[[Path], None]) -> None:
    """Durely produce *dest* by having *writer* fill a temp file.

    *writer* receives the temp path and writes into it (used for large,
    streamed sources such as the content-addressed archive). The caller is
    responsible for verifying the temp contents before this returns; the
    rename (and its crash hooks) happen here.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = make_tmp_path(dest)
    writer(tmp)
    fsync_file(tmp)
    atomic_rename(tmp, dest)
