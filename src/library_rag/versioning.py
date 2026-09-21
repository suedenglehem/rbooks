"""Software version signature (M7 slice 9).

The *software version* is a content hash of the installed package source:
the SHA-256 over every non-cache file under the package root, hashed as
``(relative path, file sha256)`` pairs in sorted path order. It is
deterministic for a given tree, independent of git metadata (it works the
same from a checkout or an sdist), and changes whenever any shipped source
file changes — which is exactly the boundary where enqueued-but-unexecuted
work must be re-confirmed by the operator: a reboot + upgrade (or
rollback) must not silently execute jobs created by another code version.

The full digest is stored on job rows (``jobs.created_by_version``);
:func:`short_version` renders a stable 12-char prefix for messages.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

__all__ = ["short_version", "software_version", "software_version_for"]


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def software_version_for(root: Path) -> str:
    """Content hash of every non-cache file under *root* (test seam)."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
        h.update(b"\0")
    return h.hexdigest()


@lru_cache(maxsize=1)
def software_version() -> str:
    """The version of the running code (computed once per process)."""
    return software_version_for(_package_root())


def short_version(version: str) -> str:
    """Stable 12-char prefix for human-facing messages."""
    return version[:12]
