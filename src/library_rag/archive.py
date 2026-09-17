"""Content-addressed source archive (PRD §5, §8A).

Authoritative originals are copied into a store laid out by their SHA-256
(``<root>/<2-hex>/<sha256>``), so identical bytes are stored exactly once and the
archive path is a pure function of content. Ingestion is a *streamed copy* —
never a hardlink — because a hardlink to a mutable source would let the source
modify the "immutable" archive. The copy is checksum-verified, fsync'd, and made
visible via the atomic rename in :mod:`library_rag.artifacts`.

The archive is append-only here: this module never deletes stored objects.
Garbage collection (with reference checks) is a later milestone (M7).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .artifacts import commit_stream

__all__ = [
    "ArchiveError",
    "IngestResult",
    "archive_path_for",
    "archive_relpath",
    "ingest_source",
    "stream_hash",
]

_CHUNK = 1 << 20  # 1 MiB streaming unit


class ArchiveError(Exception):
    """Raised when an archive copy cannot be verified."""


@dataclass(frozen=True)
class IngestResult:
    sha256: str
    size_bytes: int
    dest: Path
    already_present: bool


def archive_relpath(sha256: str) -> str:
    """Path of *sha256* within the archive (relative to the archive root)."""
    return f"{sha256[:2]}/{sha256}"


def archive_path_for(root: Path, sha256: str) -> Path:
    return root / sha256[:2] / sha256


def stream_hash(path: Path, chunk: int = _CHUNK) -> tuple[str, int]:
    """SHA-256 of *path* read in a streaming fashion; return (hexdigest, size)."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def _copy_stream(src: Path, dst: Path) -> None:
    with src.open("rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout, length=_CHUNK)
        fout.flush()
        os.fsync(fout.fileno())


def ingest_source(root: Path, src: Path, expect_sha: str | None = None) -> IngestResult:
    """Copy *src* into the content-addressed archive and verify it.

    * The source is hashed (streamed); if *expect_sha* is given and differs, the
      copy is refused (guards against a file changing while being read).
    * If the destination already exists at the expected size, ingestion is a
      cheap dedup no-op (the exact-duplicate case).
    * Otherwise the source is streamed to a temp file in the destination
      directory, fsync'd, re-hashed and verified, then atomically renamed into
      place (with the rename crash hooks).
    """
    sha, size = stream_hash(src)
    if expect_sha is not None and sha != expect_sha:
        raise ArchiveError(f"source hash mismatch: expected {expect_sha}, got {sha}")

    dest = archive_path_for(root, sha)
    if dest.exists():
        if dest.stat().st_size == size:
            return IngestResult(sha, size, dest, already_present=True)
        # Size disagrees with the recorded layout: treat as corrupt and recopy.
        dest.unlink()

    def _write(tmp: Path) -> None:
        _copy_stream(src, tmp)
        # Verify the written copy before it is ever made visible.
        got, got_size = stream_hash(tmp)
        if got != sha or got_size != size:
            raise ArchiveError(f"archive copy hash mismatch for {sha}")

    # commit_stream fsyncs the temp, then atomically renames it into place and
    # fsyncs the directory (firing the rename crash hooks).
    commit_stream(dest, _write)
    return IngestResult(sha, size, dest, already_present=False)


def verify_archived(root: Path, sha256: str, size_bytes: int | None = None) -> bool:
    """Return True if the archived object exists and hashes to *sha256*."""
    dest = archive_path_for(root, sha256)
    if not dest.exists():
        return False
    if size_bytes is not None and dest.stat().st_size != size_bytes:
        return False
    got, _ = stream_hash(dest)
    return got == sha256
