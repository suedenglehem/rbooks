"""Content-addressed archive: streamed copy, dedup, verification, crash safety."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from library_rag import crash
from library_rag.archive import (
    ArchiveError,
    archive_path_for,
    archive_relpath,
    ingest_source,
    stream_hash,
    verify_archived,
)
from library_rag.crash import CrashPhase, SimulatedCrash


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_ingest_creates_content_addressed_file(roots: dict[str, Path], tmp_path: Path) -> None:
    root = roots["archive_root"]
    src = tmp_path / "book.pdf"
    data = b"%PDF-1.7 fake book bytes " * 100
    src.write_bytes(data)

    result = ingest_source(root, src)
    expected = _sha(data)
    assert result.sha256 == expected
    assert not result.already_present
    assert result.dest == archive_path_for(root, expected)
    assert result.dest.exists()
    assert archive_relpath(expected) == f"{expected[:2]}/{expected}"
    assert verify_archived(root, expected, len(data)) is True


def test_ingest_dedups_identical_content(
    roots: dict[str, Path], tmp_path: Path
) -> None:
    root = roots["archive_root"]
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"  # different path, identical bytes
    data = b"identical content"
    a.write_bytes(data)
    b.write_bytes(data)

    r1 = ingest_source(root, a)
    r2 = ingest_source(root, b)
    assert r1.sha256 == r2.sha256
    assert r1.already_present is False
    assert r2.already_present is True  # second copy is a no-op
    # Exactly one object on disk.
    files = list(root.rglob("*"))
    assert [f for f in files if f.is_file()] == [r1.dest]


def test_ingest_rejects_hash_mismatch(roots: dict[str, Path], tmp_path: Path) -> None:
    root = roots["archive_root"]
    src = tmp_path / "x.pdf"
    src.write_bytes(b"real bytes")
    wrong = "00" * 32
    with pytest.raises(ArchiveError):
        ingest_source(root, src, expect_sha=wrong)


def test_crash_before_rename_leaves_temp_not_destination(
    roots: dict[str, Path], tmp_path: Path
) -> None:
    root = roots["archive_root"]
    src = tmp_path / "c.pdf"
    data = b"will crash before rename"
    src.write_bytes(data)

    def _die() -> None:
        raise SimulatedCrash()

    crash.set_hook(CrashPhase.BEFORE_ARTIFACT_RENAME, _die)

    with pytest.raises(SimulatedCrash):
        ingest_source(root, src)

    dest = archive_path_for(root, _sha(data))
    assert not dest.exists()  # destination never became visible
    # A temp file was left behind in the destination directory to reconcile.
    leftovers = [p for p in dest.parent.iterdir() if p.name.startswith(".")]
    assert leftovers, "expected a leftover temp file to reconcile"


def test_stream_hash_reads_in_full(tmp_path: Path) -> None:
    p = tmp_path / "h.bin"
    data = bytes(range(256)) * 4096
    p.write_bytes(data)
    sha, size = stream_hash(p)
    assert sha == _sha(data)
    assert size == len(data)
