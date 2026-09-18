"""Source discovery: streaming walk, fast checks, archive, register (PRD §8A).

The scan is a *report*, not a mutation of catalog state it cannot stand behind:

* discovery streams the tree (``os.walk``), pruning configured ignore sets and
  never following symlinks, so memory is O(depth), not O(tree);
* a file is only re-hashed when its size *and* mtime changed since the last
  scan (``scan_state`` fast check);
* format is validated from content (magic bytes), never trusted from the
  extension — a ``.pdf`` that is not a PDF is reported ``invalid`` and skipped;
* files that change during hashing are skipped for this pass (they will be
  stable on the next scan);
* missing files (known aliases no longer visible) are *reported*, never
  deleted; a missing root or a failed mount sentinel makes the whole root
  ``mount_unavailable`` and skips it entirely.

New and changed revisions get an ``extract`` job enqueued (idempotent on the
task key, so a crashed re-scan cannot double-enqueue).
"""

from __future__ import annotations

import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .archive import ingest_source, stream_hash
from .catalog import Format, RegistrationStatus, register_source
from .config import Config
from .db import Database
from .identity import make_task_key, normalize_path
from .jobs import Jobs

__all__ = [
    "STAGE_CHUNK",
    "STAGE_EXTRACT",
    "STAGE_OCR",
    "ScanReport",
    "detect_format",
    "scan_root",
    "scan_roots",
]

STAGE_EXTRACT = "extract"
STAGE_OCR = "ocr"
STAGE_CHUNK = "chunk"

_CANDIDATE_SUFFIXES = {".pdf", ".epub"}


@dataclass
class ScanReport:
    """Outcome of scanning one source root (JSON-friendly: paths as strings)."""

    root: str
    mount_unavailable: bool = False
    discovered: int = 0
    unchanged: int = 0
    new_documents: int = 0
    new_revisions: int = 0
    aliases: int = 0
    invalid: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    changed_during_scan: list[str] = field(default_factory=list)
    jobs_enqueued: int = 0


def detect_format(path: Path) -> Format | None:
    """Validate the format from content magic bytes (PRD §8A), or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return None
    if head.startswith(b"%PDF-"):
        return Format.PDF
    if head[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(path) as zf:
                data = zf.read("mimetype")
        except (KeyError, zipfile.BadZipFile, OSError):
            return None
        return Format.EPUB if data.strip() == b"application/epub+zip" else None
    return None


def _upsert_scan_state(
    db: Database, path: str, size: int, mtime: float, sha: str, fmt: Format, rev_id: str
) -> None:
    with db.transaction():
        db.execute(
            """
            INSERT INTO scan_state (path, size_bytes, mtime, sha256, format, rev_id, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size_bytes = excluded.size_bytes,
                mtime = excluded.mtime,
                sha256 = excluded.sha256,
                format = excluded.format,
                rev_id = excluded.rev_id,
                last_seen_at = excluded.last_seen_at
            """,
            (path, size, mtime, sha, fmt.value, rev_id, time.time()),
        )


def _process_file(db: Database, cfg: Config, jobs: Jobs | None, path: Path, report: ScanReport) -> None:
    report.discovered += 1
    norm = normalize_path(path)
    try:
        st = path.stat()
    except OSError:
        report.missing.append(norm)
        return
    # Fast check: unchanged size+mtime since the last verified hash => skip.
    state = db.query_one("SELECT sha256, size_bytes, mtime FROM scan_state WHERE path = ?", (norm,))
    if (
        state is not None
        and state["sha256"] is not None
        and int(state["size_bytes"]) == st.st_size
        and float(state["mtime"]) == st.st_mtime
    ):
        report.unchanged += 1
        return
    try:
        sha, _size = stream_hash(path)
    except OSError:
        report.missing.append(norm)
        return
    # Detect files that changed *during* hashing: skip this pass, retry next scan.
    st_after = path.stat()
    if (st_after.st_size, st_after.st_mtime) != (st.st_size, st.st_mtime):
        report.changed_during_scan.append(norm)
        return
    fmt = detect_format(path)
    if fmt is None:
        report.invalid.append(norm)
        return
    ingest = ingest_source(cfg.paths.archive_root, path, expect_sha=sha)
    reg = register_source(db, norm, ingest.sha256, ingest.size_bytes, fmt)
    _upsert_scan_state(db, norm, st_after.st_size, st_after.st_mtime, ingest.sha256, fmt, reg.rev_id)
    if reg.status is RegistrationStatus.NEW_DOCUMENT:
        report.new_documents += 1
    elif reg.status is RegistrationStatus.NEW_REVISION:
        report.new_revisions += 1
    else:
        report.aliases += 1
    if reg.status in (RegistrationStatus.NEW_DOCUMENT, RegistrationStatus.NEW_REVISION) and jobs is not None:
        jobs.enqueue(
            make_task_key(STAGE_EXTRACT, reg.rev_id, ingest.sha256),
            STAGE_EXTRACT,
            input_id=reg.rev_id,
            input_version=ingest.sha256,
        )
        report.jobs_enqueued += 1


def scan_root(db: Database, cfg: Config, root: Path, jobs: Jobs | None = None) -> ScanReport:
    """Scan one source root; return its report. Never deletes catalog content."""
    report = ScanReport(root=normalize_path(root))
    sentinel = cfg.mount_sentinels.get(str(root))
    if sentinel is not None and not Path(sentinel).exists():
        report.mount_unavailable = True
        return report
    if not root.is_dir():
        report.mount_unavailable = True
        return report

    visible: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        # Prune ignore sets and symlinked directories in place (never follow).
        dirnames[:] = [
            d
            for d in sorted(dirnames)
            if d not in cfg.scan.ignore_dirs and not os.path.islink(os.path.join(dirpath, d))
        ]
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.is_symlink() or name in cfg.scan.ignore_files:
                continue
            if p.suffix.lower() not in _CANDIDATE_SUFFIXES:
                continue
            norm = normalize_path(p)
            visible.add(norm)
            _process_file(db, cfg, jobs, p, report)

    # Known aliases under this root that are no longer visible: report only.
    prefix = normalize_path(root) + "/"
    rows = db.query("SELECT path FROM path_aliases")
    gone = [r["path"] for r in rows if r["path"].startswith(prefix) and r["path"] not in visible]
    report.missing = sorted(set(gone) | set(report.missing))
    return report


def scan_roots(db: Database, cfg: Config, jobs: Jobs | None = None) -> list[ScanReport]:
    """Scan every configured source root; one report each (JSON-friendly)."""
    return [scan_root(db, cfg, root, jobs) for root in cfg.paths.source_roots]
