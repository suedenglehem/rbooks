"""Backup, restore, and verification of a running system (M7, PRD §14).

A backup is a self-contained snapshot under one destination directory::

    <dest>/
      manifest.json    checksum manifest, written LAST — a destination
                       without a complete manifest is a partial backup and
                       :func:`restore_backup` refuses it
      config.yaml      the config in use, secrets redacted
      state/library.db SQLite *consistent snapshot* (sqlite3 backup API,
                       correct against a live WAL database)
      state/job_logs/  retained per-job failure logs
      qdrant/          Qdrant local storage, copied only while no Qdrant
                       client holds the storage lock (publication quiesced)
      archive/         content-addressed source store
      artifacts/       extraction + embedding artifacts

Immutable files (archive, and Qdrant storage while locked) are hardlinked
onto the same device and byte-copied across devices; either way the backup
survives accidental loss of the original directory.

Internal-HDD backups recover accidental loss but are NOT independent
disaster backups (PRD §14). ``paths.backup_root`` (or ``--to``) may point at
external storage, but nothing here assumes or pretends such a target exists.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .archive import archive_relpath
from .artifacts import commit_bytes, fsync_dir
from .config import Config
from .db import Database
from .embeddings import Embedder
from .indexing import FieldCond, IndexFilter, QdrantOps
from .jobs import Jobs
from .locks import QDRANT_LOCK_NAME
from .migrations import current_version
from .retrieval import search

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
CONFIG_NAME = "config.yaml"

_CHUNK = 1 << 20
_PROGRESS_EVERY = 2000

_log = logging.getLogger("library_rag.backup")


class BackupError(RuntimeError):
    """A backup/restore/verify operation was refused or failed."""


@dataclass(frozen=True)
class CheckResult:
    """One named verification check: name, pass/fail, human detail."""

    name: str
    ok: bool
    detail: str


# --- low-level file helpers ------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _same_device(a: Path, b: Path) -> bool:
    # *b* usually does not exist yet (fresh backup copy): a file's device is
    # the device of its containing directory.
    if not b.exists():
        b = b.parent
    return os.stat(a).st_dev == os.stat(b).st_dev


def _copy_one(src: Path, dst: Path, *, prefer_link: bool) -> str:
    """Copy one file (hardlink when *prefer_link* and on the same device)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if prefer_link and _same_device(src, dst):
        os.link(src, dst)
        mode = "link"
    else:
        with src.open("rb") as fin, dst.open("wb") as fout:
            shutil.copyfileobj(fin, fout, length=_CHUNK)
            fout.flush()
            os.fsync(fout.fileno())
        mode = "copy"
    if dst.stat().st_size != src.stat().st_size:
        raise BackupError(f"size mismatch after copying {src} -> {dst}")
    return mode


def _copy_tree(
    src: Path, dst: Path, *, skip_names: frozenset[str] = frozenset()
) -> tuple[int, int]:
    """Recursively copy *src* into *dst*; returns (n_linked, n_copied)."""
    n_link = n_copy = 0
    for root, _dirs, files in os.walk(src):
        base = Path(root)
        for name in files:
            if name in skip_names:
                continue
            s = base / name
            d = dst / s.relative_to(src)
            if _copy_one(s, d, prefer_link=True) == "link":
                n_link += 1
            else:
                n_copy += 1
            if (n_link + n_copy) % _PROGRESS_EVERY == 0:
                _log.info(
                    "backup progress",
                    extra={"extra": {"files": n_link + n_copy, "last": str(d)}},
                )
    return n_link, n_copy


def _record(
    manifest: dict[str, Any],
    path: Path,
    dest: Path,
    kind: str,
    *,
    sha: str | None = None,
    sha_source: str | None = None,
) -> None:
    """Append one file to the manifest's checksum list."""
    rel = path.relative_to(dest).as_posix()
    if sha is None:
        sha, source = _sha256_file(path), "computed"
    else:
        source = sha_source or "computed"
    manifest["files"].append(
        {
            "relpath": rel,
            "kind": kind,
            "size": path.stat().st_size,
            "sha256": sha,
            "sha256_source": source,
        }
    )


def _path_sha(rel: str) -> tuple[str, str] | None:
    """If *rel* follows the archive layout ``<sha[:2]>/<sha>``, the path IS the
    checksum: return it (``sha256_source='path'``) instead of re-hashing."""
    parts = rel.split("/")
    if len(parts) == 2 and len(parts[1]) == 64 and parts[0] == parts[1][:2]:
        return parts[1], "path"
    return None


# --- Qdrant quiescence gate -------------------------------------------------------


def _qdrant_lock_path(cfg: Config) -> Path | None:
    p = cfg.services.qdrant_path
    return Path(p) / QDRANT_LOCK_NAME if p is not None else None


def acquire_qdrant_lock(cfg: Config) -> int | None:
    """Acquire the exclusive flock Qdrant local clients take on their storage.

    Returns the fd to keep open for the whole backup (closing releases the
    lock), or None when no local storage exists yet (nothing to quiesce).
    Raises :class:`BackupError` when another client holds the lock — a
    running worker — because PRD §14 snapshots Qdrant while publication is
    paused.
    """
    lock = _qdrant_lock_path(cfg)
    if lock is None:
        raise BackupError(
            "remote Qdrant mode has no local storage to snapshot; use a local "
            "qdrant_path (embedded) deployment for backups"
        )
    if not lock.exists():
        return None
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise BackupError(
            "Qdrant storage is locked by a running client; pause ingestion and "
            "stop the worker, then retry"
        ) from None
    return fd


# --- backup -----------------------------------------------------------------------


def _counts(db: Database) -> dict[str, int]:
    specs = (
        ("documents", "SELECT COUNT(*) AS n FROM documents"),
        ("source_revisions", "SELECT COUNT(*) AS n FROM source_revisions"),
        ("chunks", "SELECT COUNT(*) AS n FROM chunks"),
        ("jobs", "SELECT COUNT(*) AS n FROM jobs"),
        (
            "jobs_failed",
            "SELECT COUNT(*) AS n FROM jobs "
            "WHERE state IN ('permanent_failed','retryable_failed')",
        ),
        (
            "publications_active",
            "SELECT COUNT(*) AS n FROM publications WHERE state = 'active'",
        ),
        (
            "active_expected_points",
            "SELECT COALESCE(SUM(expected_points), 0) AS n FROM publications "
            "WHERE state = 'active'",
        ),
    )
    counts: dict[str, int] = {}
    for key, sql in specs:
        row = db.query_one(sql)
        assert row is not None, f"count query returned no row: {sql}"
        counts[key] = int(row["n"])
    return counts


def _backup_state(db: Database, state_root: Path, dest: Path, manifest: dict[str, Any]) -> None:
    """SQLite consistent snapshot via the backup API (PRD §14)."""
    state_dir = dest / "state"
    state_dir.mkdir(exist_ok=True)
    snap = state_dir / "library.db"
    dst = sqlite3.connect(str(snap))
    try:
        db.conn.backup(dst)
        # Stand the snapshot up on its own (no WAL sidecars to list).
        dst.execute("PRAGMA journal_mode=DELETE")
    finally:
        dst.close()
    _record(manifest, snap, dest, "state")
    # Retained per-job failure logs: debugging evidence for restored state.
    src_logs = state_root / "job_logs"
    if src_logs.is_dir():
        for log_file in sorted(src_logs.iterdir()):
            if log_file.is_file():
                d = dest / "state" / "job_logs" / log_file.name
                _copy_one(log_file, d, prefer_link=True)
                _record(manifest, d, dest, "job_log")


def _backup_qdrant(cfg: Config, dest: Path, manifest: dict[str, Any]) -> None:
    qpath = cfg.services.qdrant_path
    if qpath is None or not Path(qpath).is_dir():
        manifest["qdrant"] = "absent"
        return
    qsrc = Path(qpath)
    manifest["qdrant"] = "local"
    _copy_tree(qsrc, dest / "qdrant", skip_names=frozenset({QDRANT_LOCK_NAME}))
    for root, _dirs, files in os.walk(dest / "qdrant"):
        for name in files:
            _record(manifest, Path(root) / name, dest, "qdrant")


def _backup_tree(src_root: Path, dest: Path, kind: str, manifest: dict[str, Any]) -> None:
    if not src_root.is_dir():
        return
    dest_dir = dest / kind
    _copy_tree(src_root, dest_dir)
    for root, _dirs, files in os.walk(dest_dir):
        for name in files:
            p = Path(root) / name
            sha, sha_source = None, "computed"
            if kind == "archive":
                hit = _path_sha(p.relative_to(dest_dir).as_posix())
                if hit is not None:
                    sha, sha_source = hit
            _record(manifest, p, dest, kind, sha=sha, sha_source=sha_source)


def _backup_config(dest: Path, manifest: dict[str, Any], config_source: Path | None) -> None:
    if config_source is None or not config_source.is_file():
        manifest["config"] = {
            "source": None,
            "note": "no config file in use (built-in defaults)",
        }
        return
    src = config_source
    with src.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise BackupError(f"config file root must be a mapping: {src}")
    services = raw.get("services")
    redacted = False
    if isinstance(services, dict) and services.get("api_token"):
        services["api_token"] = "REDACTED"
        redacted = True
    target = dest / CONFIG_NAME
    commit_bytes(target, yaml.safe_dump(raw, sort_keys=True).encode("utf-8"))
    manifest["config"] = {"source": str(src), "api_token_redacted": redacted}
    _record(manifest, target, dest, "config")


def create_backup(
    db: Database,
    cfg: Config,
    dest: Path,
    *,
    include_contents: bool = True,
    config_source: Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Create a backup under *dest* and return its manifest.

    The manifest is written LAST, so a destination without a complete
    ``manifest.json`` is partial and :func:`restore_backup` refuses it.
    """
    if dest.exists():
        if any(dest.iterdir()):
            raise BackupError(f"backup destination {dest} exists and is not empty")
    else:
        dest.mkdir(parents=True)
    ts = time.time() if now is None else now
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds"),
        "schema_version": current_version(db),
        "paused": Jobs(db).is_paused(),
        "include_contents": include_contents,
        "counts": _counts(db),
        "files": [],
    }

    # Quiesce gate first: while this lock is held, no worker can publish (and,
    # since the worker holds the Qdrant client for its whole run, nothing can
    # be writing archive/artifact files either).
    lock_fd = acquire_qdrant_lock(cfg)
    started = time.monotonic()
    try:
        _backup_state(db, cfg.paths.state_root, dest, manifest)
        _backup_qdrant(cfg, dest, manifest)
        if include_contents:
            _backup_tree(cfg.paths.archive_root, dest, "archive", manifest)
            _backup_tree(cfg.paths.artifact_root, dest, "artifacts", manifest)
        _backup_config(dest, manifest, config_source)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)

    manifest["elapsed_seconds"] = round(time.monotonic() - started, 1)
    commit_bytes(
        dest / MANIFEST_NAME,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    fsync_dir(dest)
    _log.info(
        "backup complete",
        extra={
            "extra": {
                "dest": str(dest),
                "files": len(manifest["files"]),
                "elapsed_seconds": manifest["elapsed_seconds"],
            }
        },
    )
    return manifest


# --- restore ----------------------------------------------------------------------


def load_manifest(backup_dir: Path) -> dict[str, Any]:
    """Read and sanity-check the manifest of *backup_dir*."""
    path = Path(backup_dir) / MANIFEST_NAME
    if not path.is_file():
        raise BackupError(
            f"{backup_dir} has no {MANIFEST_NAME}; a backup is written "
            "manifest-last, so this directory is a partial backup"
        )
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != MANIFEST_VERSION:
        raise BackupError(f"unsupported or corrupt manifest in {backup_dir}")
    return manifest


def _write_restored_config(backup_dir: Path, target: Path) -> None:
    """Write ``<target>/config.yaml``: the backed-up config with every managed
    root (except the shared model_root) re-pointed at the restored tree."""
    cfg_src = backup_dir / CONFIG_NAME
    if not cfg_src.is_file():
        return
    with cfg_src.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        return
    paths = raw.setdefault("paths", {})
    for key, subdir in (
        ("archive_root", "archive"),
        ("artifact_root", "artifacts"),
        ("state_root", "state"),
        ("qdrant_root", "qdrant"),
        ("scratch_root", "scratch"),
    ):
        paths[key] = str(target / subdir)
    paths.pop("backup_root", None)  # never back up into a restored tree
    raw.setdefault("services", {})["qdrant_path"] = str(target / "qdrant")
    for subdir in ("archive", "artifacts", "state", "qdrant", "scratch"):
        (target / subdir).mkdir(parents=True, exist_ok=True)
    with (target / CONFIG_NAME).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=True)


def restore_backup(backup_dir: Path, target: Path, *, force: bool = False) -> dict[str, Any]:
    """Materialize the backup under an isolated *target* directory.

    Restored files are hardlinked onto the same device and byte-copied
    otherwise. A rewritten ``config.yaml`` (and a manifest noting the
    rewrite) makes the tree self-contained, so
    ``library-rag verify --config <target>/config.yaml --backup <target>``
    exercises the restored system (M7 gate).
    """
    backup_dir = Path(backup_dir)
    target = Path(target)
    manifest = load_manifest(backup_dir)
    if target.exists() and not (force and not any(target.iterdir())):
        raise BackupError(
            f"restore target {target} exists; restore is always into a "
            "fresh directory (pass --force for an existing EMPTY one)"
        )
    target.mkdir(parents=True, exist_ok=True)  # --force: existing but EMPTY

    n_link = n_copy = 0
    for entry in manifest.get("files", []):
        src = backup_dir / entry["relpath"]
        if not src.is_file():
            raise BackupError(f"backup file missing: {src} (incomplete backup)")
        if _copy_one(src, target / entry["relpath"], prefer_link=True) == "link":
            n_link += 1
        else:
            n_copy += 1
    # The manifest itself travels with the tree (verify re-checks it here).
    _copy_one(backup_dir / MANIFEST_NAME, target / MANIFEST_NAME, prefer_link=True)
    _write_restored_config(backup_dir, target)

    manifest = dict(manifest)
    manifest["restore"] = {
        "target": str(target),
        "files_linked": n_link,
        "files_copied": n_copy,
        "rewritten": [CONFIG_NAME],
    }
    commit_bytes(
        target / MANIFEST_NAME,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    fsync_dir(target)
    return manifest


# --- verify -----------------------------------------------------------------------


def _check_db(db: Database) -> CheckResult:
    row = db.query_one("PRAGMA integrity_check")
    integrity = str(row["integrity_check"]) if row is not None else "error"
    return CheckResult(
        "db.integrity",
        integrity == "ok",
        f"PRAGMA integrity_check: {integrity}; schema_version={current_version(db)}",
    )


def _check_archive_links(db: Database, cfg: Config) -> CheckResult:
    missing: list[str] = []
    total = 0
    for r in db.query(
        "SELECT rev_id, sha256, size_bytes FROM source_revisions ORDER BY rev_id"
    ):
        total += 1
        path = cfg.paths.archive_root / archive_relpath(r["sha256"])
        if not path.is_file() or path.stat().st_size != int(r["size_bytes"]):
            missing.append(str(r["rev_id"]))
    if total == 0:
        return CheckResult("archive.links", True, "no source revisions registered")
    detail = f"{total - len(missing)}/{total} revisions present at expected size"
    if missing:
        detail += f"; missing: {missing[:5]}{'…' if len(missing) > 5 else ''}"
    return CheckResult("archive.links", not missing, detail)


def _check_artifact_links(db: Database, cfg: Config) -> CheckResult:
    missing: list[str] = []
    checked = 0
    for table in ("source_units", "embedding_batches"):
        for r in db.query(
            f"SELECT artifact_relpath FROM {table} WHERE artifact_relpath IS NOT NULL"
        ):
            checked += 1
            if not (cfg.paths.artifact_root / str(r["artifact_relpath"])).is_file():
                missing.append(str(r["artifact_relpath"]))
    detail = f"{checked - len(missing)}/{checked} referenced artifact files present"
    if missing:
        detail += f"; missing: {missing[:5]}{'…' if len(missing) > 5 else ''}"
    return CheckResult("artifacts.links", not missing, detail)


def _check_qdrant_index(db: Database, qdrant: QdrantOps) -> CheckResult:
    row = db.query_one(
        "SELECT COALESCE(SUM(expected_points), 0) AS n FROM publications "
        "WHERE state = 'active'"
    )
    assert row is not None
    expected = int(row["n"])
    if not qdrant.collection_exists():
        return CheckResult(
            "qdrant.index",
            expected == 0,
            "collection absent"
            + ("" if expected == 0 else f" but {expected} active points expected"),
        )
    active = qdrant.count(IndexFilter.all(FieldCond("active", "eq", True)))
    total = qdrant.count(IndexFilter())
    ok = expected == 0 or active == expected
    detail = (
        f"active points={active} expected={expected}; "
        f"total points={total} (superseded publications may linger inactive)"
    )
    return CheckResult("qdrant.index", ok, detail)


def _check_search_smoke(
    db: Database,
    cfg: Config,
    qdrant: QdrantOps,
    embedder: Embedder | None,
    query: str,
) -> CheckResult:
    try:
        result = search(db, cfg, qdrant, embedder, query)
    except Exception as exc:  # a smoke check must report, not raise
        return CheckResult("search.smoke", False, f"search raised {type(exc).__name__}: {exc}")
    note = f" (degraded: {result.degraded_reason})" if result.degraded else ""
    return CheckResult("search.smoke", True, f"{len(result.passages)} passages{note}")


def _is_wal_database(path: Path) -> bool:
    """True if the file's SQLite header declares WAL mode (bytes 18/19 = 2).

    A backed-up state DB is a self-contained DELETE-mode snapshot; the first
    ``Database.connect`` on it flips the header to WAL. That is legitimate
    drift for a booted tree, not corruption, so such files are size-checked
    only (integrity is covered by the db.integrity check on the live DB).
    """
    try:
        with path.open("rb") as fh:
            header = fh.read(20)
    except OSError:
        return False
    return len(header) >= 20 and header[18] == 2 and header[19] == 2


def _check_backup_manifest(backup_dir: Path, full_checksums: bool) -> CheckResult:
    try:
        manifest = load_manifest(backup_dir)
    except BackupError as exc:
        return CheckResult("backup.checksums", False, str(exc))
    rewritten = set((manifest.get("restore") or {}).get("rewritten", []))
    bad: list[str] = []
    total = 0
    for entry in manifest.get("files", []):
        total += 1
        path = Path(backup_dir) / entry["relpath"]
        if not path.is_file():
            bad.append(f"{entry['relpath']} (missing)")
            continue
        if entry["relpath"] in rewritten:
            continue  # restored trees rewrite config.yaml on purpose
        if path.stat().st_size != entry["size"]:
            bad.append(f"{entry['relpath']} (size)")
            continue
        if entry.get("sha256_source") == "path" and not full_checksums:
            continue  # content-addressed: the path IS the checksum
        if entry["kind"] == "state" and _is_wal_database(path):
            continue  # booted tree: header legitimately flipped to WAL
        if _sha256_file(path) != entry["sha256"]:
            bad.append(f"{entry['relpath']} (sha256)")
    if bad:
        return CheckResult(
            "backup.checksums",
            False,
            f"{len(bad)}/{total} files bad: {bad[:5]}{'…' if len(bad) > 5 else ''}",
        )
    note = "" if full_checksums else " (archive verified by name+size; --full-checksums re-hashes)"
    return CheckResult("backup.checksums", True, f"{total} files verified{note}")


def verify_system(
    cfg: Config,
    db: Database,
    qdrant: QdrantOps | None,
    *,
    search_query: str | None = None,
    backup_dir: Path | None = None,
    full_checksums: bool = False,
    embedder: Embedder | None = None,
) -> list[CheckResult]:
    """Run the verification suite; one :class:`CheckResult` per check.

    *qdrant* may be None (e.g. the local storage is locked by a running
    client) — the index check then reports failure instead of crashing.
    """
    results: list[CheckResult] = [
        _check_db(db),
        _check_archive_links(db, cfg),
        _check_artifact_links(db, cfg),
    ]
    if qdrant is None:
        results.append(
            CheckResult(
                "qdrant.index",
                False,
                "Qdrant client unavailable (local storage locked by a running "
                "client, or remote server unreachable)",
            )
        )
    elif not qdrant.ping():
        results.append(CheckResult("qdrant.index", False, "Qdrant unreachable"))
    else:
        results.append(_check_qdrant_index(db, qdrant))
    if search_query is not None:
        if qdrant is None:
            results.append(CheckResult("search.smoke", False, "skipped: no Qdrant client"))
        else:
            results.append(_check_search_smoke(db, cfg, qdrant, embedder, search_query))
    if backup_dir is not None:
        results.append(_check_backup_manifest(backup_dir, full_checksums))
    return results


def verify_backup_manifest(backup_dir: Path, full_checksums: bool) -> CheckResult:
    """Standalone manifest-integrity check (no live system needed)."""
    return _check_backup_manifest(backup_dir, full_checksums)


__all__ = [
    "CONFIG_NAME",
    "MANIFEST_NAME",
    "MANIFEST_VERSION",
    "BackupError",
    "CheckResult",
    "acquire_qdrant_lock",
    "create_backup",
    "load_manifest",
    "restore_backup",
    "verify_backup_manifest",
    "verify_system",
]
