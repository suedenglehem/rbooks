"""M7 backup / restore / verify (PRD §14).

A backup is a self-contained, manifest-last snapshot: a consistent SQLite
snapshot (backup API), Qdrant local storage copied only while its storage
lock is free (publication quiesced), hardlinked archive/artifacts, and a
secret-redacted config. Restore materializes the tree into an isolated
directory with a rewritten config that makes it self-contained, and
``verify`` exercises the restored system (the M7 gate).

Real embedded (local) Qdrant storage is used, as in the M4 local-mode
tests: no server, no Docker, no GPU.
"""

from __future__ import annotations

import fcntl
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from fixtures import ingest_and_publish, make_pdf
from library_rag.archive import archive_relpath
from library_rag.backup import (
    CONFIG_NAME,
    MANIFEST_NAME,
    MANIFEST_VERSION,
    QDRANT_LOCK_NAME,
    BackupError,
    create_backup,
    load_manifest,
    restore_backup,
    verify_backup_manifest,
    verify_system,
)
from library_rag.config import Config, load_config
from library_rag.db import Database, db_path_for
from library_rag.embeddings import FakeEmbedder
from library_rag.indexing import RealQdrantOps

_PAGES: list[str | None] = [
    "Chapter one introduces the lighthouse keeper.",
    "Chapter two follows the keeper's daughter to sea.",
    "Chapter three ends with the storm and the lamp.",
]


@pytest.fixture
def local_library(
    state_db: Database, base_config: Config
) -> tuple[Database, Config]:
    """One real book (3 pages) scanned and published into embedded Qdrant.

    The Qdrant client is closed before yielding: a backup requires the
    storage lock to be free (publication quiesced, PRD §14), and the
    verify tests open a fresh client against the (restored) storage.
    """
    base_config.embedding.fake = True
    base_config.services.qdrant_path = str(base_config.paths.qdrant_root)
    q = RealQdrantOps(base_config)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    src = base_config.paths.source_roots[0] / "book.pdf"
    make_pdf(src, _PAGES)
    try:
        ingest_and_publish(state_db, base_config, src, qdrant=q, embedder=emb)
    finally:
        q.close()
    return state_db, base_config


def _config_source_file(cfg: Config, tmp_path: Path, **services: Any) -> Path:
    """A YAML config file matching the live config's roots (backup input)."""
    raw: dict[str, Any] = {
        "paths": {
            "source_roots": [str(p) for p in cfg.paths.source_roots],
            "archive_root": str(cfg.paths.archive_root),
            "artifact_root": str(cfg.paths.artifact_root),
            "state_root": str(cfg.paths.state_root),
            "qdrant_root": str(cfg.paths.qdrant_root),
            "model_root": str(cfg.paths.model_root),
            "scratch_root": str(cfg.paths.scratch_root),
        },
        "services": {"qdrant_path": str(cfg.services.qdrant_path), **services},
        "embedding": {
            "fake": True,
            "model_revision": "fake-v1",
            "model_name": "fake",
            "dimensions": cfg.embedding.dimensions,
        },
        "chunking": {
            "target_tokens": cfg.chunking.target_tokens,
            "overlap_tokens": cfg.chunking.overlap_tokens,
        },
        "answer": {"fake": True},
    }
    p = tmp_path / "source-config.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    return p


# --- 1. Creation: layout, checksums, hardlinks, quiescence --------------------


def test_create_backup_layout(local_library: tuple[Database, Config], tmp_path: Path) -> None:
    db, cfg = local_library
    dest = tmp_path / "backup"
    manifest = create_backup(db, cfg, dest, now=1_700_000_000)

    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["created_at"] == datetime.fromtimestamp(
        1_700_000_000, tz=UTC
    ).isoformat(timespec="seconds")
    assert manifest["paused"] is False
    assert manifest["include_contents"] is True
    assert manifest["qdrant"] == "local"
    assert "elapsed_seconds" in manifest

    counts = manifest["counts"]
    assert counts["documents"] == 1
    assert counts["source_revisions"] == 1
    assert counts["publications_active"] == 1
    assert counts["jobs_failed"] == 0
    assert counts["active_expected_points"] > 0

    # The manifest is written last and re-reads cleanly.
    assert (dest / MANIFEST_NAME).is_file()
    loaded = load_manifest(dest)
    assert loaded["counts"] == counts
    assert [f["relpath"] for f in loaded["files"]] == [
        f["relpath"] for f in manifest["files"]
    ]

    by_kind: dict[str, list[dict[str, Any]]] = {}
    for f in manifest["files"]:
        by_kind.setdefault(f["kind"], []).append(f)
    assert any(f["relpath"] == "state/library.db" for f in by_kind["state"])
    assert by_kind["qdrant"]  # local storage materialized
    assert by_kind["archive"]
    assert by_kind["artifacts"]
    assert "config" not in by_kind  # no config_source passed

    # Archive entries are content-addressed: the path IS the checksum.
    arch = by_kind["archive"][0]
    assert arch["sha256_source"] == "path"
    assert len(arch["sha256"]) == 64

    # No Qdrant lock file travels with the backup; the storage itself does.
    qfiles = [p.name for p in (dest / "qdrant").rglob("*") if p.is_file()]
    assert ".lock" not in qfiles
    assert qfiles

    # Same-device files are hardlinked, not byte-copied.
    src = cfg.paths.archive_root / archive_relpath(arch["sha256"])
    assert (dest / arch["relpath"]).stat().st_ino == src.stat().st_ino
    assert verify_backup_manifest(dest, full_checksums=True).ok


def test_backup_refuses_locked_qdrant(
    local_library: tuple[Database, Config], tmp_path: Path
) -> None:
    db, cfg = local_library
    assert cfg.services.qdrant_path is not None
    lock = Path(cfg.services.qdrant_path) / QDRANT_LOCK_NAME
    assert lock.is_file()
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BackupError, match="locked"):
            create_backup(db, cfg, tmp_path / "backup")
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_backup_refuses_remote_qdrant_mode(
    local_library: tuple[Database, Config], tmp_path: Path
) -> None:
    db, cfg = local_library
    cfg.services.qdrant_path = None
    with pytest.raises(BackupError, match="remote Qdrant"):
        create_backup(db, cfg, tmp_path / "backup")


def test_backup_refuses_nonempty_destination(
    local_library: tuple[Database, Config], tmp_path: Path
) -> None:
    db, cfg = local_library
    dest = tmp_path / "backup"
    dest.mkdir()
    (dest / "junk.txt").write_text("leftover", encoding="utf-8")
    with pytest.raises(BackupError, match="not empty"):
        create_backup(db, cfg, dest)


def test_state_only_backup_skips_contents(
    local_library: tuple[Database, Config], tmp_path: Path
) -> None:
    db, cfg = local_library
    dest = tmp_path / "backup"
    manifest = create_backup(db, cfg, dest, include_contents=False)
    assert manifest["include_contents"] is False
    assert not (dest / "archive").exists()
    assert not (dest / "artifacts").exists()
    kinds = {f["kind"] for f in manifest["files"]}
    assert "state" in kinds and "qdrant" in kinds
    assert "archive" not in kinds and "artifacts" not in kinds
    assert verify_backup_manifest(dest, full_checksums=False).ok


# --- 2. M7 gate: restore into an isolated directory and verify ----------------


def test_restore_and_verify(local_library: tuple[Database, Config], tmp_path: Path) -> None:
    db, cfg = local_library
    source_cfg = _config_source_file(cfg, tmp_path)
    assert load_config(source_cfg).paths.state_root == cfg.paths.state_root

    dest = tmp_path / "backup"
    manifest = create_backup(db, cfg, dest, config_source=source_cfg, now=1_700_000_000)
    assert manifest["config"]["api_token_redacted"] is False
    assert any(f["kind"] == "config" for f in manifest["files"])

    target = tmp_path / "restored"
    rm = restore_backup(dest, target)
    assert rm["restore"]["target"] == str(target)
    assert rm["restore"]["rewritten"] == [CONFIG_NAME]
    assert rm["restore"]["files_linked"] + rm["restore"]["files_copied"] == len(
        manifest["files"]
    )

    # The rewritten config makes the restored tree self-contained.
    rcfg = load_config(target / CONFIG_NAME)
    assert rcfg.paths.state_root == target / "state"
    assert rcfg.paths.qdrant_root == target / "qdrant"
    assert rcfg.paths.archive_root == target / "archive"
    assert rcfg.paths.artifact_root == target / "artifacts"
    assert rcfg.paths.scratch_root == target / "scratch"
    assert rcfg.paths.backup_root is None
    assert rcfg.services.qdrant_path == str(target / "qdrant")
    assert rcfg.paths.source_roots == cfg.paths.source_roots  # not rewritten

    # Exercise the restored system: state DB + embedded Qdrant + search
    # smoke + manifest integrity, all against the restored tree.
    db2 = Database.connect(db_path_for(rcfg.paths.state_root))
    q2 = RealQdrantOps(rcfg)
    try:
        results = verify_system(
            rcfg,
            db2,
            q2,
            search_query="the lighthouse keeper",
            backup_dir=target,
            embedder=FakeEmbedder(rcfg.embedding.dimensions),
        )
    finally:
        q2.close()
        db2.close()
    assert [r.name for r in results] == [
        "db.integrity",
        "archive.links",
        "artifacts.links",
        "qdrant.index",
        "search.smoke",
        "backup.checksums",
    ]
    failed = [f"{r.name}: {r.detail}" for r in results if not r.ok]
    assert not failed, failed


def test_backup_redacts_api_token(
    local_library: tuple[Database, Config], tmp_path: Path
) -> None:
    db, cfg = local_library
    source_cfg = _config_source_file(cfg, tmp_path, api_token="hunter2-secret")
    dest = tmp_path / "backup"
    manifest = create_backup(db, cfg, dest, config_source=source_cfg)
    assert manifest["config"]["api_token_redacted"] is True
    text = (dest / CONFIG_NAME).read_text(encoding="utf-8")
    assert "hunter2-secret" not in text
    assert "REDACTED" in text


# --- 3. Corruption and partial backups ----------------------------------------


def test_backup_detects_corruption(local_library: tuple[Database, Config], tmp_path: Path) -> None:
    db, cfg = local_library
    dest = tmp_path / "backup"
    manifest = create_backup(db, cfg, dest)
    assert verify_backup_manifest(dest, full_checksums=False).ok

    # (a) A manifest-listed file disappears.
    victim = next(f for f in manifest["files"] if f["kind"] == "qdrant")
    (dest / victim["relpath"]).unlink()
    r = verify_backup_manifest(dest, full_checksums=False)
    assert r.ok is False and "(missing)" in r.detail
    with pytest.raises(BackupError, match="backup file missing"):
        restore_backup(dest, tmp_path / "restored")

    # (b) A file's bytes change. The backup file is a hardlink to the live
    # tree here, so replace it via a fresh inode (temp file + rename) before
    # mutating, or the live source would be corrupted too.
    victim2 = next(f for f in manifest["files"] if f["kind"] == "artifacts")
    p = dest / victim2["relpath"]
    data = p.read_bytes()
    assert len(data) > 2
    tmp = p.with_name(p.name + ".corrupt-tmp")
    mid = len(data) // 2
    tmp.write_bytes(data[:mid] + b"\x00" + data[mid + 1 :])
    tmp.replace(p)
    r = verify_backup_manifest(dest, full_checksums=False)
    assert r.ok is False and "(sha256)" in r.detail


def test_restore_refuses_partial_backup(tmp_path: Path) -> None:
    partial = tmp_path / "partial"
    (partial / "state").mkdir(parents=True)
    (partial / "state" / "library.db").write_bytes(b"not a database")
    with pytest.raises(BackupError, match="partial"):
        restore_backup(partial, tmp_path / "restored")


def test_restore_refuses_existing_target(
    local_library: tuple[Database, Config], tmp_path: Path
) -> None:
    db, cfg = local_library
    dest = tmp_path / "backup"
    create_backup(db, cfg, dest)
    target = tmp_path / "restored"
    target.mkdir()
    (target / "existing.txt").write_text("x", encoding="utf-8")
    with pytest.raises(BackupError, match="fresh directory"):
        restore_backup(dest, target)
    # An existing EMPTY target is allowed with force.
    empty = tmp_path / "restored-empty"
    empty.mkdir()
    rm = restore_backup(dest, empty, force=True)
    assert rm["restore"]["target"] == str(empty)


# --- 4. Verification degradation ----------------------------------------------


def test_verify_degrades_without_qdrant_client(
    local_library: tuple[Database, Config], tmp_path: Path
) -> None:
    db, cfg = local_library
    results = verify_system(cfg, db, None, search_query="keeper")
    by = {r.name: r for r in results}
    assert by["db.integrity"].ok
    assert by["archive.links"].ok
    assert by["artifacts.links"].ok
    assert by["qdrant.index"].ok is False
    assert "unavailable" in by["qdrant.index"].detail
    assert by["search.smoke"].ok is False
    assert "skipped" in by["search.smoke"].detail
