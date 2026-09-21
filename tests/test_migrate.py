"""M7 generation migration (PRD §14, ``library-rag migrate``).

A chunker or embedding configuration change invalidates the published
evidence. The migration planner must report *what* will be re-chunked,
re-embedded, and superseded; estimate whether the new generation fits
beside the current one (PRD line 190: two generations must fit, or a
documented maintenance window is required); and only with ``execute=True``
enqueue the same jobs the worker's reconcile pass would enqueue.
"""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest

from fixtures import publish_handbuilt
from library_rag.config import Config
from library_rag.db import Database
from library_rag.indexing import FakeQdrant
from library_rag.migrate import MigrationError, run_migration

_TEXTS = [
    "The first passage, about the lighthouse keeper.",
    "The second passage, about the keeper's daughter.",
]


@pytest.fixture
def one_book(state_db: Database, base_config: Config) -> tuple[Database, Config, FakeQdrant]:
    """One hand-built published book: two chunks, one active publication."""
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    publish_handbuilt(
        state_db,
        base_config,
        q,
        doc_id="docA",
        rev_id="revA",
        run_id="runA",
        texts=_TEXTS,
    )
    return state_db, base_config, q


def test_no_drift_is_a_noop(one_book: tuple[Database, Config, FakeQdrant]) -> None:
    db, cfg, _q = one_book
    r = run_migration(db, cfg)
    assert (r.rechunk_runs, r.rechunk_active_runs, r.reembed_runs) == (0, 0, 0)
    assert r.publications_to_supersede == 0
    assert r.migrating_points == 0
    assert r.active_points == 2
    assert r.fits is True
    assert r.maintenance_window_required is False
    assert r.enqueued_jobs == 0


def test_chunker_drift_rechunks_and_supersedes(
    one_book: tuple[Database, Config, FakeQdrant],
) -> None:
    db, cfg, _q = one_book
    cfg.chunking.target_tokens += 1
    r = run_migration(db, cfg)
    assert r.rechunk_runs == 1
    assert r.rechunk_active_runs == 1
    assert r.reembed_runs == 0
    assert r.publications_to_supersede == 1
    assert r.migrating_points == 2
    # Planning enqueues nothing.
    assert r.enqueued_jobs == 0
    assert db.query_one("SELECT 1 AS x FROM jobs WHERE stage = 'chunk'") is None
    # Executing enqueues exactly the re-chunk job (its own handoff re-drives
    # embed and publish from the new fingerprint). FakeQdrant has no storage
    # directory, so capacity is unknown and the window is "required" — the
    # explicit acceptance stands in for a real two-generation check.
    r = run_migration(db, cfg, execute=True, accept_maintenance_window=True)
    assert r.enqueued_jobs == 1
    assert db.query_one("SELECT 1 AS x FROM jobs WHERE stage = 'chunk'") is not None


def test_embedding_drift_reembeds(one_book: tuple[Database, Config, FakeQdrant]) -> None:
    db, cfg, _q = one_book
    cfg.embedding.dimensions += 1
    r = run_migration(db, cfg)
    assert r.rechunk_runs == 0
    assert r.reembed_runs == 1
    assert r.publications_to_supersede == 1
    assert r.migrating_points == 2


def test_fit_estimated_from_measured_store(
    one_book: tuple[Database, Config, FakeQdrant],
) -> None:
    db, cfg, _q = one_book
    cfg.chunking.target_tokens += 1
    # Simulate a non-empty local store: 8 MiB of point files.
    root = cfg.paths.qdrant_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "points.bin").write_bytes(b"\x00" * (8 * 1024 * 1024))
    r = run_migration(db, cfg)
    assert r.qdrant_storage_bytes == 8 * 1024 * 1024
    # All points migrate → the second generation is estimated at the size
    # of the current store.
    assert r.second_generation_bytes == 8 * 1024 * 1024
    assert r.fits is True  # the tmp volume has plenty of free space


def test_no_fit_refuses_without_acceptance(
    one_book: tuple[Database, Config, FakeQdrant],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, cfg, _q = one_book
    cfg.chunking.target_tokens += 1
    root = cfg.paths.qdrant_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "points.bin").write_bytes(b"\x00" * (8 * 1024 * 1024))
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=1024))
    r = run_migration(db, cfg)
    assert r.fits is False
    assert r.maintenance_window_required is True
    with pytest.raises(MigrationError, match="maintenance"):
        run_migration(db, cfg, execute=True)
    # The refusal enqueued nothing.
    assert db.query_one("SELECT 1 AS x FROM jobs WHERE stage = 'chunk'") is None
    # Explicit acceptance proceeds.
    r = run_migration(db, cfg, execute=True, accept_maintenance_window=True)
    assert r.enqueued_jobs == 1


def test_unknown_capacity_treated_as_not_fitting(
    one_book: tuple[Database, Config, FakeQdrant],
) -> None:
    db, cfg, _q = one_book
    cfg.chunking.target_tokens += 1
    # Remote mode: no local storage directory → the size is unknown.
    cfg.services.qdrant_path = str(cfg.paths.qdrant_root / "no-such-dir")
    r = run_migration(db, cfg)
    assert r.qdrant_storage_bytes is None
    assert r.fits is None
    assert r.maintenance_window_required is True
    assert any("cannot be verified" in n for n in r.notes)
    # Explicit acceptance still proceeds (the operator took the window).
    r = run_migration(db, cfg, execute=True, accept_maintenance_window=True)
    assert r.enqueued_jobs == 1
