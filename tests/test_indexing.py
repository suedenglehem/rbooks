"""M4 indexing: filters, point construction, the fake store, publication, lock,
and the post-crash reconciliation pass (PRD §8F)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from fixtures import ingest_and_publish, make_pdf
from library_rag.config import Config
from library_rag.db import Database
from library_rag.embeddings import SparseStats, bm25_weights
from library_rag.identity import point_id
from library_rag.indexing import (
    _PUBLISH_LOCK_TTL,
    COLLECTION,
    FakeQdrant,
    FieldCond,
    IndexFilter,
    IndexingError,
    PublicationError,
    QdrantPoint,
    RealQdrantOps,
    acquire_publish_lock,
    build_point,
    publish_generation,
    reconcile_publications,
    release_publish_lock,
)

_BOOK = "The quick brown fox jumps over the lazy dog."


def _pt(point_id_: str = "p1", dense: tuple[float, ...] = (1.0, 0.0),
        payload: dict[str, object] | None = None) -> QdrantPoint:
    return QdrantPoint(
        point_id=point_id_,
        payload={"doc_id": "d1", **(payload or {})},
        dense=dense,
        sparse=((), ()),
    )


# --- Filters -----------------------------------------------------------------


def test_filter_eq_ne_in_matches() -> None:
    p: dict[str, object] = {"pub_id": "p1", "active": True, "doc_id": "d1"}
    assert FieldCond("pub_id", "eq", "p1").matches(p)
    assert not FieldCond("pub_id", "eq", "p2").matches(p)
    assert not FieldCond("pub_id", "ne", "p1").matches(p)
    assert FieldCond("pub_id", "ne", "p2").matches(p)
    assert FieldCond("missing", "ne", "p1").matches(p)  # missing key satisfies ne
    assert FieldCond("pub_id", "in", ["p1", "p2"]).matches(p)
    assert not FieldCond("pub_id", "in", ["p2", "p3"]).matches(p)
    with pytest.raises(ValueError):
        FieldCond("pub_id", "in", "p1").matches(p)  # non-list value
    with pytest.raises(ValueError):
        FieldCond("pub_id", "like", "p%").matches(p)  # unknown op


def test_filter_to_qdrant_shape_and_matches_agree() -> None:
    f = IndexFilter.all(
        FieldCond("pub_id", "eq", "p1"),
        FieldCond("active", "ne", False),
        FieldCond("doc_id", "in", ["d1", "d2"]),
    )
    raw = f.to_qdrant()
    assert raw is not None
    # The qdrant-client stubs type Filter fields as a wide union; this test
    # inspects the concrete shape, so drop the stub typing here.
    qf: Any = raw
    # eq and in land in must, ne in must_not.
    assert [c.key for c in qf.must] == ["pub_id", "doc_id"]
    assert qf.must[0].match.value == "p1"
    assert qf.must[1].match.any == ["d1", "d2"]
    assert [c.key for c in qf.must_not] == ["active"]
    assert qf.must_not[0].match.value is False
    # The pure-Python evaluator agrees on every relevant payload.
    assert f.matches({"pub_id": "p1", "active": True, "doc_id": "d1"})
    assert not f.matches({"pub_id": "p2", "active": True, "doc_id": "d1"})
    assert not f.matches({"pub_id": "p1", "active": False, "doc_id": "d1"})
    assert not f.matches({"pub_id": "p1", "active": True, "doc_id": "d3"})
    assert IndexFilter.all().to_qdrant() is None
    assert IndexFilter.all().matches({})
    with pytest.raises(ValueError):
        IndexFilter.all(FieldCond("k", "in", "not-a-list")).to_qdrant()


# --- Point construction ---------------------------------------------------------


def test_build_point_payload_contract() -> None:
    stats = SparseStats(stats_sha="s", doc_count=2, avg_doc_len=2.0, df={"fox": 1})
    p = build_point(
        chunk_id="c1",
        gen_id="g1",
        pub_id="pub1",
        rev_id="r1",
        doc_id="d1",
        run_id="run1",
        model_revision="m1",
        embedding_sha_value="esh",
        stats_sha_value="s",
        dense=[0.5, -0.5],
        text="fox fox",
        stats=stats,
        k1=1.5,
        b=0.75,
    )
    assert p.point_id == point_id("c1", "g1")
    assert set(p.payload) == {
        "chunk_id", "rev_id", "doc_id", "run_id", "gen_id", "pub_id",
        "model_revision", "embedding_sha", "stats_sha",
    }
    # The active flag and the raw text are deliberately not payload fields.
    assert "active" not in p.payload and "text" not in p.payload
    assert p.dense == (0.5, -0.5)
    indices, values = bm25_weights("fox fox", stats, k1=1.5, b=0.75)
    assert p.sparse == (tuple(indices), tuple(values))


# --- FakeQdrant -----------------------------------------------------------------


def test_fake_qdrant_upsert_is_idempotent() -> None:
    q = FakeQdrant(dimensions=2)
    q.upsert([_pt()])
    q.upsert([_pt()])  # same point id again
    assert q.count(IndexFilter.all()) == 1


def test_fake_qdrant_dimension_mismatch() -> None:
    q = FakeQdrant(dimensions=2)
    with pytest.raises(IndexingError, match="dims"):
        q.upsert([_pt(dense=(1.0,))])


def test_fake_qdrant_set_active_merges_only_the_flag() -> None:
    q = FakeQdrant(dimensions=2)
    q.upsert([_pt(payload={"pub_id": "p1"})])
    q.set_active(IndexFilter.all(FieldCond("pub_id", "eq", "p1")), True)
    hits = q.dense_search([1.0, 0.0], limit=5, f=IndexFilter.all())
    assert len(hits) == 1
    assert hits[0].payload["active"] is True
    assert hits[0].payload["doc_id"] == "d1"  # untouched
    q.set_active(IndexFilter.all(FieldCond("pub_id", "eq", "OTHER")), True)
    assert q.count(IndexFilter.all(FieldCond("active", "eq", True))) == 1


def test_fake_qdrant_delete_requires_a_filter() -> None:
    q = FakeQdrant(dimensions=2)
    with pytest.raises(IndexingError, match="unfiltered"):
        q.delete(IndexFilter.all())


def test_fake_qdrant_dense_search_orders_and_filters() -> None:
    q = FakeQdrant(dimensions=2)
    q.upsert([
        _pt("a", dense=(1.0, 0.0), payload={"pub_id": "pa"}),
        _pt("b", dense=(0.5, 0.0), payload={"pub_id": "pb"}),
        _pt("c", dense=(0.9, 0.0), payload={"pub_id": "pc"}),
    ])
    # The "active" flag only exists where set_active has set it (publish flips
    # every point of a publication), so all three points get a flag.
    q.set_active(IndexFilter.all(FieldCond("pub_id", "eq", "pa")), True)
    q.set_active(IndexFilter.all(FieldCond("pub_id", "eq", "pb")), True)
    q.set_active(IndexFilter.all(FieldCond("pub_id", "eq", "pc")), False)
    hits = q.dense_search(
        [1.0, 0.0], limit=2, f=IndexFilter.all(FieldCond("active", "eq", True))
    )
    assert [h.point_id for h in hits] == ["a", "b"]  # (-score, id) order


def test_fake_qdrant_sparse_search_skips_zero_overlap() -> None:
    q = FakeQdrant(dimensions=1)

    def _sp(pt_id: str, sparse: tuple[tuple[int, ...], tuple[float, ...]]) -> QdrantPoint:
        return QdrantPoint(point_id=pt_id, payload={}, dense=(1.0,), sparse=sparse)

    q.upsert([
        _sp("hit", ((7, 9), (2.0, 1.0))),
        _sp("miss", ((7, 9), (0.0, 0.0))),
    ])
    hits = q.sparse_search([7], [1.0], limit=5, f=IndexFilter.all())
    assert [h.point_id for h in hits] == ["hit"]


# --- publish_generation ----------------------------------------------------------


def _run_id(db: Database, rev_id: str) -> str:
    row = db.query_one("SELECT run_id FROM extraction_runs WHERE rev_id = ?", (rev_id,))
    assert row is not None
    return str(row["run_id"])


def test_publish_generation_publishes_and_is_idempotent(
    state_db: Database, base_config: Config
) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    path = make_pdf(base_config.paths.source_roots[0] / "book.pdf", [_BOOK])
    rev, pub = ingest_and_publish(state_db, base_config, path, qdrant=q)

    row = state_db.query_one("SELECT COUNT(*) AS n FROM chunks WHERE rev_id = ?", (rev,))
    assert row is not None
    n = int(row["n"])
    assert n > 0
    row = state_db.query_one("SELECT state FROM publications WHERE pub_id = ?", (pub,))
    assert row is not None
    assert row["state"] == "active"
    assert q.count(IndexFilter.all()) == n
    assert q.count(IndexFilter.all(FieldCond("active", "eq", True))) == n
    gen = state_db.query_one(
        "SELECT * FROM index_generations WHERE run_id = ?", (_run_id(state_db, rev),)
    )
    assert gen is not None and gen["state"] == "ready"
    assert int(gen["point_count"]) == n

    # A direct re-run of the same generation is a no-op (re-upsert idempotent).
    assert (
        publish_generation(state_db, base_config, q, rev_id=rev,
                           run_id=_run_id(state_db, rev), on_progress=lambda: None)
        == "noop"
    )
    assert q.count(IndexFilter.all()) == n

    # Delete the DB row (crash simulation): republishing recreates the same
    # deterministic publication, still active, with the same point set.
    state_db.execute("DELETE FROM publications")
    assert (
        publish_generation(state_db, base_config, q, rev_id=rev,
                           run_id=_run_id(state_db, rev), on_progress=lambda: None)
        == "published"
    )
    row = state_db.query_one("SELECT pub_id, state FROM publications")
    assert row is not None and row["pub_id"] == pub and row["state"] == "active"
    assert q.count(IndexFilter.all()) == n
    assert q.count(IndexFilter.all(FieldCond("active", "eq", True))) == n


def test_publish_generation_unknown_revision_raises(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    with pytest.raises(PublicationError, match="unknown revision"):
        publish_generation(
            state_db, base_config, q, rev_id="nope", run_id="nope",
            on_progress=lambda: None,
        )


class _DroppingQdrant(FakeQdrant):
    """Drops one point per upsert: simulates a lost write mid-publication."""

    def upsert(self, points: Sequence[QdrantPoint]) -> None:
        super().upsert(points[:-1])


def test_publish_generation_fails_verification_on_dropped_point(
    state_db: Database, base_config: Config
) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    path = make_pdf(base_config.paths.source_roots[0] / "book.pdf", [_BOOK])
    rev, _pub = ingest_and_publish(state_db, base_config, path, qdrant=q)
    state_db.execute("DELETE FROM publications")

    dropper = _DroppingQdrant(base_config.embedding.dimensions)
    with pytest.raises(PublicationError, match="verification failed"):
        publish_generation(
            state_db, base_config, dropper, rev_id=rev,
            run_id=_run_id(state_db, rev), on_progress=lambda: None,
        )
    # Verification happens before B1, so no publication row may exist.
    assert state_db.query_one("SELECT pub_id FROM publications") is None


# --- The publication lock ---------------------------------------------------------


def test_publish_lock_lifecycle(state_db: Database) -> None:
    assert acquire_publish_lock(state_db, "w1") is True
    assert acquire_publish_lock(state_db, "w2") is False
    release_publish_lock(state_db, "w1")
    assert acquire_publish_lock(state_db, "w2") is True


def test_publish_lock_ttl_makes_it_stealable(state_db: Database) -> None:
    base = 1000.0
    assert acquire_publish_lock(state_db, "w1", now=base) is True
    assert acquire_publish_lock(state_db, "w2", now=base + _PUBLISH_LOCK_TTL - 1) is False
    assert acquire_publish_lock(state_db, "w2", now=base + _PUBLISH_LOCK_TTL + 1) is True


def test_publish_lock_release_by_non_owner_is_noop(state_db: Database) -> None:
    base = 1000.0
    assert acquire_publish_lock(state_db, "w1", now=base) is True
    release_publish_lock(state_db, "w2", now=base + 100)
    assert acquire_publish_lock(state_db, "w2", now=base + 100) is False


# --- reconcile_publications ------------------------------------------------------


def _two_revision_pipeline(db: Database, cfg: Config, q: FakeQdrant) -> tuple[str, str, str, str]:
    v1 = cfg.paths.source_roots[0] / "book.pdf"
    make_pdf(v1, ["Version one talks about zebras."])
    rev1, pub1 = ingest_and_publish(db, cfg, v1, qdrant=q)
    make_pdf(v1, ["Version two talks about quokkas."])
    rev2, pub2 = ingest_and_publish(db, cfg, v1, qdrant=q)
    return rev1, pub1, rev2, pub2


def _rewind_staged(db: Database, pub_id: str) -> None:
    db.execute(
        "UPDATE publications SET state = 'staged', activated_at = NULL WHERE pub_id = ?",
        (pub_id,),
    )


def test_reconcile_promotes_staged_with_complete_points_and_open_job(
    state_db: Database, base_config: Config
) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    _rev1, _pub1, rev2, pub2 = _two_revision_pipeline(state_db, base_config, q)

    _rewind_staged(state_db, pub2)
    state_db.execute(
        "INSERT INTO jobs (task_key, stage, input_id, state, attempts, max_attempts, "
        "created_at, updated_at) VALUES ('publish:manual', 'publish', ?, 'pending', 0, 3, 0.0, 0.0)",
        (rev2,),
    )
    assert reconcile_publications(state_db, base_config, q) == 1
    row = state_db.query_one("SELECT state FROM publications WHERE pub_id = ?", (pub2,))
    assert row is not None
    assert row["state"] == "active"


def test_reconcile_deletes_orphaned_staged_publication(
    state_db: Database, base_config: Config
) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    _rev1, _pub1, _rev2, pub2 = _two_revision_pipeline(state_db, base_config, q)

    _rewind_staged(state_db, pub2)
    q.set_active(IndexFilter.all(FieldCond("pub_id", "eq", pub2)), False)
    assert reconcile_publications(state_db, base_config, q) == 0
    assert state_db.query_one(
        "SELECT pub_id FROM publications WHERE pub_id = ?", (pub2,)
    ) is None


def test_reconcile_leaves_incomplete_publication_alone(
    state_db: Database, base_config: Config
) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    _rev1, _pub1, rev2, pub2 = _two_revision_pipeline(state_db, base_config, q)
    first_chunk = state_db.query_one(
        "SELECT chunk_id FROM chunks WHERE rev_id = ? ORDER BY position LIMIT 1", (rev2,)
    )
    assert first_chunk is not None
    q.delete(IndexFilter.all(
        FieldCond("pub_id", "eq", pub2),
        FieldCond("chunk_id", "eq", first_chunk["chunk_id"]),
    ))
    _rewind_staged(state_db, pub2)
    assert reconcile_publications(state_db, base_config, q) == 0
    row = state_db.query_one("SELECT state FROM publications WHERE pub_id = ?", (pub2,))
    assert row is not None
    assert row["state"] == "staged"


def test_collection_name_is_stable() -> None:
    assert COLLECTION == "library_chunks"


def test_real_qdrant_local_mode(base_config: Config) -> None:
    # Embedded (local) mode: qdrant_path set, no server process. This is the
    # pilot sandbox backend; the client API is backend-agnostic.
    base_config.services.qdrant_path = str(base_config.paths.qdrant_root)
    ops = RealQdrantOps(base_config)
    assert ops.ping() is True
    assert ops.collection_exists() is False
    ops.ensure_collection(dimensions=8)
    assert ops.collection_exists() is True
    # Re-ensure with matching dimensions is a no-op (idempotent startup).
    ops.ensure_collection(dimensions=8)
    # Local mode allows exactly one open client per storage path (exclusive
    # flock), so close before reopening for the mismatch check below.
    ops.close()
    # A dimension mismatch is an explicit operator error, not a silent
    # recreate: the embedding config changed, so point at fresh roots.
    with pytest.raises(IndexingError, match="dense dimensions"):
        RealQdrantOps(base_config).ensure_collection(dimensions=16)
