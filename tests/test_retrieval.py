"""M4 retrieval: fused dense+sparse search over the SQLite-visible
publications, SQLite postvalidation, degraded sparse-only mode, and the
selection contract (PRD §8F, §9, §12 gate "sparse degraded mode works")."""

from __future__ import annotations

import pytest

from fixtures import publish_handbuilt
from library_rag.config import Config
from library_rag.db import Database
from library_rag.embeddings import (
    FakeEmbedder,
    ModelUnavailableError,
    bm25_weights,
    load_sparse_stats,
)
from library_rag.indexing import FakeQdrant, QdrantPoint, visible_pub_ids
from library_rag.retrieval import (
    IndexUnavailableError,
    _Cand,
    _validate,
    search,
)

_TEXTS_A = [
    "Zebra stripe alpha one.",
    "Zebra mane alpha two.",
    "Zebra hooves alpha three.",
    "Zebra herd alpha four.",
    "Zebra plains alpha five.",
    "Zebra dust alpha six.",
]
_TEXTS_B = [
    "Quokka burrow beta one.",
    "Quokka tail beta two.",
    "Quokka pouch beta three.",
    "Quokka meadow beta four.",
    "Quokka cliff beta five.",
    "Quokka mist beta six.",
]


# The library fixture hands back everything a search test needs.
Library = tuple[
    Database, Config, FakeQdrant, FakeEmbedder,
    str, str, list[str],  # A: (state_db, cfg, qdrant, embedder), gen_a, pub_a, chunks_a
    str, str, list[str],  # B: gen_b, pub_b, chunks_b
]


class _DownQdrant(FakeQdrant):
    def ping(self) -> bool:
        return False


class _BrokenQueryEmbedder(FakeEmbedder):
    """Query inference is down (or the model is down mid-session)."""

    def encode_query(self, text: str) -> list[float]:
        raise ModelUnavailableError("simulated outage")


@pytest.fixture()
def library(state_db: Database, base_config: Config) -> Library:
    """Two published books in two statistics epochs (A then A+B)."""
    base_config.embedding.fake = True
    dims = base_config.embedding.dimensions
    q = FakeQdrant(dims)
    emb = FakeEmbedder(dims)
    gen_a, pub_a, chunks_a = publish_handbuilt(
        state_db, base_config, q,
        doc_id="docA", rev_id="revA", run_id="runA",
        texts=_TEXTS_A, title="Book A",
    )
    gen_b, pub_b, chunks_b = publish_handbuilt(
        state_db, base_config, q,
        doc_id="docB", rev_id="revB", run_id="runB",
        texts=_TEXTS_B, title="Book B",
    )
    return state_db, base_config, q, emb, gen_a, pub_a, chunks_a, gen_b, pub_b, chunks_b


# --- Availability guards -------------------------------------------------------


def test_qdrant_down_is_an_explicit_error(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = _DownQdrant(base_config.embedding.dimensions)
    with pytest.raises(IndexUnavailableError, match="unreachable"):
        search(state_db, base_config, q, FakeEmbedder(base_config.embedding.dimensions), "zebra")


def test_blank_query_rejected_before_any_io(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = _DownQdrant(base_config.embedding.dimensions)  # would raise if reached
    with pytest.raises(ValueError, match="non-empty"):
        search(state_db, base_config, q, None, "   ")


def test_no_publications_yields_an_empty_result(
    state_db: Database, base_config: Config
) -> None:
    base_config.embedding.fake = True
    dims = base_config.embedding.dimensions
    q = FakeQdrant(dims)
    hits = search(state_db, base_config, q, FakeEmbedder(dims), "zebra")
    assert hits.passages == ()
    assert all(v == 0 for v in hits.counts.values())


# --- Degraded sparse-only mode (PRD §9, M4 gate) ---------------------------------


def test_degraded_without_embedder_is_sparse_only(library: Library) -> None:
    state_db, base_config, q, _emb, _gen_a, pub_a, chunks_a, *_ = library
    hits = search(state_db, base_config, q, None, "zebra")
    assert hits.degraded is True
    assert hits.degraded_reason == "embedding model not configured; sparse-only search"
    assert len(hits.passages) >= 1
    assert all(p.pub_id == pub_a for p in hits.passages)  # only book A has "zebra"
    assert all(p.chunk_id in chunks_a for p in hits.passages)
    assert all(p.dense_rank is None for p in hits.passages)
    assert all(p.sparse_rank is not None for p in hits.passages)


def test_degraded_when_query_inference_is_down(library: Library) -> None:
    state_db, base_config, q, _emb, *_rest = library
    broken = _BrokenQueryEmbedder(base_config.embedding.dimensions)
    hits = search(state_db, base_config, q, broken, "zebra")
    assert hits.degraded is True
    assert hits.degraded_reason is not None
    assert "embedding inference unavailable" in hits.degraded_reason
    assert len(hits.passages) >= 1
    assert all(p.dense_rank is None for p in hits.passages)


# --- Fused search over both books -------------------------------------------------


def test_fused_search_diversifies_across_books(library: Library) -> None:
    state_db, base_config, q, emb, _gen_a, _pub_a, chunks_a, _gen_b, _pub_b, chunks_b = library
    hits = search(state_db, base_config, q, emb, "zebra quokka")

    assert len(hits.passages) == 8  # max_per_book=4 -> 4 + 4
    per_book: dict[str, int] = {}
    for p in hits.passages:
        per_book[p.doc_id] = per_book.get(p.doc_id, 0) + 1
    assert per_book == {"docA": 4, "docB": 4}
    assert hits.counts["fused"] == 12  # 6 chunks per book
    assert hits.counts["validated"] == 12
    assert hits.counts["postvalidation_removed"] == 0
    assert hits.counts["selected"] == 8
    assert hits.counts["topup_rounds"] == 0

    # Identity fields are materialized from SQLite, not the point payload.
    for p in hits.passages:
        assert p.spans == ()  # hand-built chunks carry empty spans
        if p.doc_id == "docA":
            assert p.title == "Book A" and p.chunk_id in chunks_a
        else:
            assert p.title == "Book B" and p.chunk_id in chunks_b


# --- SQLite postvalidation (PRD §8F: the payload is a cache) -----------------------


def test_validate_drops_uncommitted_and_stale_candidates(library: Library) -> None:
    state_db, base_config, q, _emb, gen_a, pub_a, chunks_a, *_ = library

    # A third book whose source revision is no longer active: its chunks must
    # never become evidence even though its publication is active.
    _gen_c, pub_c, chunks_c = publish_handbuilt(
        state_db, base_config, q,
        doc_id="docC", rev_id="revC", run_id="runC",
        texts=["Lonely chapter one."], pub_state="active", rev_active=0,
    )
    visible = visible_pub_ids(state_db)
    assert pub_c in visible

    def _cand(chunk_id: str, payload: dict[str, object]) -> _Cand:
        return _Cand(chunk_id, 1.0, None, None, dict(payload))

    base_payload = {"pub_id": pub_a, "rev_id": "revA", "doc_id": "docA", "gen_id": gen_a}
    c_ok = _cand(chunks_a[0], dict(base_payload))
    c_unknown = _cand("nope:chunk-0", dict(base_payload))
    c_rev_mismatch = _cand(chunks_a[0], {**base_payload, "rev_id": "revB"})
    c_invisible_pub = _cand(chunks_a[0], {**base_payload, "pub_id": "pub-ghost"})
    c_inactive_rev = _cand(chunks_c[0], {
        "pub_id": pub_c, "rev_id": "revC", "doc_id": "docC", "gen_id": "genC",
    })

    out = _validate(
        state_db, [c_ok, c_unknown, c_rev_mismatch, c_invisible_pub, c_inactive_rev], visible
    )
    assert [p.chunk_id for p in out] == [chunks_a[0]]
    # The survivor is materialized from the chunk row.
    assert out[0].doc_id == "docA" and out[0].rev_id == "revA"
    assert out[0].text == _TEXTS_A[0]


def test_ghost_point_is_dropped_end_to_end(library: Library) -> None:
    state_db, base_config, q, emb, gen_a, pub_a, _chunks_a, *_ = library
    # A point Qdrant somehow still serves, but whose chunk row no longer
    # exists in SQLite: it must be fetched, ranked, and dropped.
    stats_row = state_db.query_one(
        "SELECT sparse_stats_sha FROM index_generations WHERE gen_id = ?", (gen_a,)
    )
    assert stats_row is not None
    stats = load_sparse_stats(state_db, str(stats_row["sparse_stats_sha"]))
    assert stats is not None
    idx, val = bm25_weights("zebra", stats, k1=1.5, b=0.75)
    assert idx  # "zebra" is in this epoch's vocabulary
    q.upsert([
        QdrantPoint(
            point_id="ghost-point",
            payload={
                "chunk_id": "c-ghost", "pub_id": pub_a, "rev_id": "revA",
                "doc_id": "docA", "gen_id": gen_a, "active": True,
            },
            dense=(0.0,) * base_config.embedding.dimensions,
            sparse=(tuple(idx), tuple(val)),
        )
    ])

    hits = search(state_db, base_config, q, emb, "zebra")
    assert hits.counts["postvalidation_removed"] >= 1
    assert all(p.chunk_id != "c-ghost" for p in hits.passages)


# --- Book filters disable diversification ----------------------------------------


def test_doc_and_rev_filters_restrict_and_disable_diversity(library: Library) -> None:
    state_db, base_config, q, emb, *_rest = library

    hits = search(state_db, base_config, q, emb, "zebra quokka", doc_id="docA")
    assert len(hits.passages) == 6  # the whole book, cap disabled
    assert all(p.doc_id == "docA" for p in hits.passages)

    hits = search(state_db, base_config, q, emb, "zebra quokka", rev_id="revB")
    assert len(hits.passages) == 6
    assert all(p.rev_id == "revB" for p in hits.passages)


# --- The counts contract -----------------------------------------------------------


def test_counts_contract(library: Library) -> None:
    state_db, base_config, q, emb, *_rest = library
    hits = search(state_db, base_config, q, emb, "zebra quokka")
    assert set(hits.counts) == {
        "dense", "sparse", "fused", "validated", "postvalidation_removed",
        "reranked", "selected", "topup_rounds",
    }
    assert hits.counts["selected"] == len(hits.passages)
    assert hits.query == "zebra quokka"
