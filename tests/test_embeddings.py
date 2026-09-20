"""M4 embeddings: fakes, the encoding key, checkpoints, BM25, and the HTTP client."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from library_rag.config import Config, ConfigError
from library_rag.db import Database
from library_rag.embeddings import (
    CheckpointCorruptError,
    EmbeddingError,
    EmbeddingOOMError,
    FakeEmbedder,
    LlamaCppEmbedder,
    ModelUnavailableError,
    SparseStats,
    bm25_weights,
    checkpoint_manifest_path,
    compute_sparse_stats,
    embedding_sha,
    encode_batch_oom,
    load_sparse_stats,
    make_embedder,
    read_checkpoint,
    stats_sha,
    term_id,
    write_checkpoint,
)

# --- Fake embedder -------------------------------------------------------------


def test_fake_embedder_is_deterministic_and_normalized() -> None:
    emb = FakeEmbedder(dimensions=32)
    assert emb.model_revision == "fake-v1"
    assert emb.dimensions == 32
    v1 = emb.encode_documents(["the quick brown fox"])[0]
    v2 = emb.encode_documents(["the quick brown fox"])[0]
    assert v1 == v2
    assert len(v1) == 32
    assert math.sqrt(sum(x * x for x in v1)) == pytest.approx(1.0, rel=1e-9)
    v3 = emb.encode_documents(["a completely different sentence"])[0]
    assert v1 != v3


def test_fake_embedder_separates_document_and_query_roles() -> None:
    emb = FakeEmbedder(dimensions=64)
    doc = emb.encode_documents(["hello world"])[0]
    query = emb.encode_query("hello world")
    assert doc != query


# --- The encoding key (PRD §6) ----------------------------------------------------


def test_embedding_sha_invariant_to_batch_policy(base_config: Config) -> None:
    base_config.embedding.fake = True
    base = embedding_sha(base_config)
    base_config.embedding.batch_size = 1
    base_config.embedding.oom_max_halvings = 9
    assert embedding_sha(base_config) == base


def test_embedding_sha_sensitive_to_value_settings(base_config: Config) -> None:
    base_config.embedding.fake = True
    base = embedding_sha(base_config)
    base_config.embedding.dimensions = 512
    assert embedding_sha(base_config) != base
    base_config.embedding.dimensions = 1024
    base_config.embedding.normalize = False
    assert embedding_sha(base_config) != base
    base_config.embedding.normalize = True


def test_embedding_sha_unconfigured_raises(base_config: Config) -> None:
    with pytest.raises(ConfigError):
        embedding_sha(base_config)


def test_embedding_sha_fake_and_real_keys_differ(base_config: Config) -> None:
    base_config.embedding.fake = True
    fake_sha = embedding_sha(base_config)
    base_config.embedding.fake = False
    base_config.embedding.model_revision = "real-model@1"
    assert embedding_sha(base_config) != fake_sha


# --- make_embedder --------------------------------------------------------------


def test_make_embedder_unconfigured_raises(base_config: Config) -> None:
    with pytest.raises(ConfigError):
        make_embedder(base_config)


def test_make_embedder_fake(base_config: Config) -> None:
    base_config.embedding.fake = True
    base_config.embedding.dimensions = 32
    emb = make_embedder(base_config)
    assert isinstance(emb, FakeEmbedder)
    assert emb.dimensions == 32


def test_make_embedder_real(base_config: Config) -> None:
    base_config.embedding.model_revision = "rev-1"
    base_config.embedding.model_name = "bge-m3"
    emb = make_embedder(base_config)
    assert isinstance(emb, LlamaCppEmbedder)
    assert emb.model_revision == "rev-1"


# --- Vector checkpoints (PRD §5) ----------------------------------------------------


def _vectors() -> list[list[float]]:
    # Every value is exactly representable in float32, so roundtrips compare equal.
    return [[0.5, 0.25, 0.0, 1.0], [0.125, -0.5, 0.75, 0.0], [0.0, 0.0, 0.25, -1.0]]


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    vectors = _vectors()
    path = tmp_path / "batch_00000.bin"
    vector_sha = write_checkpoint(path, vectors, 4, {"chunk_ids": ["a", "b", "c"]})
    assert vector_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (tmp_path / "batch_00000.json").is_file()
    assert read_checkpoint(path, ["a", "b", "c"]) == vectors


@pytest.mark.parametrize(
    "mutate",
    [
        "missing_file",
        "missing_manifest",
        "too_short",
        "bad_magic",
        "bad_version",
        "bad_dtype",
        "size_mismatch",
        "hash_mismatch",
        "chunk_id_mismatch",
        "manifest_shape_mismatch",
        "unreadable_manifest",
    ],
)
def test_checkpoint_corruption_detected(tmp_path: Path, mutate: str) -> None:
    vectors = [[0.5, 0.25, 0.0, 1.0], [0.125, -0.5, 0.75, 0.0]]
    path = tmp_path / "batch_00000.bin"
    write_checkpoint(path, vectors, 4, {"chunk_ids": ["c0", "c1"]})
    manifest_path = checkpoint_manifest_path(path)
    if mutate == "missing_file":
        path.unlink()
    elif mutate == "missing_manifest":
        manifest_path.unlink()
    elif mutate == "too_short":
        path.write_bytes(path.read_bytes()[:10])
    elif mutate == "bad_magic":
        raw = bytearray(path.read_bytes())
        raw[0:4] = b"XXXX"
        path.write_bytes(bytes(raw))
    elif mutate == "bad_version":
        raw = bytearray(path.read_bytes())
        struct.pack_into("<I", raw, 4, 2)
        path.write_bytes(bytes(raw))
    elif mutate == "bad_dtype":
        raw = bytearray(path.read_bytes())
        raw[12] = 1  # header: magic(4) + version(4) + dims(4) -> dtype at offset 12
        path.write_bytes(bytes(raw))
    elif mutate == "size_mismatch":
        path.write_bytes(path.read_bytes() + b"\x00")
    elif mutate == "hash_mismatch":
        raw = bytearray(path.read_bytes())
        raw[21] ^= 0xFF  # first vector byte, after the 21-byte header
        path.write_bytes(bytes(raw))
    elif mutate == "chunk_id_mismatch":
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        m["chunk_ids"] = ["x", "y"]
        manifest_path.write_text(json.dumps(m), encoding="utf-8")
    elif mutate == "manifest_shape_mismatch":
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        m["count"] = 99
        manifest_path.write_text(json.dumps(m), encoding="utf-8")
    else:  # unreadable_manifest
        manifest_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointCorruptError):
        read_checkpoint(path, ["c0", "c1"])


# --- Client-side BM25 -------------------------------------------------------------


def test_bm25_weights_reference_values() -> None:
    stats = SparseStats(stats_sha="s", doc_count=2, avg_doc_len=2.0, df={"fox": 1, "dog": 1})
    indices, values = bm25_weights("fox fox dog", stats, k1=1.5, b=0.75)
    # Terms are emitted in sorted term-name order, so "dog" precedes "fox".
    assert indices == [term_id("dog"), term_id("fox")]
    idf = math.log(1.0 + (2 - 1 + 0.5) / (1 + 0.5))
    norm = 1 - 0.75 + 0.75 * 3 / 2.0  # doc_len 3, avg 2
    exp_dog = idf * (1 * 2.5) / (1 + 1.5 * norm)
    exp_fox = idf * (2 * 2.5) / (2 + 1.5 * norm)
    assert values == pytest.approx([exp_dog, exp_fox])
    assert values[1] > values[0]


def test_bm25_weights_edge_cases() -> None:
    stats = SparseStats(stats_sha="s", doc_count=1, avg_doc_len=1.0, df={"alpha": 1})
    assert bm25_weights("omega zeta", stats) == ([], [])  # unknown terms
    assert bm25_weights("", stats) == ([], [])
    empty = SparseStats(stats_sha="s", doc_count=0, avg_doc_len=0.0, df={})
    assert bm25_weights("alpha", empty) == ([], [])

    # avg_len <= 0 falls back to 1.0.
    zero_avg = SparseStats(stats_sha="s", doc_count=1, avg_doc_len=0.0, df={"alpha": 1})
    indices, values = bm25_weights("alpha", zero_avg)
    assert indices == [term_id("alpha")]
    idf = math.log(1.0 + (1 - 1 + 0.5) / (1 + 0.5))
    # doc_len == avg_len == 1.0 makes the norm factor (1 - b + b) == 1, so the
    # denominator is count + k1 and the weight collapses to idf * 2.5 / 2.5.
    assert values == pytest.approx([idf])


def test_stats_sha_order_insensitive_and_parameter_sensitive() -> None:
    base = stats_sha(2, 2.0, {"a": 1, "b": 2}, 1.5, 0.75)
    assert stats_sha(2, 2.0, {"b": 2, "a": 1}, 1.5, 0.75) == base
    assert stats_sha(3, 2.0, {"a": 1, "b": 2}, 1.5, 0.75) != base
    assert stats_sha(2, 2.5, {"a": 1, "b": 2}, 1.5, 0.75) != base
    assert stats_sha(2, 2.0, {"a": 1, "b": 3}, 1.5, 0.75) != base
    assert stats_sha(2, 2.0, {"a": 1, "b": 2}, 2.0, 0.75) != base
    assert stats_sha(2, 2.0, {"a": 1, "b": 2}, 1.5, 0.5) != base


# --- Corpus statistics epochs -------------------------------------------------------


def _add_revision(
    db: Database,
    doc_id: str,
    rev_id: str,
    run_id: str,
    texts: list[str],
    *,
    pub_state: str | None = "active",
) -> None:
    """Minimal catalog rows for stats coverage (publications optional)."""
    db.execute(
        "INSERT INTO documents (doc_id, anchor_sha256, created_at, updated_at) VALUES (?, ?, 0, 0)",
        (doc_id, "a" * 64),
    )
    db.execute(
        "INSERT INTO source_revisions (rev_id, doc_id, sha256, size_bytes, format, "
        "archive_relpath, first_path, is_active, created_at) VALUES (?, ?, ?, 1, 'pdf', 'x', 'x', 1, 0)",
        (rev_id, doc_id, "b" * 64),
    )
    db.execute(
        "INSERT INTO extraction_runs (run_id, rev_id, doc_id, parser_version, settings_sha, "
        "state, created_at, updated_at) VALUES (?, ?, ?, 'p', 's', 'succeeded', 0, 0)",
        (run_id, rev_id, doc_id),
    )
    for i, text in enumerate(texts):
        db.execute(
            "INSERT INTO chunks (chunk_id, run_id, rev_id, position, text, token_count, "
            "spans, created_at) VALUES (?, ?, ?, ?, ?, 1, '[]', 0)",
            (f"{run_id}:c{i}", run_id, rev_id, i, text),
        )
    if pub_state is not None:
        db.execute(
            "INSERT INTO index_generations (gen_id, run_id, rev_id, model_revision, "
            "embedding_sha, sparse_stats_sha, dimensions, dtype, normalized, state, "
            "created_at, updated_at) VALUES (?, ?, ?, 'm', 'e', 's', 4, 'float32', 1, "
            "'ready', 0, 0)",
            (f"gen-{run_id}", run_id, rev_id),
        )
        db.execute(
            "INSERT INTO publications (pub_id, rev_id, doc_id, gen_id, run_id, "
            "expected_points, state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (f"pub-{rev_id}", rev_id, doc_id, f"gen-{run_id}", run_id, len(texts), pub_state),
        )


def test_compute_sparse_stats_covers_active_publications_only(
    state_db: Database, base_config: Config
) -> None:
    _add_revision(state_db, "d1", "r1", "run1", ["alpha beta gamma", "delta epsilon zeta"])
    _add_revision(state_db, "d2", "r2", "run2", ["omega"], pub_state="staged")
    stats = compute_sparse_stats(state_db, base_config, now=0.0)
    assert stats.doc_count == 2
    assert set(stats.df) == {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"}
    assert "omega" not in stats.df  # the staged revision must not leak in
    assert stats.avg_doc_len == pytest.approx(3.0)
    bm25 = base_config.retrieval.bm25
    assert stats.stats_sha == stats_sha(2, 3.0, stats.df, bm25.k1, bm25.b)


def test_compute_sparse_stats_include_rev_id(state_db: Database, base_config: Config) -> None:
    _add_revision(state_db, "d1", "r1", "run1", ["alpha beta gamma", "delta epsilon zeta"])
    _add_revision(state_db, "d2", "r2", "run2", ["omega"], pub_state=None)
    stats = compute_sparse_stats(state_db, base_config, include_rev_id="r2", now=0.0)
    assert stats.doc_count == 3
    assert "omega" in stats.df
    row = state_db.query_one(
        "SELECT * FROM sparse_corpus_stats WHERE stats_sha = ?", (stats.stats_sha,)
    )
    assert row is not None


def test_compute_sparse_stats_idempotent(state_db: Database, base_config: Config) -> None:
    _add_revision(state_db, "d1", "r1", "run1", ["alpha"])
    s1 = compute_sparse_stats(state_db, base_config, now=0.0)
    s2 = compute_sparse_stats(state_db, base_config, now=1.0)
    assert s1.stats_sha == s2.stats_sha
    row = state_db.query_one("SELECT COUNT(*) AS n FROM sparse_corpus_stats")
    assert row is not None
    assert int(row["n"]) == 1


def test_load_sparse_stats_roundtrip_and_missing(state_db: Database, base_config: Config) -> None:
    _add_revision(state_db, "d1", "r1", "run1", ["alpha beta"])
    stats = compute_sparse_stats(state_db, base_config, now=0.0)
    loaded = load_sparse_stats(state_db, stats.stats_sha)
    assert loaded is not None
    assert loaded.doc_count == stats.doc_count
    assert loaded.avg_doc_len == pytest.approx(stats.avg_doc_len)
    assert loaded.df == stats.df
    assert load_sparse_stats(state_db, "no-such-sha") is None


# --- OOM batch halving (PRD §8F) ------------------------------------------------------


class _HalvingEmbedder:
    """Raises OOM for chunks of size >= *oom_at*; records the sizes it saw."""

    def __init__(self, oom_at: int) -> None:
        self.oom_at = oom_at
        self.calls: list[int] = []

    @property
    def model_revision(self) -> str:
        return "fake-v1"

    @property
    def dimensions(self) -> int:
        return 2

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        if len(texts) >= self.oom_at:
            raise EmbeddingOOMError("simulated OOM")
        return [[1.0, 0.0] for _ in texts]

    def encode_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_encode_batch_oom_halves_and_recovers() -> None:
    emb = _HalvingEmbedder(oom_at=3)  # chunks of 3+ OOM; 2 fit
    out = encode_batch_oom(emb, ["a", "b", "c", "d"], start_size=4, max_halvings=2)
    assert len(out) == 4
    assert emb.calls == [4, 2, 2]


def test_encode_batch_oom_exhausts_budget() -> None:
    emb = _HalvingEmbedder(oom_at=1)  # even a single text OOMs
    with pytest.raises(EmbeddingOOMError):
        encode_batch_oom(emb, ["a", "b"], start_size=4, max_halvings=2)
    assert emb.calls == [2, 2, 1]


def test_encode_batch_oom_zero_halvings_reraises() -> None:
    emb = _HalvingEmbedder(oom_at=1)
    with pytest.raises(EmbeddingOOMError):
        encode_batch_oom(emb, ["a", "b"], start_size=4, max_halvings=0)
    assert emb.calls == [2]


# --- The llama.cpp HTTP client -----------------------------------------------------------


def _llama(handler: object) -> LlamaCppEmbedder:
    emb = LlamaCppEmbedder(
        "127.0.0.1",
        1,
        model_revision="r1",
        model_name="m",
        dimensions=4,
        normalize=True,
        timeout=5,
    )
    emb._http = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return emb


def test_llamacpp_orders_by_index_and_normalizes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 0.0, 0.0, 2.0]},
                    {"index": 0, "embedding": [2.0, 0.0, 0.0, 0.0]},
                ]
            },
        )

    out = _llama(handler).encode_documents(["t0", "t1"])
    assert out[0] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert out[1] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_llamacpp_dimension_mismatch_is_permanent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    with pytest.raises(EmbeddingError, match="dimensions"):
        _llama(handler).encode_documents(["t"])


def test_llamacpp_invalid_json_is_permanent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    with pytest.raises(EmbeddingError, match="invalid JSON"):
        _llama(handler).encode_documents(["t"])


def test_llamacpp_missing_data_is_permanent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    with pytest.raises(EmbeddingError, match="no data"):
        _llama(handler).encode_documents(["t"])


def test_llamacpp_oom_report_halving_signal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(507, text="out of memory: cuda")

    with pytest.raises(EmbeddingOOMError):
        _llama(handler).encode_documents(["t"])


def test_llamacpp_input_too_large_is_permanent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "error": {
                    "code": 500,
                    "message": (
                        "input (572 tokens) is too large to process. "
                        "increase the physical batch size (current batch size: 512)"
                    ),
                    "type": "server_error",
                }
            },
        )

    with pytest.raises(EmbeddingError, match="too large"):
        _llama(handler).encode_documents(["t"])


def test_llamacpp_server_error_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    with pytest.raises(ModelUnavailableError, match="500"):
        _llama(handler).encode_documents(["t"])


def test_llamacpp_rejected_request_is_permanent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    with pytest.raises(EmbeddingError, match="rejected"):
        _llama(handler).encode_documents(["t"])


def test_llamacpp_unreachable_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ModelUnavailableError, match="unreachable"):
        _llama(handler).encode_documents(["t"])


def test_llamacpp_encode_query_posts_single_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {"model": "m", "input": ["q"]}
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [2.0, 0.0, 0.0, 0.0]}]}
        )

    out = _llama(handler).encode_query("q")
    assert out == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_llamacpp_all_zero_vector_is_permanent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 4}]})

    with pytest.raises(EmbeddingError, match="all-zero"):
        _llama(handler).encode_documents(["t"])
