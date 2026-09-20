"""Embeddings: dense model interface, BM25 sparse side, and vector checkpoints.

Two encoding sides share this module (PRD §6/§8F):

* **Dense** — a pluggable embedder behind the service config: the llama.cpp
  adapter in production, and a *deterministic fake* for tests (clearly labeled
  ``fake-v1``, forbidden in production config by default). Encoded vectors are
  checkpointed to a non-pickle numeric artifact **before** any Qdrant write, so
  an index rebuild re-reads vectors instead of re-embedding (PRD §5/§8F).
* **Sparse** — client-side BM25. The pinned Qdrant release has no server-side
  BM25 index (only fixed sparse-index RAM/mmap), so document and query sparse
  vectors are computed here from corpus statistics persisted in
  ``sparse_corpus_stats``. Term IDs are derived from the term's SHA-256, so a
  query process and an indexing process agree on IDs without a shared
  vocabulary file, and the IDF uses the non-negative form
  ``ln(1 + (N - df + 0.5) / (df + 0.5))`` so no term scores below zero.

The **embedding key** (PRD §6) hashes the chunk key, the model revision, and
the value-affecting encoding settings (model name, dimensions, dtype,
normalization). Batch policy affects cost, not values, and therefore does not
enter the key: changing ``batch_size`` never invalidates checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

from .artifacts import commit_bytes
from .config import Config, ConfigError
from .db import Database

__all__ = [
    "CheckpointCorruptError",
    "Embedder",
    "EmbeddingError",
    "EmbeddingOOMError",
    "FakeEmbedder",
    "LlamaCppEmbedder",
    "ModelUnavailableError",
    "SparseStats",
    "bm25_weights",
    "checkpoint_path",
    "compute_sparse_stats",
    "embedding_sha",
    "encode_batch_oom",
    "load_sparse_stats",
    "make_embedder",
    "read_checkpoint",
    "stats_row_upsert",
    "term_id",
    "tokenize",
    "write_checkpoint",
]


# --- Errors ------------------------------------------------------------------

class EmbeddingError(RuntimeError):
    """Permanent model-side error (bad response shape, dimension mismatch...)."""


class ModelUnavailableError(RuntimeError):
    """The embedding server is unreachable or unhealthy.

    Transient: the worker retries with backoff, and the search path degrades to
    sparse-only with an explicit status (PRD §9).
    """


class EmbeddingOOMError(RuntimeError):
    """The model server reported out-of-memory for a batch (PRD §8F).

    The caller halves the batch (bounded); a single text that cannot fit is a
    permanent failure that stops the worker rather than looping forever.
    """


class CheckpointCorruptError(RuntimeError):
    """A checkpoint artifact failed validation (magic, size, manifest mismatch)."""


# --- Embedder interface -------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    """Explicit query/document encoding (PRD §8F).

    Implementations must be deterministic for the same text and settings; the
    dense-side identity of every vector derives from
    :func:`embedding_sha` + the model revision, not from this object.
    """

    @property
    def model_revision(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


def make_embedder(cfg: Config) -> Embedder:
    """Build the embedder from the service config (PRD: model wiring behind config).

    Raises :class:`ConfigError` when no model is configured: scan/extract/chunk
    stages are allowed to run before the model exists, but embedding and search
    are not.
    """
    emb = cfg.embedding
    if not emb.is_configured:
        raise ConfigError(
            "embedding is not configured: set embedding.model_revision "
            "(or embedding.fake: true in tests only)"
        )
    if emb.fake:
        return FakeEmbedder(dimensions=emb.dimensions)
    return LlamaCppEmbedder(
        host=cfg.services.embed_host,
        port=cfg.services.embed_port,
        model_revision=emb.effective_revision,
        model_name=emb.model_name or emb.effective_revision,
        dimensions=emb.dimensions,
        normalize=emb.normalize,
    )


class FakeEmbedder:
    """Deterministic, non-semantic embedder for tests (PRD §12 M4 scope).

    Clearly labeled: its model revision is the literal ``fake-v1``, so every
    checkpoint, generation, and point it touches is identifiable as non-real.
    ``EmbeddingSettings.fake`` defaults to False, so production configs never
    reach this class unless explicitly opted in.

    Vectors are L2-normalized digests of the text (doc and query use distinct
    seeds, mirroring "explicit query/document encoding"). Identical texts map
    to identical vectors and different texts to different ones — enough to
    exercise retrieval, scoring, and checkpoints without a model.
    """

    def __init__(self, dimensions: int = 1024) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def model_revision(self) -> str:
        return "fake-v1"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t, role="doc") for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vector(text, role="query")

    def _vector(self, text: str, *, role: str) -> list[float]:
        seed = hashlib.sha256(f"{role}\x00{text}".encode()).digest()
        values: list[float] = []
        while len(values) < self._dimensions:
            seed = hashlib.sha256(seed).digest()
            # 32 bytes -> 16 uint16 -> 16 floats in [0, 1): no inf/nan, stable.
            values.extend(b / 65536.0 for b in struct.unpack("<16H", seed))
        values = values[: self._dimensions]
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0:  # all-zero digest is impossible for sha256; defensive.
            return values
        return [v / norm for v in values]


class LlamaCppEmbedder:
    """OpenAI-compatible ``/v1/embeddings`` client for a local llama.cpp server.

    The server is started by the operator (pinned llama.cpp + GGUF weights
    under the model root); this client only talks HTTP. Unreachable/5xx ->
    :class:`ModelUnavailableError` (transient); an OOM report ->
    :class:`EmbeddingOOMError` (caller halves the batch); malformed or
    dimension-mismatched responses -> :class:`EmbeddingError` (permanent).
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        model_revision: str,
        model_name: str,
        dimensions: int,
        normalize: bool = True,
        timeout: float = 120.0,
    ) -> None:
        self._url = f"http://{host}:{port}/v1/embeddings"
        self._model_name = model_name
        self._model_revision = model_revision
        self._dimensions = dimensions
        self._normalize = normalize
        self._http = httpx.Client(timeout=timeout)

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._post(list(texts))
        if len(data) != len(texts):
            raise EmbeddingError(
                f"model returned {len(data)} vectors for {len(texts)} inputs"
            )
        vectors: list[tuple[int, list[float]]] = []
        for d in data:
            index = d.get("index")
            if not isinstance(index, int):
                raise EmbeddingError(f"model returned a non-integer index: {index!r}")
            embedding = d.get("embedding")
            if not isinstance(embedding, (list, tuple)):
                raise EmbeddingError("model returned a vector that is not a list")
            try:
                vectors.append((index, [float(v) for v in embedding]))
            except (TypeError, ValueError) as exc:
                raise EmbeddingError(f"model returned a non-numeric vector: {exc}") from exc
        vectors.sort(key=lambda pair: pair[0])
        return [self._finish(vec) for _index, vec in vectors]

    def encode_query(self, text: str) -> list[float]:
        # llama.cpp's /v1/embeddings has no query/document role parameter, so
        # both roles post the raw text; the explicit-role contract is honored
        # by the fake and by any future adapter that supports it.
        return self.encode_documents([text])[0]

    def _post(self, texts: list[str]) -> list[dict[str, object]]:
        try:
            resp = self._http.post(self._url, json={"model": self._model_name, "input": texts})
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(f"embedding server unreachable: {exc}") from exc
        body = resp.text
        if resp.status_code == 200:
            try:
                payload = resp.json()
            except ValueError as exc:
                raise EmbeddingError("embedding server returned invalid JSON") from exc
            data = payload.get("data")
            if not isinstance(data, list) or not data:
                raise EmbeddingError("embedding server response has no data[]")
            return data
        lowered = body.lower()
        if resp.status_code >= 500 and ("out of memory" in lowered or "oom" in lowered):
            raise EmbeddingOOMError(f"embedding server OOM: {body[:200]}")
        if "too large to process" in lowered:
            # llama.cpp rejects inputs longer than the physical batch (a 500
            # "input (N tokens) is too large to process"). No retry will ever
            # fit such an input, so it is a permanent bad-input error: the job
            # fails explicitly (``embedding_error``, transient=False) instead
            # of zombie-retrying as a transient server error.
            raise EmbeddingError(f"model rejected an input that is too large: {body[:200]}")
        if resp.status_code >= 500:
            raise ModelUnavailableError(f"embedding server error {resp.status_code}: {body[:200]}")
        raise EmbeddingError(f"embedding server rejected request: {resp.status_code} {body[:200]}")

    def _finish(self, vector: list[float]) -> list[float]:
        if len(vector) != self._dimensions:
            raise EmbeddingError(
                f"model returned {len(vector)} dimensions, configured {self._dimensions}"
            )
        if self._normalize:
            norm = math.sqrt(sum(v * v for v in vector))
            if norm == 0.0:
                raise EmbeddingError("model returned an all-zero vector")
            vector = [v / norm for v in vector]
        return vector


def encode_batch_oom(
    embedder: Embedder,
    texts: Sequence[str],
    start_size: int,
    max_halvings: int,
) -> list[list[float]]:
    """Encode *texts*, halving the batch on OOM (bounded, PRD §8F).

    The halving budget (*max_halvings*) is consumed across the whole call. A
    single text that still cannot fit — or an exhausted budget — re-raises
    :class:`EmbeddingOOMError` so the worker records a failure and stops that
    worker rather than looping forever.
    """
    remaining = list(texts)
    size = max(1, start_size)
    halvings = 0
    out: list[list[float]] = []
    while remaining:
        chunk, remaining = remaining[:size], remaining[size:]
        try:
            out.extend(embedder.encode_documents(chunk))
        except EmbeddingOOMError:
            if size == 1 or halvings >= max_halvings:
                raise
            size = max(1, size // 2)
            halvings += 1
            remaining = chunk + remaining
    return out


# --- Embedding key ------------------------------------------------------------

def embedding_sha(cfg: Config) -> str:
    """SHA-256 of the value-affecting encoding configuration (PRD §6).

    Only settings that change vector values enter the key: the model revision
    and server model name, dimensions, dtype, and the normalization flag.
    Batch policy is deliberately excluded, so ``batch_size`` changes never
    invalidate checkpoints.
    """
    emb = cfg.embedding
    if not emb.is_configured:
        raise ConfigError("embedding is not configured; cannot compute the embedding key")
    payload = {
        "model_revision": emb.effective_revision,
        "model_name": emb.model_name or emb.effective_revision,
        "dimensions": emb.dimensions,
        "dtype": emb.dtype,
        "normalize": emb.normalize,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- Vector checkpoints (PRD §5: non-pickle numeric format + chunk manifest) --

_CKPT_MAGIC = b"LBEM"
_CKPT_VERSION = 1
_DTYPE_FLOAT32 = 0
_CKPT_HEADER = struct.Struct("<4sIIBQ")  # magic, version, dims, dtype, count


def checkpoint_path(artifact_root: Path, run_id: str, embedding_sha_value: str, batch_index: int) -> Path:
    """Artifact location for one dense-vector batch (binary + ``.json`` sidecar)."""
    directory = artifact_root / "embeddings" / run_id / embedding_sha_value
    return directory / f"batch_{batch_index:05d}.bin"


def write_checkpoint(
    path: Path,
    vectors: Sequence[Sequence[float]],
    dims: int,
    manifest: dict[str, object],
) -> str:
    """Atomically write the binary vectors plus a JSON manifest sidecar.

    Returns the SHA-256 of the binary bytes (recorded in ``embedding_batches``
    as ``vector_sha256``). The manifest carries the chunk IDs in row order, the
    embedding key, and the returned hash, so a resume can verify identity
    before reusing a file.
    """
    data = _CKPT_HEADER.pack(_CKPT_MAGIC, _CKPT_VERSION, dims, _DTYPE_FLOAT32, len(vectors))
    for vector in vectors:
        data += struct.pack(f"<{dims}f", *vector)
    commit_bytes(path, data)
    vector_sha = hashlib.sha256(data).hexdigest()
    manifest = dict(manifest)
    manifest.setdefault("vector_sha256", vector_sha)
    manifest.setdefault("dims", dims)
    manifest.setdefault("count", len(vectors))
    manifest_path = checkpoint_manifest_path(path)
    commit_bytes(manifest_path, json.dumps(manifest, sort_keys=True).encode("utf-8"))
    return vector_sha


def read_checkpoint(path: Path, expected_chunk_ids: Sequence[str]) -> list[list[float]]:
    """Read and validate a checkpoint; return its vectors in row order.

    Raises :class:`CheckpointCorruptError` on any mismatch (magic, version,
    size, manifest chunk IDs, recorded hash) — a replay must never silently
    reuse the wrong vectors.
    """
    manifest_path = checkpoint_manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        raise CheckpointCorruptError(f"checkpoint missing: {path}")
    raw = path.read_bytes()
    if len(raw) < _CKPT_HEADER.size:
        raise CheckpointCorruptError(f"checkpoint too short: {path}")
    magic, version, dims, dtype, count = _CKPT_HEADER.unpack_from(raw, 0)
    if magic != _CKPT_MAGIC:
        raise CheckpointCorruptError(f"bad checkpoint magic: {path}")
    if version != _CKPT_VERSION:
        raise CheckpointCorruptError(f"unsupported checkpoint version {version}: {path}")
    if dtype != _DTYPE_FLOAT32:
        raise CheckpointCorruptError(f"unsupported checkpoint dtype {dtype}: {path}")
    expected_size = _CKPT_HEADER.size + count * dims * 4
    if len(raw) != expected_size:
        raise CheckpointCorruptError(f"checkpoint size mismatch: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointCorruptError(f"unreadable checkpoint manifest: {manifest_path}") from exc
    if manifest.get("vector_sha256") != hashlib.sha256(raw).hexdigest():
        raise CheckpointCorruptError(f"checkpoint hash mismatch: {path}")
    if manifest.get("chunk_ids") != list(expected_chunk_ids):
        raise CheckpointCorruptError(
            f"checkpoint manifest chunk IDs do not match the run: {manifest_path}"
        )
    if int(manifest.get("dims", -1)) != dims or int(manifest.get("count", -1)) != count:
        raise CheckpointCorruptError(f"checkpoint manifest shape mismatch: {manifest_path}")
    body = raw[_CKPT_HEADER.size :]
    unpack = struct.Struct(f"<{dims}f")
    return [list(unpack.unpack_from(body, i * unpack.size)) for i in range(count)]


def checkpoint_manifest_path(path: Path) -> Path:
    if path.name.endswith(".bin"):
        return path.with_name(path.name[: -len(".bin")] + ".json")
    return path.with_suffix(".json")


# --- Client-side BM25 (PRD §8F/§9) ---------------------------------------------

_TOKEN_RE = re.compile(r"\S+")


def tokenize(text: str) -> list[str]:
    """BM25 tokens: maximal non-whitespace runs, lowercased.

    The same tokenization the "words" chunk tokenizer uses, so the sparse side
    and the chunk text stay mutually consistent; document and query encoding
    share this exact function so term IDs always line up.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def term_id(term: str) -> int:
    """Stable 32-bit ID for a term (low 4 bytes of its SHA-256, little-endian).

    Content-derived, so any process — indexer, querier, any machine — computes
    the same ID with no shared vocabulary file. Collisions in 32 bits are
    possible in principle; for a personal-library vocabulary they are not
    worth a bigger ID space (Qdrant sparse indices are uint32 anyway).
    """
    return int.from_bytes(hashlib.sha256(term.encode("utf-8")).digest()[:4], "little")


def _idf(doc_count: int, doc_freq: int) -> float:
    # Non-negative form (the "required IDF handling"): ln(1 + (N-df+0.5)/(df+0.5))
    # is always >= 0, so a term never contributes a negative score.
    return math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


def bm25_weights(
    text: str,
    stats: SparseStats,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[list[int], list[float]]:
    """Sparse vector for one text under *stats* (indices, values).

    Terms absent from the statistics epoch get zero weight (they do not occur
    in the corpus this epoch was computed from), which keeps a query vector
    honest when the corpus grew since the generation was built.
    """
    tokens = tokenize(text)
    if not tokens or stats.doc_count == 0:
        return [], []
    tf: dict[str, int] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    doc_len = len(tokens)
    avg_len = stats.avg_doc_len if stats.avg_doc_len > 0 else 1.0
    indices: list[int] = []
    values: list[float] = []
    for term in sorted(tf):
        doc_freq = stats.df.get(term)
        if doc_freq is None:
            continue
        count = tf[term]
        weight = _idf(stats.doc_count, doc_freq) * (
            count * (k1 + 1.0) / (count + k1 * (1.0 - b + b * doc_len / avg_len))
        )
        indices.append(term_id(term))
        values.append(weight)
    return indices, values


@dataclass(frozen=True)
class SparseStats:
    """Corpus-wide BM25 statistics for one epoch (PRD §8F)."""

    stats_sha: str
    doc_count: int
    avg_doc_len: float
    df: dict[str, int] = field(default_factory=dict)


def stats_sha(doc_count: int, avg_doc_len: float, df: dict[str, int], k1: float, b: float) -> str:
    """SHA-256 over the corpus statistics **and** the BM25 parameters.

    Changing k1/b changes every sparse vector, so it must move the statistics
    epoch (and therefore every generation) exactly like adding a document does.
    """
    canonical = json.dumps(
        {"N": doc_count, "avgdl": avg_doc_len, "df": dict(sorted(df.items())), "k1": k1, "b": b},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stats_row_upsert(db: Database, stats: SparseStats, now: float) -> None:
    """Persist one statistics epoch (idempotent on stats_sha)."""
    db.execute(
        """
        INSERT INTO sparse_corpus_stats (stats_sha, doc_count, avg_doc_len, df_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(stats_sha) DO UPDATE SET
            doc_count = excluded.doc_count,
            avg_doc_len = excluded.avg_doc_len,
            df_json = excluded.df_json
        """,
        (
            stats.stats_sha,
            stats.doc_count,
            stats.avg_doc_len,
            json.dumps(dict(sorted(stats.df.items())), separators=(",", ":")),
            now,
        ),
    )


def compute_sparse_stats(
    db: Database,
    cfg: Config,
    include_rev_id: str | None = None,
    now: float | None = None,
) -> SparseStats:
    """Corpus-wide BM25 statistics over the currently active publications.

    The corpus is exactly what search can return: the chunks whose revision has
    an **active** publication in SQLite (never Qdrant flags). *include_rev_id*
    adds the revision being staged for publication, so the generation that
    activates it is built against the corpus it will join — and the
    statistics row persisted afterwards matches the active set post-switch.
    """
    ts = time.time() if now is None else now
    bm25 = cfg.retrieval.bm25
    texts: list[str] = []
    rows = db.query(
        """
        SELECT c.text FROM chunks c
        JOIN publications p ON p.rev_id = c.rev_id AND p.state = 'active'
        """
    )
    active_revs = {r["rev_id"] for r in db.query("SELECT rev_id FROM publications WHERE state = 'active'")}
    for row in rows:
        texts.append(row["text"])
    if include_rev_id is not None and include_rev_id not in active_revs:
        for row in db.query("SELECT text FROM chunks WHERE rev_id = ?", (include_rev_id,)):
            texts.append(row["text"])

    df: dict[str, int] = {}
    total_len = 0
    for text in texts:
        tokens = tokenize(text)
        total_len += len(tokens)
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    doc_count = len(texts)
    avg_doc_len = (total_len / doc_count) if doc_count else 0.0
    sha = stats_sha(doc_count, avg_doc_len, df, bm25.k1, bm25.b)
    stats = SparseStats(stats_sha=sha, doc_count=doc_count, avg_doc_len=avg_doc_len, df=df)
    stats_row_upsert(db, stats, ts)
    return stats


def load_sparse_stats(db: Database, stats_sha_value: str) -> SparseStats | None:
    """Load one persisted statistics epoch (used by the query path)."""
    row = db.query_one("SELECT * FROM sparse_corpus_stats WHERE stats_sha = ?", (stats_sha_value,))
    if row is None:
        return None
    df = json.loads(row["df_json"])
    return SparseStats(
        stats_sha=row["stats_sha"],
        doc_count=int(row["doc_count"]),
        avg_doc_len=float(row["avg_doc_len"]),
        df={k: int(v) for k, v in df.items()},
    )


# ---------------------------------------------------------------------------
# Corpus-epoch record (M6 fix B1 — kills the O(N^2) republish recompute)
#
# The statistics epoch (stats_sha) includes the document count, so it moves on
# every real publish. Without a record, every publish — including the ~99.9%
# that are stale-epoch no-op republishes — pays a full O(corpus) tokenization
# to discover the current epoch (12.8k such jobs x ~4.2 s = the measured M6
# pilot tail). The record below is written inside publish_generation's B4
# switch transaction, so it is crash-atomically consistent with the committed
# active set: the only place the active set changes is that transaction.
# ---------------------------------------------------------------------------

CORPUS_STATS_META_KEY = "corpus_stats_epoch"


def corpus_stats_record_load(db: Database) -> dict[str, object] | None:
    """The last persisted corpus-epoch record, or None (no publish yet)."""
    row = db.query_one("SELECT value FROM meta WHERE key = ?", (CORPUS_STATS_META_KEY,))
    if row is None:
        return None
    try:
        held = json.loads(row["value"])
    except ValueError:
        return None
    if not isinstance(held, dict) or not isinstance(held.get("stats_sha"), str):
        return None
    return held


def corpus_stats_record_store(db: Database, stats: SparseStats, bm25_sha: str, now: float) -> None:
    """Persist the corpus-epoch record.

    Callers must run this inside the B4 switch transaction so the record
    commits exactly with the publication switch it describes (a crash between
    the two would otherwise leave the record pointing at an uncommitted set).
    """
    value = json.dumps(
        {"stats_sha": stats.stats_sha, "bm25": bm25_sha, "doc_count": stats.doc_count, "at": now},
        sort_keys=True,
        separators=(",", ":"),
    )
    if db.query_one("SELECT 1 AS x FROM meta WHERE key = ?", (CORPUS_STATS_META_KEY,)) is None:
        db.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (CORPUS_STATS_META_KEY, value))
    else:
        db.execute("UPDATE meta SET value = ? WHERE key = ?", (value, CORPUS_STATS_META_KEY))


def corpus_stats_version_part(db: Database) -> str:
    """The stats-sha part of a publish job's ``input_version`` (M6 fix B1).

    The recorded corpus epoch, or the sentinel ``"init"`` when no publish has
    committed one. The version is an idempotency hint only — the publish
    handler (re)uses the real epoch under the publish lock, and the epoch
    fan-out converges republishes — so this must stay O(1) and must never
    trigger a corpus-wide compute.
    """
    record = corpus_stats_record_load(db)
    return str(record["stats_sha"]) if record is not None else "init"
