"""Indexing: Qdrant operations, generations, and the publication protocol.

The Qdrant side is a **single collection** (``library_chunks``) whose points
carry version identity in the payload (run/gen/pub ids, model revision, stats
epoch). The collection is created once and never recreated — versioning lives
in the payload, and the **SQLite ``publications`` table is the source of
truth** for what is visible (PRD §8F: "the SQLite row, not the Qdrant flags,
is the source of truth").

Publication is a four-boundary protocol that is safe under a crash at *any*
boundary (PRD §12 M4 gate "crash at each publication boundary never returns
uncommitted or obsolete evidence"):

  stage  upsert all of the new generation's points (``active`` unset/false);
         verify the exact point-ID set and count
  B1     SQLite: insert the publication row ``staged``
  B2     Qdrant: ``active=true`` where ``pub_id = new``
  B3     Qdrant: ``active=false`` where the doc's *other* publications are active
  B4     SQLite: new row -> ``active``, other active rows of the doc -> ``superseded``

Every step is idempotent, so re-running a crashed job (or
:func:`reconcile_publications` after a restart) converges to the same state.
Search never trusts the Qdrant flag alone: it filters on ``active=true``
**and** ``pub_id`` restricted to the SQLite-active publications, so a point
that was activated before its row switched is invisible until SQLite agrees.

Sparse vectors are client-side BM25 (see :mod:`library_rag.embeddings`); the
pinned Qdrant release has no server-side BM25 index.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from qdrant_client import QdrantClient
from qdrant_client import models as m

from .config import Config
from .db import Database
from .embeddings import (
    SparseStats,
    bm25_weights,
    checkpoint_path,
    compute_sparse_stats,
    corpus_stats_record_load,
    corpus_stats_record_store,
    embedding_sha,
    load_sparse_stats,
    read_checkpoint,
)
from .identity import generation_id, point_id, publication_id

__all__ = [
    "COLLECTION",
    "FieldCond",
    "Hit",
    "IndexFilter",
    "IndexingError",
    "PublicationError",
    "QdrantOps",
    "QdrantPoint",
    "RealQdrantOps",
    "acquire_publish_lock",
    "build_point",
    "publication_is_current",
    "publish_generation",
    "reconcile_publications",
    "release_publish_lock",
    "visible_pub_ids",
]

COLLECTION = "library_chunks"

# Points per Qdrant upsert call. Payloads are small (ids + flags), so this is
# bounded both for memory and for server-side batch limits.
_UPSERT_BATCH = 64

# How long a publish lock may be held before another worker may steal it.
# Must comfortably exceed one full publication (stats + full re-upsert).
_PUBLISH_LOCK_TTL = 900.0

_PUBLISH_LOCK_KEY = "publish_lock"
_LOCK_FREE = "free"


class IndexingError(RuntimeError):
    """Qdrant-side error that is not a transient connectivity problem."""


class PublicationError(IndexingError):
    """The publication protocol detected an inconsistency (failed verification)."""


# --- Filters -----------------------------------------------------------------

@dataclass(frozen=True)
class FieldCond:
    """One payload condition. *op* is ``eq``, ``ne``, or ``in`` (list value)."""

    key: str
    op: str
    value: object

    def matches(self, payload: dict[str, object]) -> bool:
        if self.op == "eq":
            return payload.get(self.key) == self.value
        if self.op == "ne":
            return payload.get(self.key) != self.value
        if self.op == "in":
            if not isinstance(self.value, (list, tuple, set, frozenset)):
                raise ValueError(f"filter op 'in' needs a list value, got {type(self.value).__name__}")
            return payload.get(self.key) in self.value
        raise ValueError(f"unknown filter op: {self.op}")


@dataclass(frozen=True)
class IndexFilter:
    """A conjunction of :class:`FieldCond`.

    ``matches`` is the pure-Python evaluation used by :class:`FakeQdrant` (and
    by tests asserting on filter shape); ``to_qdrant`` produces the server
    filter. The two must agree — that is part of what the tests check.
    """

    conds: tuple[FieldCond, ...] = ()

    @classmethod
    def all(cls, *conds: FieldCond) -> IndexFilter:
        return cls(conds=conds)

    def matches(self, payload: dict[str, object]) -> bool:
        return all(c.matches(payload) for c in self.conds)

    def to_qdrant(self) -> m.Filter | None:
        if not self.conds:
            return None
        # m.Condition is exactly the union Filter.must/must_not accept as list
        # elements, so annotating with it avoids the invariance mismatch.
        must: list[m.Condition] = []
        must_not: list[m.Condition] = []
        for c in self.conds:
            if c.op == "in":
                if not isinstance(c.value, (list, tuple, set, frozenset)):
                    raise ValueError(f"filter op 'in' needs a list value for {c.key}")
                # MatchAny is declared list[str] | list[int]; our "in" values
                # come from internal code (pub_id lists). The cast is sound —
                # pydantic validates the elements at model construction.
                vals = cast(list[str] | list[int], list(c.value))
                must.append(m.FieldCondition(key=c.key, match=m.MatchAny(any=vals)))
            elif c.op in ("eq", "ne"):
                if not isinstance(c.value, bool | int | str):
                    raise ValueError(f"filter op {c.op!r} needs a scalar value for {c.key}")
                cond = m.FieldCondition(key=c.key, match=m.MatchValue(value=c.value))
                (must if c.op == "eq" else must_not).append(cond)
            else:  # pragma: no cover - matches() raises first
                raise ValueError(f"unknown filter op: {c.op}")
        return m.Filter(must=must, must_not=must_not)


# --- Points and hits -----------------------------------------------------------

@dataclass(frozen=True)
class QdrantPoint:
    """One point to upsert. ``point_id`` is a deterministic UUID (server rule).

    The payload carries identity and the *staged* state only; ``active`` is
    never set at upsert time — it is flipped by :meth:`QdrantOps.set_active`,
    so a re-upsert of an already-active point can never deactivate it.
    """

    point_id: str
    payload: dict[str, object]
    dense: tuple[float, ...]
    sparse: tuple[tuple[int, ...], tuple[float, ...]]


@dataclass(frozen=True)
class Hit:
    """One search result: point id, score, and payload."""

    point_id: str
    score: float
    payload: dict[str, object]


# --- Qdrant operations interface ------------------------------------------------

@runtime_checkable
class QdrantOps(Protocol):
    """The surface of Qdrant the application uses (real client or fake)."""

    collection: str

    def ping(self) -> bool: ...

    def collection_exists(self) -> bool: ...

    def ensure_collection(self, *, dimensions: int) -> None: ...

    def upsert(self, points: Sequence[QdrantPoint]) -> None: ...

    def set_active(self, f: IndexFilter, active: bool) -> None: ...

    def count(self, f: IndexFilter) -> int: ...

    def ids(self, f: IndexFilter) -> frozenset[str]: ...

    def dense_search(self, vector: Sequence[float], limit: int, f: IndexFilter) -> list[Hit]: ...

    def sparse_search(
        self, indices: Sequence[int], values: Sequence[float], limit: int, f: IndexFilter
    ) -> list[Hit]: ...

    def delete(self, f: IndexFilter) -> None: ...


_PAYLOAD_INDEX_FIELDS: tuple[tuple[str, m.PayloadSchemaType], ...] = (
    ("doc_id", m.PayloadSchemaType.KEYWORD),
    ("rev_id", m.PayloadSchemaType.KEYWORD),
    ("pub_id", m.PayloadSchemaType.KEYWORD),
    ("chunk_id", m.PayloadSchemaType.KEYWORD),
    ("gen_id", m.PayloadSchemaType.KEYWORD),
    ("active", m.PayloadSchemaType.BOOL),
)


class RealQdrantOps:
    """Thin wrapper over ``qdrant_client.QdrantClient`` (server pinned v1.12.4).

    Only the verified API surface of the pinned client is used; nothing here
    depends on deprecated calls (no ``recreate_collection``).

    Two backends, selected by config (M6):

    * remote — ``qdrant_host``/``qdrant_port`` (a running Qdrant server);
    * local  — ``qdrant_path`` set: qdrant-client embedded mode with storage
      under that directory (the pilot sandbox; no server process needed).
      Local mode uses the same client API, so every method below is
      backend-agnostic.
    """

    def __init__(self, cfg: Config, *, timeout: int = 30) -> None:
        self._local = cfg.services.qdrant_path is not None
        if self._local:
            self._client = QdrantClient(path=cfg.services.qdrant_path, timeout=timeout)
        else:
            self._client = QdrantClient(
                host=cfg.services.qdrant_host, port=cfg.services.qdrant_port, timeout=timeout
            )
        self.collection = COLLECTION

    # -- lifecycle -------------------------------------------------------------
    def close(self) -> None:
        """Release the client (local mode drops its storage flock on close)."""
        self._client.close()

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        try:
            return bool(self._client.collection_exists(self.collection))
        except Exception:
            return False

    def ensure_collection(self, *, dimensions: int) -> None:
        if self._client.collection_exists(self.collection):
            self._check_dimensions(dimensions)
            self._ensure_payload_indexes()
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": m.VectorParams(size=dimensions, distance=m.Distance.COSINE)
            },
            sparse_vectors_config={
                "text": m.SparseVectorParams(index=m.SparseIndexParams(on_disk=False))
            },
            on_disk_payload=True,
        )
        self._ensure_payload_indexes()

    def _check_dimensions(self, dimensions: int) -> None:
        try:
            info = self._client.get_collection(self.collection)
            vectors = info.config.params.vectors
            dense = vectors.get("dense") if isinstance(vectors, dict) else vectors
            if dense is not None and int(dense.size) != int(dimensions):
                raise IndexingError(
                    f"collection {self.collection!r} has {dense.size} dense dimensions, "
                    f"configured {dimensions}; the embedding config changed — point the "
                    "state/qdrant roots at a fresh pair or migrate explicitly"
                )
        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError(f"could not inspect collection {self.collection!r}: {exc}") from exc

    def _ensure_payload_indexes(self) -> None:
        for key, schema in _PAYLOAD_INDEX_FIELDS:
            with contextlib.suppress(Exception):
                # "field index already exists" is the expected steady state.
                self._client.create_payload_index(
                    self.collection, field_name=key, field_schema=schema
                )

    # -- writes -----------------------------------------------------------------
    def upsert(self, points: Sequence[QdrantPoint]) -> None:
        if not points:
            return
        self._client.upsert(
            collection_name=self.collection,
            points=[
                m.PointStruct(
                    id=p.point_id,
                    vector={
                        "dense": list(p.dense),
                        "text": m.SparseVector(
                            indices=list(p.sparse[0]), values=list(p.sparse[1])
                        ),
                    },
                    payload=dict(p.payload),
                )
                for p in points
            ],
            wait=True,
        )

    def set_active(self, f: IndexFilter, active: bool) -> None:
        cf = f.to_qdrant()
        if cf is None:
            raise IndexingError("refusing to set payload on an unfiltered whole collection")
        # set_payload merges top-level keys, so only "active" changes.
        self._client.set_payload(
            collection_name=self.collection,
            points=m.FilterSelector(filter=cf),
            payload={"active": bool(active)},
            wait=True,
        )

    def delete(self, f: IndexFilter) -> None:
        cf = f.to_qdrant()
        if cf is None:
            raise IndexingError("refusing to delete an unfiltered whole collection")
        self._client.delete(
            collection_name=self.collection,
            points_selector=m.FilterSelector(filter=cf),
            wait=True,
        )

    # -- reads -------------------------------------------------------------------
    def count(self, f: IndexFilter) -> int:
        cf = f.to_qdrant()
        return int(self._client.count(self.collection, count_filter=cf).count)

    def ids(self, f: IndexFilter) -> frozenset[str]:
        out: set[str] = set()
        cf = f.to_qdrant()
        offset: int | str | None = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=cf,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            out.update(str(p.id) for p in points)
            if offset is None:
                break
        return frozenset(out)

    def dense_search(self, vector: Sequence[float], limit: int, f: IndexFilter) -> list[Hit]:
        res = self._client.query_points(
            collection_name=self.collection,
            query=list(vector),
            using="dense",
            limit=limit,
            query_filter=f.to_qdrant(),
            with_payload=True,
        )
        return [Hit(str(r.id), float(r.score), dict(r.payload or {})) for r in res.points]

    def sparse_search(
        self, indices: Sequence[int], values: Sequence[float], limit: int, f: IndexFilter
    ) -> list[Hit]:
        res = self._client.query_points(
            collection_name=self.collection,
            query=m.SparseVector(indices=list(indices), values=list(values)),
            using="text",
            limit=limit,
            query_filter=f.to_qdrant(),
            with_payload=True,
        )
        return [Hit(str(r.id), float(r.score), dict(r.payload or {})) for r in res.points]


# --- Fake (unit tests, no Docker) -----------------------------------------------

class _FakePoint:
    __slots__ = ("dense", "payload", "sparse")

    def __init__(
        self,
        payload: dict[str, object],
        dense: tuple[float, ...],
        sparse: tuple[tuple[int, ...], tuple[float, ...]],
    ) -> None:
        self.payload = payload
        self.dense = dense
        self.sparse = sparse


class FakeQdrant:
    """In-memory stand-in for the pinned Qdrant server for unit tests.

    Faithful to the contract that matters for correctness: upsert is
    idempotent by point ID, filters evaluate the payload exactly as
    :meth:`IndexFilter.matches`, dense scoring is dot product (cosine for the
    normalized vectors this app stores), sparse scoring is sparse dot product,
    and ``set_active`` merges the flag without touching other payload keys.
    """

    def __init__(self, dimensions: int = 1024) -> None:
        self.collection = COLLECTION
        self._dims = dimensions
        self._points: dict[str, _FakePoint] = {}

    def ping(self) -> bool:
        return True

    def collection_exists(self) -> bool:
        return bool(self._points)

    def ensure_collection(self, *, dimensions: int) -> None:
        self._dims = dimensions

    def upsert(self, points: Sequence[QdrantPoint]) -> None:
        for p in points:
            if len(p.dense) != self._dims:
                raise IndexingError(f"fake qdrant: expected {self._dims} dims, got {len(p.dense)}")
            self._points[p.point_id] = _FakePoint(
                dict(p.payload), tuple(p.dense), (tuple(p.sparse[0]), tuple(p.sparse[1]))
            )

    def set_active(self, f: IndexFilter, active: bool) -> None:
        for p in self._points.values():
            if f.matches(p.payload):
                p.payload["active"] = bool(active)

    def count(self, f: IndexFilter) -> int:
        return sum(1 for p in self._points.values() if f.matches(p.payload))

    def ids(self, f: IndexFilter) -> frozenset[str]:
        return frozenset(pid for pid, p in self._points.items() if f.matches(p.payload))

    def dense_search(self, vector: Sequence[float], limit: int, f: IndexFilter) -> list[Hit]:
        scored = []
        for pid, p in self._points.items():
            if not f.matches(p.payload):
                continue
            scored.append((self._dot(vector, p.dense), pid))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            Hit(pid, score, dict(self._points[pid].payload)) for score, pid in scored[:limit]
        ]

    def sparse_search(
        self, indices: Sequence[int], values: Sequence[float], limit: int, f: IndexFilter
    ) -> list[Hit]:
        q = dict(zip(indices, values, strict=True))
        scored = []
        for pid, p in self._points.items():
            if not f.matches(p.payload):
                continue
            doc = dict(zip(p.sparse[0], p.sparse[1], strict=True))
            score = sum(w * doc.get(i, 0.0) for i, w in q.items())
            if score != 0.0:
                scored.append((score, pid))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            Hit(pid, score, dict(self._points[pid].payload)) for score, pid in scored[:limit]
        ]

    def delete(self, f: IndexFilter) -> None:
        if not f.conds:
            raise IndexingError("refusing to delete an unfiltered whole collection")
        for pid in [pid for pid, p in self._points.items() if f.matches(p.payload)]:
            del self._points[pid]

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))


# --- Point building ---------------------------------------------------------------

def build_point(
    *,
    chunk_id: str,
    gen_id: str,
    pub_id: str,
    rev_id: str,
    doc_id: str,
    run_id: str,
    model_revision: str,
    embedding_sha_value: str,
    stats_sha_value: str,
    dense: Sequence[float],
    text: str,
    stats: SparseStats,
    k1: float,
    b: float,
) -> QdrantPoint:
    """Assemble one point from a chunk plus its encodings (PRD §6 identity)."""
    indices, values = bm25_weights(text, stats, k1=k1, b=b)
    payload: dict[str, object] = {
        "chunk_id": chunk_id,
        "rev_id": rev_id,
        "doc_id": doc_id,
        "run_id": run_id,
        "gen_id": gen_id,
        "pub_id": pub_id,
        "model_revision": model_revision,
        "embedding_sha": embedding_sha_value,
        "stats_sha": stats_sha_value,
    }
    return QdrantPoint(
        point_id=point_id(chunk_id, gen_id),
        payload=payload,
        dense=tuple(dense),
        sparse=(tuple(indices), tuple(values)),
    )


# --- The publish lock ---------------------------------------------------------------

def acquire_publish_lock(db: Database, owner: str, *, ttl: float = _PUBLISH_LOCK_TTL, now: float | None = None) -> bool:
    """Acquire the global publication lock (serialized corpus epochs).

    The lock owner is the job's lease token, so a crashed holder is
    distinguishable from a live one and the lock can be stolen after *ttl*.
    Returns False when another live worker holds it (the caller should defer).
    """
    ts = time.time() if now is None else now
    with db.transaction():
        row = db.query_one("SELECT value FROM meta WHERE key = ?", (_PUBLISH_LOCK_KEY,))
        if row is not None:
            try:
                held = json.loads(row["value"])
            except ValueError:
                held = None
            if (
                isinstance(held, dict)
                and held.get("owner") not in (None, _LOCK_FREE)
                and ts - float(held.get("at", 0.0)) < ttl
            ):
                return False
        value = json.dumps({"owner": owner, "at": ts})
        if row is None:
            db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)", (_PUBLISH_LOCK_KEY, value)
            )
        else:
            db.execute(
                "UPDATE meta SET value = ? WHERE key = ?", (value, _PUBLISH_LOCK_KEY)
            )
        return True


def release_publish_lock(db: Database, owner: str, *, now: float | None = None) -> None:
    """Release the lock if (and only if) we still hold it."""
    ts = time.time() if now is None else now
    with db.transaction():
        row = db.query_one("SELECT value FROM meta WHERE key = ?", (_PUBLISH_LOCK_KEY,))
        if row is None:
            return
        try:
            held = json.loads(row["value"])
        except ValueError:
            held = None
        if isinstance(held, dict) and held.get("owner") == owner:
            db.execute(
                "UPDATE meta SET value = ? WHERE key = ?",
                (json.dumps({"owner": _LOCK_FREE, "at": ts}), _PUBLISH_LOCK_KEY),
            )


# --- Visibility ---------------------------------------------------------------------

def visible_pub_ids(db: Database) -> frozenset[str]:
    """The pub ids a search may return: SQLite-active publications only.

    This is the application-level validation the PRD requires: Qdrant flags
    alone are never trusted (a crashed activation can leave points ``active``
    whose row never switched).
    """
    rows = db.query("SELECT pub_id FROM publications WHERE state = 'active'")
    return frozenset(r["pub_id"] for r in rows)


def publication_is_current(
    db: Database, *, rev_id: str, gen_id: str
) -> str | None:
    """State of the (rev, gen) publication, or None if no row exists."""
    row = db.query_one(
        "SELECT state FROM publications WHERE rev_id = ? AND gen_id = ?", (rev_id, gen_id)
    )
    return row["state"] if row is not None else None


# --- The publication protocol -----------------------------------------------------------


def _corpus_stats_for_publish(db: Database, cfg: Config, rev_id: str) -> SparseStats:
    """The statistics epoch a publication of *rev_id* joins (M6 fix B1).

    The record (see :func:`corpus_stats_record_load`) is committed inside this
    function's B4 switch transaction, so it always describes the *current*
    active set. A revision that is **already active** therefore joins exactly
    the recorded epoch — reusing it skips the O(corpus) tokenization that
    used to cost ~4.2 s on every stale-epoch republish (12.8k such jobs in
    the M6 pilot; that recompute *was* the tail). A not-yet-active revision
    grows the corpus, so it must compute.

    ``retrieval.stats_epoch == "frozen"`` (M6 fix B2) extends the reuse to
    not-yet-active revisions: the epoch is pinned to the record for the whole
    campaign (a new book's novel terms get zero sparse weight until the
    campaign-end re-freeze), and because every publication shares one sha,
    the epoch fan-out finds no mismatches and enqueues nothing.

    Missing or mismatched record (corpus-stats row deleted, k1/b changed,
    first publish ever) falls back to the full compute — self-healing.

    One transient: publishing a *replacement* revision of a doc whose old
    revision is still active computes against the pre-switch corpus (the old
    revision's chunks are included until the B4 switch supersedes them). The
    record then lags the true corpus until the next genuinely new revision
    computes; the epoch fan-out republishes the affected books under the
    corrected epoch, so this converges within one fan-out round.
    """
    record = corpus_stats_record_load(db)
    if record is not None and record.get("bm25") == cfg.retrieval.bm25.settings_sha():
        staged_active = (
            db.query_one(
                "SELECT 1 AS x FROM publications WHERE rev_id = ? AND state = 'active'",
                (rev_id,),
            )
            is not None
        )
        frozen = cfg.retrieval.stats_epoch == "frozen"
        if staged_active or frozen:
            stats = load_sparse_stats(db, str(record["stats_sha"]))
            if stats is not None:
                return stats
    return compute_sparse_stats(db, cfg, include_rev_id=rev_id)


def publish_generation(
    db: Database,
    cfg: Config,
    qdrant: QdrantOps,
    *,
    rev_id: str,
    run_id: str,
    on_progress: Callable[[], None],
) -> str:
    """Publish one revision under the current corpus-stats epoch.

    *on_progress* (called between upsert batches) is where the caller
    heartbeats its job lease. Returns ``"published"`` or ``"noop"`` (the
    generation is already active/superseded — nothing to do).

    The caller must hold the publish lock (see :func:`acquire_publish_lock`);
    this function does not lock, so it is safe to unit-test directly.
    """
    emb = cfg.embedding
    emb_sha = embedding_sha(cfg)
    bm25 = cfg.retrieval.bm25

    rev = db.query_one("SELECT doc_id FROM source_revisions WHERE rev_id = ?", (rev_id,))
    if rev is None:
        raise PublicationError(f"unknown revision: {rev_id}")
    doc_id = rev["doc_id"]

    # Corpus statistics for the epoch this publication will join (includes the
    # revision being staged, so the persisted row matches the post-switch set).
    # Fast path: an already-active revision reuses the recorded epoch (see
    # _corpus_stats_for_publish); only a genuinely new revision computes.
    stats = _corpus_stats_for_publish(db, cfg, rev_id)
    gen_id = generation_id(run_id, emb_sha, stats.stats_sha)
    pub_id = publication_id(rev_id, gen_id)

    state = publication_is_current(db, rev_id=rev_id, gen_id=gen_id)
    if state in ("active", "superseded"):
        return "noop"

    chunk_rows = db.query(
        "SELECT chunk_id, text FROM chunks WHERE rev_id = ? ORDER BY position", (rev_id,)
    )

    # The generation row (idempotent; points are verified below, not here).
    ts = time.time()
    with db.transaction():
        db.execute(
            """
            INSERT INTO index_generations (
                gen_id, run_id, rev_id, model_revision, embedding_sha, sparse_stats_sha,
                dimensions, dtype, normalized, state, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
            ON CONFLICT(gen_id) DO NOTHING
            """,
            (
                gen_id,
                run_id,
                rev_id,
                emb.effective_revision,
                emb_sha,
                stats.stats_sha,
                emb.dimensions,
                emb.dtype,
                1 if emb.normalize else 0,
                ts,
                ts,
            ),
        )

    expected_ids = frozenset(point_id(c["chunk_id"], gen_id) for c in chunk_rows)
    expected_count = len(chunk_rows)

    # Dense vectors are read back from checkpoints (PRD: index builds do not
    # repeat embeddings); a missing checkpoint is a consistency bug. The
    # checkpoint manifests key vectors by chunk id (point ids are minted only
    # here, from chunk id + generation), so the completeness check is on
    # chunk ids.
    vectors = _load_run_vectors(
        db, cfg, run_id, emb_sha, frozenset(c["chunk_id"] for c in chunk_rows)
    )

    # -- stage: upsert every point (never active); re-upsert is idempotent ----
    points = [
        build_point(
            chunk_id=c["chunk_id"],
            gen_id=gen_id,
            pub_id=pub_id,
            rev_id=rev_id,
            doc_id=doc_id,
            run_id=run_id,
            model_revision=emb.effective_revision,
            embedding_sha_value=emb_sha,
            stats_sha_value=stats.stats_sha,
            dense=vectors[c["chunk_id"]],
            text=c["text"],
            stats=stats,
            k1=bm25.k1,
            b=bm25.b,
        )
        for c in chunk_rows
    ]
    for i in range(0, len(points), _UPSERT_BATCH):
        qdrant.upsert(points[i : i + _UPSERT_BATCH])
        on_progress()

    # -- verify: exact ID set and count (PRD: "verify expected IDs/count") ----
    pub_filter = IndexFilter.all(FieldCond("pub_id", "eq", pub_id))
    got_ids = qdrant.ids(pub_filter)
    got_count = qdrant.count(pub_filter)
    if got_ids != expected_ids or got_count != expected_count:
        raise PublicationError(
            f"verification failed for {pub_id}: expected {expected_count} points, "
            f"got {got_count}"
        )

    # -- B1: the staged row (only after verification) --------------------------
    with db.transaction():
        db.execute(
            """
            INSERT INTO publications (
                pub_id, rev_id, doc_id, gen_id, run_id, expected_points, state, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'staged', ?)
            ON CONFLICT(pub_id) DO NOTHING
            """,
            (pub_id, rev_id, doc_id, gen_id, run_id, expected_count, ts),
        )

    # -- B2: activate the new points -------------------------------------------
    qdrant.set_active(IndexFilter.all(FieldCond("pub_id", "eq", pub_id)), True)

    # -- B3: deactivate this doc's other active publications -------------------
    other_ids = [
        r["pub_id"]
        for r in db.query(
            "SELECT pub_id FROM publications WHERE doc_id = ? AND state = 'active' AND pub_id != ?",
            (doc_id, pub_id),
        )
    ]
    if other_ids:
        qdrant.set_active(IndexFilter.all(FieldCond("pub_id", "in", list(other_ids))), False)

    # -- B4: switch in SQLite ----------------------------------------------------
    with db.transaction():
        db.execute(
            "UPDATE publications SET state = 'active', activated_at = ? WHERE pub_id = ?",
            (ts, pub_id),
        )
        db.execute(
            """
            UPDATE publications SET state = 'superseded', superseded_at = ?
            WHERE doc_id = ? AND pub_id != ? AND state = 'active'
            """,
            (ts, doc_id, pub_id),
        )
        # An older staged attempt of the same revision loses to this one.
        db.execute(
            """
            UPDATE publications SET state = 'superseded', superseded_at = ?
            WHERE rev_id = ? AND pub_id != ? AND state = 'staged'
            """,
            (ts, rev_id, pub_id),
        )
        db.execute(
            "UPDATE index_generations SET point_count = ?, updated_at = ? WHERE gen_id = ?",
            (expected_count, ts, gen_id),
        )
        # The epoch record commits atomically with the switch it describes
        # (the record stays valid for the committed active set — see
        # _corpus_stats_for_publish for the invariant this maintains).
        corpus_stats_record_store(db, stats, bm25.settings_sha(), ts)

    return "published"


def _load_run_vectors(
    db: Database, cfg: Config, run_id: str, emb_sha: str, expected_chunk_ids: frozenset[str]
) -> dict[str, tuple[float, ...]]:
    """Load every chunk's dense vector from its checkpoint (PRD: no re-embedding).

    The caller (the embed stage) guarantees the checkpoint rows and artifacts
    exist before publication; a missing or corrupt checkpoint is a consistency
    bug, surfaced as :class:`PublicationError`.
    """
    root = cfg.paths.artifact_root
    rows = db.query(
        """
        SELECT batch_index, chunk_ids FROM embedding_batches
        WHERE run_id = ? AND embedding_sha = ?
        ORDER BY batch_index
        """,
        (run_id, emb_sha),
    )
    if not rows:
        raise PublicationError(f"no checkpoint rows for run {run_id} / {emb_sha}")
    out: dict[str, tuple[float, ...]] = {}
    for b in rows:
        chunk_ids = json.loads(b["chunk_ids"])
        path = checkpoint_path(root, run_id, emb_sha, int(b["batch_index"]))
        vectors = read_checkpoint(path, chunk_ids)
        for cid, vec in zip(chunk_ids, vectors, strict=True):
            out[cid] = tuple(vec)
    missing = expected_chunk_ids - out.keys()
    if missing:
        raise PublicationError(
            f"{len(missing)} chunks have no checkpoint vector: {sorted(missing)[:3]}"
        )
    return out


def reconcile_publications(db: Database, cfg: Config, qdrant: QdrantOps, *, now: float | None = None) -> int:
    """Finish any publication a crash left half-done. Returns promotions made.

    Called at worker start and before search, so "search works after restart"
    holds even when no ingest worker has run since the crash (PRD §9/§12).
    Conservative by construction:

    * staged pub whose points are incomplete in Qdrant -> left alone (the open
      publish job finishes it; re-upsert is idempotent);
    * staged pub whose points are all present but inactive and which has no
      open publish job -> the row is orphaned and removed;
    * staged pub whose points are (at least partially) active -> the switch
      was interrupted after B2; complete B3 + B4 idempotently.
    """
    ts = time.time() if now is None else now
    promoted = 0
    for pub in db.query("SELECT * FROM publications WHERE state = 'staged'"):
        pub_id = pub["pub_id"]
        rev_id = pub["rev_id"]
        gen_id = pub["gen_id"]
        run_id = pub["run_id"]
        expected = int(pub["expected_points"])

        chunk_ids = [
            r["chunk_id"]
            for r in db.query(
                "SELECT chunk_id FROM chunks WHERE run_id = ? ORDER BY position", (run_id,)
            )
        ]
        expected_ids = frozenset(point_id(cid, gen_id) for cid in chunk_ids)
        if len(chunk_ids) != expected:
            # The run's chunks no longer match the staged expectation (should
            # not happen: chunks are immutable per run). Leave it for inspection.
            continue

        # A missing collection means every staged pub is incomplete: leave it
        # for the open publish job (which ensures the collection) rather than
        # treating absent points as a vanished index.
        has_collection = qdrant.collection_exists()
        got_ids = qdrant.ids(IndexFilter.all(FieldCond("pub_id", "eq", pub_id))) if has_collection else frozenset()
        if got_ids != expected_ids:
            continue  # points incomplete: the open publish job finishes it

        active = (
            qdrant.count(
                IndexFilter.all(FieldCond("pub_id", "eq", pub_id), FieldCond("active", "eq", True))
            )
            if has_collection
            else 0
        )
        if active == 0:
            open_job = db.query_one(
                """
                SELECT 1 AS x FROM jobs
                WHERE stage = 'publish' AND input_id = ?
                  AND state IN ('pending', 'running', 'retryable_failed')
                """,
                (rev_id,),
            )
            if open_job is None:
                # Orphaned: no job will ever complete it. Drop the row; the
                # (inactive) points are invisible to search and left to GC.
                with db.transaction():
                    db.execute("DELETE FROM publications WHERE pub_id = ?", (pub_id,))
            continue

        # Interrupted after B2: complete B3 + B4 (both idempotent).
        doc_id = pub["doc_id"]
        other_ids = [
            r["pub_id"]
            for r in db.query(
                "SELECT pub_id FROM publications WHERE doc_id = ? AND state = 'active' AND pub_id != ?",
                (doc_id, pub_id),
            )
        ]
        if other_ids:
            qdrant.set_active(IndexFilter.all(FieldCond("pub_id", "in", list(other_ids))), False)
        with db.transaction():
            db.execute(
                "UPDATE publications SET state = 'active', activated_at = ? WHERE pub_id = ?",
                (ts, pub_id),
            )
            db.execute(
                """
                UPDATE publications SET state = 'superseded', superseded_at = ?
                WHERE doc_id = ? AND pub_id != ? AND state = 'active'
                """,
                (ts, doc_id, pub_id),
            )
            db.execute(
                """
                UPDATE publications SET state = 'superseded', superseded_at = ?
                WHERE rev_id = ? AND pub_id != ? AND state = 'staged'
                """,
                (ts, rev_id, pub_id),
            )
        promoted += 1
    return promoted
