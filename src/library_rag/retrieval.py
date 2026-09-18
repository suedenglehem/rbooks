"""Retrieval: fused search over the published index (PRD §9).

The query path is stateless beyond SQLite + Qdrant, and it never trusts the
Qdrant flags alone (PRD §8F): the *visible set* is exactly the SQLite-active
publications, and every returned passage is re-validated in SQLite before it
may become evidence.

Per query:

  * dense top-N and one sparse top-N per statistics epoch are fused with
    reciprocal rank fusion (k = ``rrf_k``);
  * overlapping passages are deduplicated on chunk identity — the chunk key
    already encodes the span signature, so two overlapping passages of one run
    can never carry different IDs;
  * candidates are materialized from SQLite (chunk row exists, revision is
    active, publication is active); when validation drops hits, retrieval is
    topped up with a larger candidate limit — at most ``topup_max_rounds``
    rounds, never above ``topup_limit_cap`` (PRD §9);
  * up to ``rerank_max`` candidates are reranked (identity reranker in M4; the
    real bge-reranker-v2-m3 adapter arrives in M5 behind the same protocol);
  * 8-12 passages are selected under the token budget, diversified across
    books by default (``max_per_book``) and undiversified when a book filter
    (``doc_id``/``rev_id``) is in play — a deliberate book search must not be
    suppressed by the diversity cap.

Degraded mode: when embedding inference is unavailable (or not configured),
dense retrieval is skipped, the sparse side remains fully usable, and the
result is flagged ``degraded`` with a reason (PRD §9). An unreachable Qdrant
is an explicit :class:`IndexUnavailableError` — never fake results.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .config import Config, RetrievalSettings
from .db import Database
from .embeddings import (
    Embedder,
    ModelUnavailableError,
    bm25_weights,
    load_sparse_stats,
    tokenize,
)
from .indexing import (
    FieldCond,
    Hit,
    IndexFilter,
    QdrantOps,
    visible_pub_ids,
)

__all__ = [
    "IndexUnavailableError",
    "Passage",
    "PassthroughReranker",
    "Reranker",
    "RetrievalError",
    "SearchResult",
    "search",
]


class RetrievalError(RuntimeError):
    """A retrieval-path failure that is not a transient Qdrant outage."""


class IndexUnavailableError(RetrievalError):
    """Qdrant is unreachable: search cannot run.

    The CLI surfaces this as a non-zero exit with a message; the application
    never fabricates results for an unavailable index.
    """


@dataclass(frozen=True)
class Passage:
    """One validated evidence passage, materialized from SQLite."""

    chunk_id: str
    doc_id: str
    rev_id: str
    run_id: str
    gen_id: str
    pub_id: str
    title: str | None
    text: str
    spans: tuple[list[int], ...]
    score: float
    dense_rank: int | None
    sparse_rank: int | None


@dataclass(frozen=True)
class SearchResult:
    """The outcome of one search.

    *counts* is a stable diagnostic map: ``dense``/``sparse`` raw hits,
    ``fused`` deduplicated candidates, ``validated`` survivors,
    ``postvalidation_removed`` drops, ``reranked`` fed to the reranker,
    ``selected`` returned, and ``topup_rounds`` extra fetch rounds used.
    """

    query: str
    degraded: bool
    degraded_reason: str | None
    passages: tuple[Passage, ...]
    counts: dict[str, int]


@runtime_checkable
class Reranker(Protocol):
    """Reranks fused candidates against the query.

    Implementations may reorder and rescore; they must not drop candidates
    silently (selection handles the final cut).
    """

    def rerank(self, query: str, passages: Sequence[Passage]) -> list[Passage]: ...


class PassthroughReranker:
    """M4 default: preserves the fused (RRF) order.

    The real bge-reranker-v2-m3 adapter (M5) implements the same protocol.
    """

    def rerank(self, query: str, passages: Sequence[Passage]) -> list[Passage]:
        return list(passages)


@dataclass
class _Cand:
    """A fused, deduplicated candidate awaiting SQLite validation."""

    chunk_id: str
    score: float
    dense_rank: int | None
    sparse_rank: int | None
    payload: dict[str, object]


def _empty_counts() -> dict[str, int]:
    return {
        "dense": 0,
        "sparse": 0,
        "fused": 0,
        "validated": 0,
        "postvalidation_removed": 0,
        "reranked": 0,
        "selected": 0,
        "topup_rounds": 0,
    }


def search(
    db: Database,
    cfg: Config,
    qdrant: QdrantOps,
    embedder: Embedder | None,
    query: str,
    *,
    doc_id: str | None = None,
    rev_id: str | None = None,
    reranker: Reranker | None = None,
) -> SearchResult:
    """Fused dense+sparse search over the SQLite-visible publications.

    *doc_id*/*rev_id* restrict the search to one book (or one revision);
    giving either suppresses cross-book diversification (PRD §9).
    """
    if not query.strip():
        raise ValueError("query must be non-empty")
    if not qdrant.ping():
        raise IndexUnavailableError(
            "Qdrant is unreachable; search is unavailable "
            "(refusing to fabricate results)"
        )
    ret = cfg.retrieval
    rerank = reranker if reranker is not None else PassthroughReranker()

    # Explicit query encoding (PRD §8F); on unavailability, degrade to the
    # sparse-only path rather than failing the whole search (PRD §9).
    degraded = False
    degraded_reason: str | None = None
    query_vector: list[float] | None = None
    if embedder is None:
        degraded = True
        degraded_reason = "embedding model not configured; sparse-only search"
    else:
        try:
            query_vector = embedder.encode_query(query)
        except ModelUnavailableError as exc:
            degraded = True
            degraded_reason = f"embedding inference unavailable ({exc}); sparse-only search"

    visible = visible_pub_ids(db)
    extra_conds: tuple[FieldCond, ...] = ()
    if doc_id is not None:
        extra_conds += (FieldCond("doc_id", "eq", doc_id),)
    if rev_id is not None:
        extra_conds += (FieldCond("rev_id", "eq", rev_id),)

    if not visible or not qdrant.collection_exists():
        return SearchResult(query, degraded, degraded_reason, (), _empty_counts())

    epochs = _stats_epochs(db)
    if not epochs:
        return SearchResult(query, degraded, degraded_reason, (), _empty_counts())

    # Fetch, validate, and top up until we can satisfy the minimum selection
    # or the documented cap (PRD §9: "fetch more candidates when
    # postvalidation removes hits, with a documented cap").
    limit = max(ret.dense_top, ret.sparse_top)
    topup_rounds = 0
    prev_ids: frozenset[str] | None = None
    valid: list[Passage] = []
    candidates: list[_Cand] = []
    dense_count = sparse_count = 0
    while True:
        candidates, dense_count, sparse_count = _fuse(
            db,
            cfg,
            qdrant,
            query,
            visible=visible,
            extra_conds=extra_conds,
            epochs=epochs,
            query_vector=query_vector,
            limit=limit,
        )
        valid = _validate(db, candidates, visible)
        if len(valid) >= ret.min_passages:
            break
        if limit >= ret.topup_limit_cap or topup_rounds >= ret.topup_max_rounds:
            break
        ids = frozenset(c.chunk_id for c in candidates)
        if prev_ids is not None and ids <= prev_ids:
            # A larger limit surfaced nothing new; the pool is exhausted.
            break
        prev_ids = ids
        topup_rounds += 1
        limit = min(limit * 2, ret.topup_limit_cap)

    diversify = doc_id is None and rev_id is None
    passages = _select(
        _rerank(rerank, query, valid, ret.rerank_max), ret, diversify=diversify
    )
    counts = dict(_empty_counts())
    counts.update(
        dense=dense_count,
        sparse=sparse_count,
        fused=len(candidates),
        validated=len(valid),
        postvalidation_removed=len(candidates) - len(valid),
        reranked=min(len(valid), ret.rerank_max),
        selected=len(passages),
        topup_rounds=topup_rounds,
    )
    return SearchResult(query, degraded, degraded_reason, tuple(passages), counts)


# --- Fetch and fusion ----------------------------------------------------------

def _stats_epochs(db: Database) -> dict[str, tuple[str, ...]]:
    """Group the visible publications by their generation's statistics epoch.

    Points of different epochs are not comparable under one BM25 encoding, so
    the sparse side searches each epoch separately (PRD §8F/§9).
    """
    rows = db.query(
        """
        SELECT p.pub_id AS pub_id, g.sparse_stats_sha AS stats
        FROM publications p
        JOIN index_generations g ON g.gen_id = p.gen_id
        WHERE p.state = 'active'
        """
    )
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["stats"], []).append(r["pub_id"])
    return {k: tuple(v) for k, v in out.items()}


def _fuse(
    db: Database,
    cfg: Config,
    qdrant: QdrantOps,
    query: str,
    *,
    visible: frozenset[str],
    extra_conds: tuple[FieldCond, ...],
    epochs: dict[str, tuple[str, ...]],
    query_vector: list[float] | None,
    limit: int,
) -> tuple[list[_Cand], int, int]:
    """Dense top-*limit* + per-epoch sparse top-*limit*, RRF-fused and deduped.

    Returns the fused candidates (score-descending, deterministic tiebreak),
    the raw dense hit count, and the total raw sparse hit count.
    """
    ret = cfg.retrieval
    bm25 = ret.bm25
    dense_hits: list[Hit] = []
    if query_vector is not None:
        dense_f = IndexFilter.all(
            FieldCond("active", "eq", True),
            FieldCond("pub_id", "in", sorted(visible)),
            *extra_conds,
        )
        dense_hits = qdrant.dense_search(query_vector, limit, dense_f)

    # One ranked list per statistics epoch (their sparse vectors live in
    # different BM25 spaces); RRF merges rank-based scores across all of them.
    sparse_lists: list[list[Hit]] = []
    for stats_sha in sorted(epochs):
        stats = load_sparse_stats(db, stats_sha)
        if stats is None:
            raise RetrievalError(
                f"sparse stats row missing for active epoch {stats_sha[:16]}… "
                "(corrupt statistics; re-run a publication to rebuild it)"
            )
        indices, values = bm25_weights(query, stats, k1=bm25.k1, b=bm25.b)
        if not indices:
            continue
        f = IndexFilter.all(
            FieldCond("active", "eq", True),
            FieldCond("pub_id", "in", list(epochs[stats_sha])),
            *extra_conds,
        )
        hits = qdrant.sparse_search(indices, values, limit, f)
        if hits:
            sparse_lists.append(hits)

    fused: dict[str, _Cand] = {}

    def _add(hit: Hit, rank: int, which: str) -> None:
        cid = str(hit.payload.get("chunk_id") or hit.point_id)
        inc = 1.0 / (ret.rrf_k + rank)
        c = fused.get(cid)
        if c is None:
            fused[cid] = _Cand(
                cid,
                inc,
                rank if which == "dense" else None,
                rank if which == "sparse" else None,
                hit.payload,
            )
        else:
            c.score += inc
            if which == "dense" and c.dense_rank is None:
                c.dense_rank = rank
            elif which == "sparse" and c.sparse_rank is None:
                c.sparse_rank = rank

    for rank, h in enumerate(dense_hits, start=1):
        _add(h, rank, "dense")
    for hits in sparse_lists:
        for rank, h in enumerate(hits, start=1):
            _add(h, rank, "sparse")

    out = sorted(fused.values(), key=lambda c: (-c.score, c.chunk_id))
    return out, len(dense_hits), sum(len(h) for h in sparse_lists)


# --- SQLite postvalidation ------------------------------------------------------

def _validate(
    db: Database, candidates: list[_Cand], visible: frozenset[str]
) -> list[Passage]:
    """Materialize candidates from SQLite, dropping anything not currently
    valid evidence (chunk row, active revision, active publication).

    The point payload is a cache, never the source of truth: text, title, and
    spans come from the ``chunks`` row, and a payload whose identity fields
    disagree with the row (a stale point) is dropped.
    """
    if not candidates:
        return []
    ids = [c.chunk_id for c in candidates]
    marks = ",".join("?" * len(ids))
    rows = db.query(
        f"""
        SELECT c.chunk_id, c.run_id, c.rev_id, r.doc_id, c.text, c.title, c.spans
        FROM chunks c
        JOIN source_revisions r ON r.rev_id = c.rev_id
        WHERE c.chunk_id IN ({marks})
        """,
        ids,
    )
    by_id = {r["chunk_id"]: r for r in rows}
    active_revs: set[str] = set()
    revs = {r["rev_id"] for r in by_id.values()}
    if revs:
        marks2 = ",".join("?" * len(revs))
        active_revs = {
            r["rev_id"]
            for r in db.query(
                f"""
                SELECT rev_id FROM source_revisions
                WHERE rev_id IN ({marks2}) AND is_active = 1
                """,
                sorted(revs),
            )
        }

    out: list[Passage] = []
    for c in candidates:
        row = by_id.get(c.chunk_id)
        if row is None:
            continue
        if row["rev_id"] not in active_revs:
            continue
        pub = c.payload.get("pub_id")
        if not isinstance(pub, str) or pub not in visible:
            continue
        if (
            str(c.payload.get("rev_id")) != row["rev_id"]
            or str(c.payload.get("doc_id")) != row["doc_id"]
        ):
            continue  # stale point: payload no longer matches the chunk row
        out.append(
            Passage(
                chunk_id=c.chunk_id,
                doc_id=row["doc_id"],
                rev_id=row["rev_id"],
                run_id=row["run_id"],
                gen_id=str(c.payload.get("gen_id") or ""),
                pub_id=pub,
                title=row["title"],
                text=row["text"],
                spans=tuple(json.loads(row["spans"]) or []),
                score=c.score,
                dense_rank=c.dense_rank,
                sparse_rank=c.sparse_rank,
            )
        )
    return out


# --- Rerank and selection -------------------------------------------------------

def _rerank(
    reranker: Reranker, query: str, passages: list[Passage], cap: int
) -> list[Passage]:
    return reranker.rerank(query, passages[:cap])


def _select(
    passages: list[Passage], ret: RetrievalSettings, *, diversify: bool
) -> list[Passage]:
    """Pick 8-12 passages under the token budget, diversified across books.

    The per-book cap is a *soft* preference: the first pass honors it, and if
    that cannot reach ``min_passages`` (e.g. a single-book library), a second
    pass relaxes it. A book filter disables it entirely.
    """
    selected: list[Passage] = []
    deferred: list[Passage] = []
    per_book: dict[str, int] = {}
    used = 0

    def _fits(p: Passage) -> int:
        return len(tokenize(p.text))

    for p in passages:
        if len(selected) >= ret.max_passages:
            break
        w = _fits(p)
        if used + w > ret.passage_token_budget:
            continue  # does not fit the remaining budget; a smaller one may
        if diversify and per_book.get(p.doc_id, 0) >= ret.max_per_book:
            deferred.append(p)
            continue
        selected.append(p)
        per_book[p.doc_id] = per_book.get(p.doc_id, 0) + 1
        used += w

    if len(selected) < ret.min_passages:
        for p in deferred:
            if len(selected) >= ret.min_passages:
                break
            w = _fits(p)
            if used + w > ret.passage_token_budget:
                continue
            selected.append(p)
            per_book[p.doc_id] = per_book.get(p.doc_id, 0) + 1
            used += w
    return selected
