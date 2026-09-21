"""Pilot latency: p50/p95 retrieval latency with ingestion on/off (PRD §12).

Two phases over the same index, same probes, same Qdrant client and embedder:

* ``idle`` — no worker is running;
* ``ingesting`` — a real worker thread drains a backlog of *ingest_count*
  books scanned from *ingest_root* into the same sandbox while the probes
  run. The difference between the phases is the ingestion load itself.

Probe queries come from the operator's question bank when one is given
(its questions, verbatim); otherwise they are deterministically sampled
from indexed chunk text, so the command is reproducible without a bank.
Every latency is the wall time of one ``search_candidates`` call, in ms.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

from .config import Config
from .db import Database, db_path_for
from .embeddings import Embedder
from .indexing import QdrantOps
from .jobs import Jobs
from .retrieval import search_candidates
from .scan import iter_candidate_paths, process_paths
from .worker import check_version_gate, run_worker

__all__ = [
    "LatencyStats",
    "build_probe_queries",
    "measure_phase",
    "run_latency",
]


def _percentile_ms(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, ceil(pct / 100.0 * len(ordered)) - 1))
    return round(ordered[idx], 3)


def _stats(phase: str, samples: list[float]) -> LatencyStats:
    n = len(samples)
    mean = round(sum(samples) / n, 3) if n else None
    return LatencyStats(
        phase=phase,
        n=n,
        mean_ms=mean,
        p50_ms=_percentile_ms(samples, 50),
        p95_ms=_percentile_ms(samples, 95),
        max_ms=round(max(samples), 3) if n else None,
    )


@dataclass(frozen=True)
class LatencyStats:
    """Latency distribution of one measurement phase, in ms."""

    phase: str
    n: int
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_probe_queries(
    db: Database,
    *,
    questions: list[str] | None = None,
    n: int = 10,
    seed: int = 42,
) -> list[str]:
    """Probe queries: the bank's questions when given, else sampled chunk text.

    The sampled variant is deterministic: chunks are ordered by a SHA-256 of
    ``seed:chunk_id`` and each contributes its longest sentence (truncated to
    300 chars), so the same index + parameters give the same probes.
    """
    if questions:
        return [q for q in questions if q.strip()][:n]
    rows = db.query(
        "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
    )
    if not rows:
        return []

    def _rank(cid: str) -> str:
        return hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest()

    sample = sorted(rows, key=lambda r: (_rank(str(r["chunk_id"])), str(r["chunk_id"])))
    out: list[str] = []
    for row in sample:
        if len(out) >= n:
            break
        text = str(row["text"]).replace("\n", " ")
        sentences = [s.strip() for s in text.split(". ") if len(s.strip()) >= 40]
        if not sentences:
            continue
        probe = max(sentences, key=len)
        if len(probe) > 300:
            probe = probe[:300].rsplit(" ", 1)[0] + "..."
        out.append(probe)
    return out


def measure_phase(
    db: Database,
    cfg: Config,
    qdrant: QdrantOps,
    embedder: Embedder | None,
    queries: list[str],
    *,
    limit: int = 20,
    warmup: int = 3,
) -> list[float]:
    """Wall ms per ``search_candidates`` call, after a short warmup."""
    for i in range(warmup):
        search_candidates(db, cfg, qdrant, embedder, queries[i % len(queries)], limit=limit)
    samples: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        search_candidates(db, cfg, qdrant, embedder, q, limit=limit)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def run_latency(
    db: Database,
    cfg: Config,
    qdrant: QdrantOps,
    embedder: Embedder | None,
    *,
    probe_queries: list[str],
    limit: int = 20,
    reps: int = 1,
    ingest_root: str | Path | None = None,
    ingest_count: int = 0,
) -> dict[str, object]:
    """Measure both phases; return ``{"probes", "idle", "ingesting"|None, "enqueued"}``.

    Each phase is *reps* passes over the probes (wall ms per
    ``search_candidates`` call, concatenated), so p95 is not the max of a
    single pass. With *ingest_root* set, at most ``ingest_count`` books are
    scanned from that root into the same state database (the sandbox's), and
    a worker thread drains them while the ``ingesting`` phase is measured.
    The worker stops before the function returns; any backlog it leaves
    behind stays queued in the sandbox, which is the intended behavior (the
    sandbox is throwaway).
    """
    if not probe_queries:
        raise ValueError("no probe queries")
    if reps < 1:
        raise ValueError("reps must be >= 1")

    def _samples() -> list[float]:
        out: list[float] = []
        for _ in range(reps):
            out.extend(measure_phase(db, cfg, qdrant, embedder, probe_queries, limit=limit))
        return out

    idle = _stats("idle", _samples())

    result: dict[str, object] = {
        "probes": len(probe_queries),
        "limit": limit,
        "idle": idle.to_dict(),
        "ingesting": None,
        "enqueued": 0,
    }
    if ingest_root is None or ingest_count <= 0:
        return result

    root = Path(ingest_root)
    if not root.is_dir():
        raise ValueError(f"ingest root is not a directory: {root}")
    # The backlog is capped at ingest_count books: truncate the same
    # deterministic candidate list the scanner walks (sorted, no symlinks),
    # then apply a scan's exact per-file rules to just those paths. A live
    # Jobs handle is required — scan_roots/process_paths only enqueue extract
    # jobs when one is passed, so the backlog actually reaches the worker.
    candidates = iter_candidate_paths(
        root, cfg.scan.ignore_dirs, cfg.scan.ignore_files
    )[:ingest_count]
    report = process_paths(db, cfg, Jobs(db), candidates)
    enqueued = report.jobs_enqueued
    result["enqueued"] = enqueued
    if enqueued == 0:
        # Nothing to ingest (e.g. the books were already in the sandbox): the
        # busy phase would measure idle anyway — say so instead of pretending.
        return result

    # Cross-version execution gate (M7 slice 9): the worker thread below runs
    # whatever is queued, so a sandbox left behind by an older code version
    # must not be silently drained either.
    check_version_gate(db)
    stop = threading.Event()
    worker_db = Database.connect(db_path_for(cfg.paths.state_root))

    def _work() -> None:
        try:
            run_worker(
                worker_db,
                cfg,
                qdrant=qdrant,
                embedder=embedder,
                stop_event=stop.is_set,
                reconcile_on_start=False,
            )
        finally:
            worker_db.close()

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    # Let the worker reach steady state so the busy phase overlaps real work.
    time.sleep(1.0)
    busy = _stats("ingesting", _samples())
    stop.set()
    thread.join(timeout=120)
    result["ingesting"] = busy.to_dict()
    return result
