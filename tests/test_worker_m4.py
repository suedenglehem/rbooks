"""M4 worker pipeline: embedding checkpoints, OOM/unavailability retry,
Qdrant-down defer-and-retry, and the lost-enqueue recovery windows
(PRD §12 M4 gate, §13, §14)."""

from __future__ import annotations

import json
from collections.abc import Sequence

from fixtures import make_pdf
from library_rag.config import Config
from library_rag.db import Database
from library_rag.embeddings import (
    Embedder,
    EmbeddingOOMError,
    FakeEmbedder,
    ModelUnavailableError,
    checkpoint_path,
)
from library_rag.indexing import FakeQdrant, FieldCond, IndexFilter, QdrantOps
from library_rag.jobs import Jobs
from library_rag.scan import scan_roots
from library_rag.worker import run_worker

_PAGES: list[str | None] = [
    "Chapter one introduces the lighthouse keeper.",
    "Chapter two follows the keeper's daughter to sea.",
    "Chapter three ends with the storm and the lamp.",
]


def _chunk_count(db: Database) -> int:
    row = db.query_one("SELECT COUNT(*) AS n FROM chunks")
    assert row is not None
    return int(row["n"])


def _rearm(db: Database, stage: str, input_id: str) -> None:
    """Reset a settled job to a freshly claimable one (crash simulation)."""
    db.execute(
        "UPDATE jobs SET state = 'pending', attempts = 0, lease_token = NULL, "
        "lease_expires_at = NULL, error_category = NULL, error_detail = NULL, "
        "next_attempt_at = NULL WHERE stage = ? AND input_id = ?",
        (stage, input_id),
    )


def _run(state_db: Database, base_config: Config, q: QdrantOps, embedder: Embedder) -> int:
    return run_worker(
        state_db, base_config, once=True, poll_delay=0, qdrant=q, embedder=embedder
    )


class CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that counts every text handed to ``encode_documents``."""

    def __init__(self, dimensions: int) -> None:
        super().__init__(dimensions)
        self.texts_encoded = 0

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts_encoded += len(texts)
        return super().encode_documents(texts)


class AlwaysOomEmbedder(FakeEmbedder):
    """Every batch fails with an OOM error (the halving loop gives up)."""

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingOOMError("simulated out-of-memory")


class FlakyQdrant(FakeQdrant):
    """A fake store whose reachability can be toggled; points are kept."""

    def __init__(self, dimensions: int, *, down: bool) -> None:
        super().__init__(dimensions)
        self.down = down

    def ping(self) -> bool:
        return not self.down


# --- 1. The full pipeline -------------------------------------------------------


def test_full_pipeline_publishes(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    make_pdf(base_config.paths.source_roots[0] / "book.pdf", _PAGES)
    scan_roots(state_db, base_config, Jobs(state_db))

    assert _run(state_db, base_config, q, emb) == 4  # extract, chunk, embed, publish
    assert Jobs(state_db).counts() == {"succeeded": 4}

    n = _chunk_count(state_db)
    assert n > 0
    assert state_db.query_one(
        "SELECT pub_id FROM publications WHERE state = 'active'"
    ) is not None
    assert q.count(IndexFilter.all()) == n
    assert q.count(IndexFilter.all(FieldCond("active", "eq", True))) == n

    # Every batch has a durable checkpoint under the embedding key.
    row = state_db.query_one(
        "SELECT er.run_id AS run_id, eb.embedding_sha AS sha "
        "FROM extraction_runs er JOIN embedding_batches eb ON eb.run_id = er.run_id"
    )
    assert row is not None
    cp = checkpoint_path(
        base_config.paths.artifact_root, str(row["run_id"]), str(row["sha"]), 0
    )
    assert cp.exists()


# --- 2./3. Checkpoint resume -----------------------------------------------------


def test_embed_resume_reuses_checkpoints(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = CountingEmbedder(base_config.embedding.dimensions)
    make_pdf(base_config.paths.source_roots[0] / "book.pdf", _PAGES)
    scan_roots(state_db, base_config, Jobs(state_db))

    assert _run(state_db, base_config, q, emb) == 4
    first = emb.texts_encoded
    assert first == _chunk_count(state_db)  # one pass over the whole book

    # Re-arm the settled embed job: a rerun must not re-encode (checkpoints).
    run = state_db.query_one("SELECT run_id FROM extraction_runs")
    assert run is not None
    run_id = str(run["run_id"])
    _rearm(state_db, "embed", run_id)
    assert _run(state_db, base_config, q, emb) == 1
    assert emb.texts_encoded == first
    assert Jobs(state_db).counts() == {"succeeded": 4}
    assert state_db.query_one(
        "SELECT pub_id FROM publications WHERE state = 'active'"
    ) is not None


def test_corrupt_checkpoint_is_reencoded(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = CountingEmbedder(base_config.embedding.dimensions)
    make_pdf(base_config.paths.source_roots[0] / "book.pdf", _PAGES)
    scan_roots(state_db, base_config, Jobs(state_db))

    assert _run(state_db, base_config, q, emb) == 4
    first = emb.texts_encoded

    # Flip one payload byte in every checkpoint (hash check must reject them).
    rows = state_db.query(
        "SELECT run_id, embedding_sha, batch_index FROM embedding_batches"
    )
    for r in rows:
        cp = checkpoint_path(
            base_config.paths.artifact_root, str(r["run_id"]), str(r["embedding_sha"]),
            int(r["batch_index"]),
        )
        raw = bytearray(cp.read_bytes())
        raw[21] ^= 0xFF
        cp.write_bytes(bytes(raw))

    run = state_db.query_one("SELECT run_id FROM extraction_runs")
    assert run is not None
    run_id = str(run["run_id"])
    _rearm(state_db, "embed", run_id)
    assert _run(state_db, base_config, q, emb) == 1
    assert emb.texts_encoded == 2 * first  # every batch re-encoded exactly once

    # The rewritten checkpoints are durable again, and the publication survived.
    for r in rows:
        cp = checkpoint_path(
            base_config.paths.artifact_root, str(r["run_id"]), str(r["embedding_sha"]),
            int(r["batch_index"]),
        )
        assert cp.read_bytes()[0:4] == b"LBEM"
    assert state_db.query_one(
        "SELECT pub_id FROM publications WHERE state = 'active'"
    ) is not None


# --- 4./5. Embedder failures are retryable --------------------------------------


def _run_until_embed_fails(
    state_db: Database, base_config: Config, q: FakeQdrant, bad_emb: FakeEmbedder
) -> None:
    make_pdf(base_config.paths.source_roots[0] / "book.pdf", _PAGES)
    scan_roots(state_db, base_config, Jobs(state_db))
    assert _run(state_db, base_config, q, bad_emb) == 3  # extract, chunk, embed(fail)
    assert Jobs(state_db).counts() == {"succeeded": 2, "retryable_failed": 1}


def test_embed_oom_is_retryable(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    _run_until_embed_fails(state_db, base_config, q, AlwaysOomEmbedder(32))

    job = state_db.query_one("SELECT * FROM jobs WHERE stage = 'embed'")
    assert job is not None and job["error_category"] == "embedding_oom"
    assert job["state"] == "retryable_failed"

    assert Jobs(state_db).retry() == 1
    # Reclaim: embed succeeds and hands off the single publish job (the
    # reconciler skips the run while its embed job is still in flight).
    assert _run(state_db, base_config, q, FakeEmbedder(base_config.embedding.dimensions)) == 2
    assert Jobs(state_db).counts() == {"succeeded": 4}
    assert state_db.query_one(
        "SELECT pub_id FROM publications WHERE state = 'active'"
    ) is not None
    assert q.count(IndexFilter.all(FieldCond("active", "eq", True))) == _chunk_count(state_db)


def test_embed_model_unavailable_is_retryable(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)

    class UnavailableEmbedder(FakeEmbedder):
        def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
            raise ModelUnavailableError("simulated model outage")

    _run_until_embed_fails(state_db, base_config, q, UnavailableEmbedder(32))
    job = state_db.query_one("SELECT * FROM jobs WHERE stage = 'embed'")
    assert job is not None and job["error_category"] == "model_unavailable"

    assert Jobs(state_db).retry() == 1
    assert _run(state_db, base_config, q, FakeEmbedder(base_config.embedding.dimensions)) == 2
    assert Jobs(state_db).counts() == {"succeeded": 4}
    assert state_db.query_one(
        "SELECT pub_id FROM publications WHERE state = 'active'"
    ) is not None


# --- 6. Qdrant down: publish is retryable, no fabricated state -------------------


def test_publish_defers_while_qdrant_is_down(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FlakyQdrant(base_config.embedding.dimensions, down=True)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    make_pdf(base_config.paths.source_roots[0] / "book.pdf", _PAGES)
    scan_roots(state_db, base_config, Jobs(state_db))

    # Embedding is offline from Qdrant: it succeeds, publish cannot.
    assert _run(state_db, base_config, q, emb) == 4
    assert Jobs(state_db).counts() == {"succeeded": 3, "retryable_failed": 1}
    job = state_db.query_one("SELECT * FROM jobs WHERE stage = 'publish'")
    assert job is not None and job["error_category"] == "qdrant_unavailable"
    # No publication row was created; no points were written.
    assert state_db.query_one("SELECT pub_id FROM publications") is None
    assert q.count(IndexFilter.all()) == 0

    q.down = False
    assert Jobs(state_db).retry() == 1
    # The retried publish job plus the reconciler's publication-gap job: one
    # publishes the deterministic publication, the other no-ops.
    assert _run(state_db, base_config, q, emb) == 2
    assert Jobs(state_db).counts() == {"succeeded": 5}
    pubs = state_db.query("SELECT pub_id FROM publications WHERE state = 'active'")
    assert len(pubs) == 1
    assert q.count(IndexFilter.all(FieldCond("active", "eq", True))) == _chunk_count(state_db)


# --- 7. Lost publication: reconcile republishes deterministically -----------------


def test_republish_after_publication_loss(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    make_pdf(base_config.paths.source_roots[0] / "book.pdf", _PAGES)
    scan_roots(state_db, base_config, Jobs(state_db))
    assert _run(state_db, base_config, q, emb) == 4

    before = state_db.query_one("SELECT pub_id, gen_id FROM publications")
    assert before is not None

    # The publication row vanished (crash between B1 and B4, then cleanup):
    # the worker's reconcile pass must close the gap on its own.
    state_db.execute("DELETE FROM publications")
    assert _run(state_db, base_config, q, emb) == 1

    after = state_db.query_one("SELECT pub_id, gen_id, state FROM publications")
    assert after is not None
    assert after["pub_id"] == before["pub_id"]  # deterministic re-creation
    assert after["gen_id"] == before["gen_id"]
    assert after["state"] == "active"
    assert q.count(IndexFilter.all(FieldCond("active", "eq", True))) == _chunk_count(state_db)


# --- 8. Lost checkpoints: reconcile re-embeds ------------------------------------


def test_reembed_after_checkpoint_loss(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = CountingEmbedder(base_config.embedding.dimensions)
    make_pdf(base_config.paths.source_roots[0] / "book.pdf", _PAGES)
    scan_roots(state_db, base_config, Jobs(state_db))
    assert _run(state_db, base_config, q, emb) == 4
    first = emb.texts_encoded
    row = state_db.query_one("SELECT pub_id FROM publications")
    assert row is not None
    pub_before = str(row["pub_id"])

    # Every checkpoint artifact and its manifest row is lost.
    rows = state_db.query("SELECT run_id, embedding_sha FROM embedding_batches")
    for r in rows:
        cp = checkpoint_path(
            base_config.paths.artifact_root, str(r["run_id"]), str(r["embedding_sha"]), 0
        )
        if cp.exists():
            cp.unlink()
    state_db.execute("DELETE FROM embedding_batches")
    state_db.execute("DELETE FROM jobs WHERE stage = 'embed'")

    assert _run(state_db, base_config, q, emb) == 1
    assert emb.texts_encoded == first * 2  # full re-encode
    row = state_db.query_one(
        "SELECT state FROM publications WHERE pub_id = ?", (pub_before,)
    )
    assert row is not None
    assert row["state"] == "active"  # the publication itself is untouched
    row = state_db.query_one("SELECT COUNT(*) AS n FROM embedding_batches")
    assert row is not None
    assert int(row["n"]) >= 1


# --- 9. Statistics epoch convergence ----------------------------------------------


def test_epoch_republish_converges(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    root = base_config.paths.source_roots[0]
    make_pdf(root / "a.pdf", _PAGES)
    scan_roots(state_db, base_config, Jobs(state_db))
    assert _run(state_db, base_config, q, emb) == 4

    # A second book changes the corpus: its publish must also republish the
    # first book against the new statistics epoch.
    make_pdf(root / "b.pdf", [
        "A ledger of tides, currents, and the price of salt.",
        "The harbor master keeps the light and the customs seal.",
        "In the end the ship goes out and the chart comes home.",
    ])
    scan_roots(state_db, base_config, Jobs(state_db))
    assert _run(state_db, base_config, q, emb) == 5  # B x4 + A republish

    # Converged: exactly two active publications, one statistics epoch.
    rows = state_db.query(
        """
        SELECT p.pub_id, p.state, g.sparse_stats_sha
        FROM publications p JOIN index_generations g ON g.gen_id = p.gen_id
        """
    )
    active = [r for r in rows if r["state"] == "active"]
    superseded = [r for r in rows if r["state"] == "superseded"]
    assert len(active) == 2 and len(superseded) == 1
    assert len({r["sparse_stats_sha"] for r in active}) == 1
    # A second drain is a no-op: the epoch is stable.
    assert _run(state_db, base_config, q, emb) == 0
    assert Jobs(state_db).counts() == {"succeeded": 9}


# --- 10. Publish job for a replaced revision is a no-op ---------------------------


def test_publish_noop_for_inactive_revision(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    make_pdf(base_config.paths.source_roots[0] / "book.pdf", _PAGES)
    scan_roots(state_db, base_config, Jobs(state_db))
    assert _run(state_db, base_config, q, emb) == 4
    rev = state_db.query_one("SELECT rev_id FROM source_revisions")
    assert rev is not None
    rev_id = str(rev["rev_id"])

    # The revision is replaced (e.g. a second edition registered afterwards):
    # any lingering publish job must be a no-op, never a publication.
    state_db.execute("UPDATE source_revisions SET is_active = 0 WHERE rev_id = ?", (rev_id,))
    state_db.execute(
        "INSERT INTO jobs (task_key, stage, input_id, input_version, state, attempts, "
        "max_attempts, created_at, updated_at) "
        "VALUES ('publish:manual', 'publish', ?, 'v', 'pending', 0, 3, 0.0, 0.0)",
        (rev_id,),
    )
    assert _run(state_db, base_config, q, emb) == 1

    job = state_db.query_one("SELECT * FROM jobs WHERE task_key = 'publish:manual'")
    assert job is not None and job["state"] == "succeeded"
    manifest = json.loads(job["output_manifest"] or "{}")
    assert manifest == {"rev_id": rev_id, "result": "noop"}
    # The active publication count is unchanged by the no-op.
    row = state_db.query_one(
        "SELECT COUNT(*) AS n FROM publications WHERE state = 'active'"
    )
    assert row is not None
    assert row["n"] == 1
