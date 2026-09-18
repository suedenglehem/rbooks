"""M4 gate (PRD §12): crashes at each publication boundary never return
uncommitted or obsolete evidence; search works after restart.

The two-revision pipeline ends with pub2 active and pub1 superseded in *both*
SQLite and Qdrant. Each test then rewinds committed state to simulate a crash
between the publication's transaction boundaries, asserts what a pre-reconcile
search may see, runs :func:`reconcile_publications`, and asserts the recovered
evidence.
"""

from __future__ import annotations

import pytest

from fixtures import ingest_and_publish, make_pdf
from library_rag.config import Config
from library_rag.db import Database, db_path_for
from library_rag.embeddings import FakeEmbedder
from library_rag.indexing import FakeQdrant, FieldCond, IndexFilter, reconcile_publications
from library_rag.retrieval import search
from library_rag.worker import run_worker

V1 = "The first edition speaks of zebras in the savanna."
V2 = "The second edition speaks of quokkas, of koalas, and of wombats in the hills."


def _two_revisions(
    db: Database, cfg: Config, q: FakeQdrant
) -> tuple[str, str, str, str]:
    """Drive the real pipeline through v1 then v2 of the same file."""
    path = cfg.paths.source_roots[0] / "book.pdf"
    make_pdf(path, [V1])
    rev1, pub1 = ingest_and_publish(db, cfg, path, qdrant=q)
    make_pdf(path, [V2])
    rev2, pub2 = ingest_and_publish(db, cfg, path, qdrant=q)
    return rev1, pub1, rev2, pub2


def _set_flag(q: FakeQdrant, pub_id: str, active: bool) -> None:
    q.set_active(IndexFilter.all(FieldCond("pub_id", "eq", pub_id)), active)


def test_revision_replacement_supersedes_old_evidence(
    state_db: Database, base_config: Config
) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)

    path = base_config.paths.source_roots[0] / "book.pdf"
    make_pdf(path, [V1])
    rev1, pub1 = ingest_and_publish(state_db, base_config, path, qdrant=q)
    hits = search(state_db, base_config, q, emb, "zebras")
    assert len(hits.passages) >= 1
    assert all(p.rev_id == rev1 for p in hits.passages)
    assert any("zebras" in p.text for p in hits.passages)

    # Replace the source with a second edition; the pipeline republishes.
    make_pdf(path, [V2])
    rev2, pub2 = ingest_and_publish(state_db, base_config, path, qdrant=q)

    states = {
        r["pub_id"]: r["state"]
        for r in state_db.query("SELECT pub_id, state FROM publications")
    }
    assert states[pub1] == "superseded"
    assert states[pub2] == "active"

    # The old edition is invisible to search even when the query matches it
    # exactly (sparse hit on "zebras" must not resurrect superseded points).
    hits = search(state_db, base_config, q, emb, "zebras")
    assert all(p.rev_id == rev2 for p in hits.passages)
    hits = search(state_db, base_config, q, emb, "quokkas")
    assert any("quokkas" in p.text for p in hits.passages)

    # The old publication's points are de-flagged in the vector store.
    assert (
        q.count(IndexFilter.all(FieldCond("pub_id", "eq", pub1), FieldCond("active", "eq", True)))
        == 0
    )


@pytest.mark.parametrize(
    ("boundary", "promotions"),
    [
        ("pre-b1", 0),  # crash before the staged row is inserted
        ("b1", 0),  # staged row exists, points not flagged active yet
        ("b2", 1),  # flagged active, but still 'staged' in SQLite
        ("b3", 1),  # old edition's flag cleared, new one not promoted in DB
        ("b4", 0),  # clean state, nothing to reconcile
    ],
)
def test_crash_at_boundary_never_returns_bad_evidence(
    state_db: Database, base_config: Config, boundary: str, promotions: int
) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    rev1, pub1, rev2, pub2 = _two_revisions(state_db, base_config, q)

    # Final (committed) state: pub2 active, pub1 superseded; qdrant flags
    # pub2=True, pub1=False.
    if boundary in ("pre-b1", "b1"):
        _set_flag(q, pub1, True)  # crash before B2/B3 de-flagged the old edition
        _set_flag(q, pub2, False)
    if boundary == "b2":
        _set_flag(q, pub1, True)  # B2 ran (both flagged), B3/B4 did not

    if boundary == "pre-b1":
        state_db.execute("DELETE FROM publications WHERE pub_id = ?", (pub2,))
    elif boundary in ("b1", "b2", "b3"):
        state_db.execute(
            "UPDATE publications SET state = 'staged', activated_at = NULL "
            "WHERE pub_id = ?",
            (pub2,),
        )
        state_db.execute("UPDATE publications SET state = 'active' WHERE pub_id = ?", (pub1,))
    # b4: no rewind — the committed state must already be clean.

    pre = search(state_db, base_config, q, emb, "zebras quokkas")
    expected_pre = rev1 if boundary in ("pre-b1", "b1", "b2", "b3") else rev2
    if len(pre.passages) >= 1:
        assert all(p.rev_id == expected_pre for p in pre.passages), (
            f"pre-reconcile search at {boundary} must only show {expected_pre}"
        )

    assert reconcile_publications(state_db, base_config, q) == promotions

    post = search(state_db, base_config, q, emb, "zebras quokkas")
    if boundary in ("pre-b1", "b1"):
        # Rev2's publication is deleted as an orphan, and rev1 is an inactive
        # source revision (rev2 was already registered), so the safe outcome
        # is an evidence gap — never obsolete or uncommitted evidence.
        assert all(p.rev_id == rev1 for p in post.passages)
        assert not any(p.rev_id == rev2 for p in post.passages)
    else:
        assert len(post.passages) >= 1
        assert all(p.rev_id == rev2 for p in post.passages), (
            f"post-reconcile search at {boundary} must only show {rev2}"
        )


def test_search_works_after_restart(state_db: Database, base_config: Config) -> None:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    path = base_config.paths.source_roots[0] / "book.pdf"
    make_pdf(path, [V1])
    rev, _pub = ingest_and_publish(state_db, base_config, path, qdrant=q)

    first = search(state_db, base_config, q, emb, "zebras")
    assert len(first.passages) >= 1

    # A "restart": a brand-new database connection against the same state
    # directory, with the same standing-in Qdrant store.
    db2 = Database.connect(db_path_for(base_config.paths.state_root))
    try:
        assert run_worker(db2, base_config, once=True, poll_delay=0, qdrant=q) == 0
        second = search(db2, base_config, q, emb, "zebras")
        assert len(second.passages) >= 1
        assert all(p.rev_id == rev for p in second.passages)
    finally:
        db2.close()
