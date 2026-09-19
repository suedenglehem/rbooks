"""M5 cited-answering orchestration (PRD §12) — the four gate cases:

1. **Unknown evidence IDs** are rejected after exactly ONE bounded repair
   attempt; the outcome is persisted ``failed`` with the evidence still
   returnable — never a fabricated fallback citation.
2. **Saved citations survive reindexing**: the manifest is a frozen snapshot,
   so a new run/publication for the same revision cannot move or rewrite what
   a saved answer points at.
3. **Abstention** is a first-class outcome: with zero evidence the pipeline
   abstains *without* calling the model; a contract ``ABSTAIN`` reply is
   persisted with its reason.
4. **Model server down**: the answer fails explicitly with an
   unavailability reason while ``search`` keeps working.
"""

from __future__ import annotations

import json
import socket
import time
from typing import cast

import pytest

from fixtures import publish_handbuilt
from library_rag.answers import (
    answer_query,
    get_answer,
    list_answers,
    resolve_citation,
)
from library_rag.citations import Evidence
from library_rag.config import Config
from library_rag.db import Database
from library_rag.embeddings import (
    FakeEmbedder,
    checkpoint_path,
    embedding_sha,
    tokenize,
    write_checkpoint,
)
from library_rag.indexing import FakeQdrant, publish_generation
from library_rag.llm import FakeAnswerModel, LlamaCppAnswerModel
from library_rag.retrieval import search

_TEXTS_A = [
    "The zebra grazes on the open savannah at dawn.",
    "Striped zebras run in tight family herds.",
    "A zebra's stripes are unique, like fingerprints.",
    "Zebra mule hybrids are called zorses.",
    "Plain zebra calves nurse within hours of birth.",
    "Equus quagga is the common name for the zebra.",
]


Library = tuple[
    Database, Config, FakeQdrant, FakeEmbedder,
    str, str, list[str],  # (state_db, cfg, qdrant, embedder), gen_id, pub_id, chunk_ids
]


@pytest.fixture()
def library(state_db: Database, base_config: Config) -> Library:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    gen, pub, chunks = publish_handbuilt(
        state_db,
        base_config,
        q,
        doc_id="docA",
        rev_id="revA",
        run_id="runA",
        texts=_TEXTS_A,
        title="Book A",
    )
    return state_db, base_config, q, emb, gen, pub, chunks


def _dead_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _evidence(stored: dict[str, object]) -> list[dict[str, object]]:
    """The persisted evidence list, narrowed from the row's object values."""
    return cast("list[dict[str, object]]", stored["evidence"])


def _purge_rev_runs(db: Database, *, rev_id: str) -> None:
    """Delete every extraction run of the revision, FK-safe (child tables
    first). This is what a replacing re-extraction looks like at the row
    level: ``publish_generation`` indexes ALL chunks of a revision but reads
    vectors only from the new run's checkpoints, so the old run's rows must
    be gone before the new publication."""
    runs = "SELECT run_id FROM extraction_runs WHERE rev_id = ?"
    db.execute(f"DELETE FROM publications WHERE gen_id IN (SELECT gen_id FROM index_generations WHERE run_id IN ({runs}))", (rev_id,))
    db.execute(f"DELETE FROM index_generations WHERE run_id IN ({runs})", (rev_id,))
    db.execute(f"DELETE FROM embedding_batches WHERE run_id IN ({runs})", (rev_id,))
    db.execute("DELETE FROM chunks WHERE rev_id = ?", (rev_id,))
    db.execute("DELETE FROM source_units WHERE rev_id = ?", (rev_id,))
    db.execute("DELETE FROM extraction_runs WHERE rev_id = ?", (rev_id,))


def _republish_run(
    db: Database,
    cfg: Config,
    qdrant: FakeQdrant,
    *,
    rev_id: str,
    run_id: str,
    texts: list[str],
    parser_version: str = "pdf/2",
) -> list[str]:
    """A second extraction run over the same revision: new chunk IDs, real
    checkpoints, and a real ``publish_generation`` (the reindexing event).

    A distinct ``parser_version`` stands in for whatever changed (a parser
    upgrade, a settings bump); the unique constraint on
    ``(rev_id, parser_version, settings_sha)`` forbids an identical re-run.
    The old run is purged first — see :func:`_purge_rev_runs`.
    """
    _purge_rev_runs(db, rev_id=rev_id)
    emb = FakeEmbedder(dimensions=cfg.embedding.dimensions)
    emb_sha = embedding_sha(cfg)
    ts = time.time()
    db.execute(
        """
        INSERT INTO extraction_runs (
            run_id, rev_id, doc_id, parser_version, settings_sha, unit_count,
            state, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 's1', ?, 'succeeded', ?, ?)
        """,
        (run_id, rev_id, "docA", parser_version, len(texts), ts, ts),
    )
    chunk_ids = [f"{run_id}:chunk-{i}" for i in range(len(texts))]
    for position, (cid, text) in enumerate(zip(chunk_ids, texts, strict=True)):
        db.execute(
            """
            INSERT INTO chunks (
                chunk_id, run_id, rev_id, position, text, token_count, title,
                spans, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, '[]', ?)
            """,
            (cid, run_id, rev_id, position, text, len(tokenize(text)), ts),
        )
    batch_size = max(1, cfg.embedding.batch_size)
    for i in range(0, len(texts), batch_size):
        batch_index = i // batch_size
        cids = chunk_ids[i : i + batch_size]
        cp = checkpoint_path(cfg.paths.artifact_root, run_id, emb_sha, batch_index)
        vector_sha = write_checkpoint(
            cp,
            emb.encode_documents(texts[i : i + batch_size]),
            emb.dimensions,
            {
                "run_id": run_id,
                "embedding_sha": emb_sha,
                "model_revision": emb.model_revision,
                "batch_index": batch_index,
                "chunk_ids": cids,
            },
        )
        db.execute(
            """
            INSERT INTO embedding_batches (
                batch_id, run_id, model_revision, embedding_sha, batch_index,
                chunk_ids, vector_sha256, artifact_relpath, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{run_id}:{emb_sha}:{batch_index:05d}",
                run_id,
                emb.model_revision,
                emb_sha,
                batch_index,
                json.dumps(cids),
                vector_sha,
                cp.relative_to(cfg.paths.artifact_root).as_posix(),
                ts,
            ),
        )
    publish_generation(db, cfg, qdrant, rev_id=rev_id, run_id=run_id, on_progress=lambda: None)
    row = db.query_one(
        "SELECT pub_id FROM publications WHERE rev_id = ? AND state = 'active'", (rev_id,)
    )
    assert row is not None
    return chunk_ids


# --- gate 1: unknown evidence IDs, exactly one bounded repair ---------------------


def test_unknown_citation_ids_fail_after_one_bounded_repair(library: Library) -> None:
    db, cfg, q, emb, _, _, _ = library
    model = FakeAnswerModel(["An answer citing [E1] and [E99].", "Still wrong: [E99]."])
    result = answer_query(db, cfg, q, emb, model, "zebra stripes")
    assert result.status == "failed"
    assert result.failure_reason == "unknown evidence IDs after one repair attempt: E99"
    assert result.citations == ()
    # Exactly one repair: two model calls total, the second carrying the
    # invalid assistant turn plus the repair instruction.
    assert len(model.calls) == 2
    assert [m["role"] for m in model.calls[1]] == ["system", "user", "assistant", "user"]
    # The evidence is persisted and returnable despite the failed answer.
    stored = get_answer(db, result.answer_id)
    assert stored is not None
    assert stored["status"] == "failed"
    assert len(_evidence(stored)) >= 1


def test_missing_citations_fail_after_one_repair(library: Library) -> None:
    db, cfg, q, emb, _, _, _ = library
    model = FakeAnswerModel(["No citations at all.", "Still none."])
    result = answer_query(db, cfg, q, emb, model, "zebra stripes")
    assert result.status == "failed"
    assert result.failure_reason == "answer contained no citations after one repair attempt"
    assert len(model.calls) == 2


def test_repair_success_is_answered(library: Library) -> None:
    db, cfg, q, emb, _, _, _ = library
    model = FakeAnswerModel(["Bad [E9].", "Fixed: zebras have stripes [E1] [E2]."])
    result = answer_query(db, cfg, q, emb, model, "zebra stripes")
    assert result.status == "answered"
    assert result.citations == ("E1", "E2")
    assert result.answer_text == "Fixed: zebras have stripes [E1] [E2]."
    assert len(model.calls) == 2  # one bad turn, one repair


# --- gate 2: saved citations survive reindexing ------------------------------------


def test_saved_citations_survive_reindexing(library: Library) -> None:
    db, cfg, q, emb, _, _, chunks_a = library
    model = FakeAnswerModel()
    result = answer_query(db, cfg, q, emb, model, "zebra stripes")
    assert result.status == "answered"
    first = get_answer(db, result.answer_id)
    assert first is not None
    evidence_before = _evidence(first)
    old_chunk_ids = {e["chunk_id"] for e in evidence_before}
    assert old_chunk_ids <= set(chunks_a)

    # Reindexing: a fresh run over the same revision -> new chunk IDs, a new
    # active publication, the old one superseded.
    chunks_b = _republish_run(db, cfg, q, rev_id="revA", run_id="runA2", texts=_TEXTS_A)
    assert set(chunks_b).isdisjoint(old_chunk_ids)
    now_visible = search(db, cfg, q, emb, "zebra stripes")
    assert all(p.chunk_id in set(chunks_b) for p in now_visible.passages)

    # The saved answer is byte-identical: the frozen snapshot is untouched.
    after = get_answer(db, result.answer_id)
    assert after is not None
    assert after["evidence"] == evidence_before
    assert after["citations"] == first["citations"]

    # Snapshot-based citation resolution: unavailable until the archived
    # original is present, then exact — pointing at the SAME (old) revision.
    entry = Evidence.from_dict(evidence_before[0])
    out = resolve_citation(db, cfg, entry)
    assert out["available"] is False
    assert out["reason"] == "archived original no longer present"
    assert out["excerpt"] == entry.text
    archive = cfg.paths.archive_root
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "revA.pdf").write_bytes(b"%PDF-1.4 test")
    out = resolve_citation(db, cfg, entry)
    assert out["available"] is True
    assert out["reason"] is None
    assert out["reader"] == {"manifest": "/books/revA", "source": "/books/revA/source"}
    assert out["excerpt"] == entry.text  # frozen text, not a live re-read


def test_citation_resolution_reports_missing_revision(library: Library) -> None:
    db, cfg, q, emb, _, _, _ = library
    model = FakeAnswerModel()
    result = answer_query(db, cfg, q, emb, model, "zebra stripes")
    assert result.status == "answered"
    stored = get_answer(db, result.answer_id)
    assert stored is not None
    entry = Evidence.from_dict(_evidence(stored)[0])
    archive = cfg.paths.archive_root
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "revA.pdf").write_bytes(b"%PDF-1.4 test")
    assert resolve_citation(db, cfg, entry)["available"] is True

    # The revision is deregistered (child rows first: foreign_keys is ON).
    db.execute("DELETE FROM publications WHERE rev_id = 'revA'")
    db.execute("DELETE FROM chunks WHERE rev_id = 'revA'")
    db.execute("DELETE FROM source_units WHERE rev_id = 'revA'")
    db.execute("DELETE FROM embedding_batches WHERE run_id IN (SELECT run_id FROM extraction_runs WHERE rev_id = 'revA')")
    db.execute("DELETE FROM index_generations WHERE run_id IN (SELECT run_id FROM extraction_runs WHERE rev_id = 'revA')")
    db.execute("DELETE FROM extraction_runs WHERE rev_id = 'revA'")
    db.execute("DELETE FROM source_revisions WHERE rev_id = 'revA'")
    out = resolve_citation(db, cfg, entry)
    assert out["available"] is False
    assert out["reason"] == "source revision no longer registered"


# --- gate 3: abstention --------------------------------------------------------------


def test_no_evidence_abstains_without_model_call(library: Library) -> None:
    db, cfg, q, emb, _, _, _ = library
    model = FakeAnswerModel()
    # A document filter that matches nothing published.
    result = answer_query(db, cfg, q, emb, model, "zebra", doc_id="docUnpublished")
    assert result.status == "abstained"
    assert result.abstain_reason == "no evidence found for this query"
    assert result.answer_text is None
    assert model.calls == []  # the model was never consulted
    stored = get_answer(db, result.answer_id)
    assert stored is not None
    assert stored["status"] == "abstained"
    assert stored["evidence"] == []


def test_scripted_abstention_is_persisted_with_reason(library: Library) -> None:
    db, cfg, q, emb, _, _, _ = library
    model = FakeAnswerModel(["ABSTAIN\nThe library does not cover this."])
    result = answer_query(db, cfg, q, emb, model, "quantum chromodynamics of zebras")
    assert result.status == "abstained"
    assert result.abstain_reason == "The library does not cover this."
    assert result.answer_text is None
    assert result.citations == ()
    assert len(model.calls) == 1
    stored = get_answer(db, result.answer_id)
    assert stored is not None
    assert stored["abstain_reason"] == "The library does not cover this."


# --- gate 4: answer model down, search unaffected -------------------------------------


def test_down_model_server_fails_answer_but_search_keeps_working(library: Library) -> None:
    db, cfg, q, emb, _, _, _ = library
    model = LlamaCppAnswerModel(
        "127.0.0.1",
        _dead_port(),
        "qwen",
        model_revision="qwen2.5-7b@dead",
        timeout_seconds=2.0,
        max_tokens=64,
        temperature=0.0,
    )
    result = answer_query(db, cfg, q, emb, model, "zebra stripes")
    assert result.status == "failed"
    assert result.failure_reason is not None
    assert "answer model unavailable" in result.failure_reason
    # The evidence was still found and persisted.
    stored = get_answer(db, result.answer_id)
    assert stored is not None
    assert len(_evidence(stored)) >= 1
    # Search is a different dependency and keeps working.
    hits = search(db, cfg, q, emb, "zebra stripes")
    assert len(hits.passages) >= 1


def test_unconfigured_model_fails_explicitly(library: Library) -> None:
    db, cfg, q, emb, _, _, _ = library
    result = answer_query(db, cfg, q, emb, None, "zebra stripes")
    assert result.status == "failed"
    assert result.failure_reason == "answer model not configured (set answer.model_revision)"
    stored = get_answer(db, result.answer_id)
    assert stored is not None
    assert len(_evidence(stored)) >= 1


# --- history ---------------------------------------------------------------------------


def test_answer_history_listing(library: Library) -> None:
    db, cfg, q, emb, _, _, _ = library
    model = FakeAnswerModel()
    r1 = answer_query(db, cfg, q, emb, model, "zebra stripes")
    r2 = answer_query(db, cfg, q, emb, model, "zebra mule")
    rows = list_answers(db, limit=10)
    assert [r["query"] for r in rows] == ["zebra mule", "zebra stripes"]
    assert {r["answer_id"] for r in rows} == {r1.answer_id, r2.answer_id}
    assert all("citation_count" in r for r in rows)
    assert len(list_answers(db, limit=1)) == 1


def test_get_answer_unknown_id_is_none(library: Library) -> None:
    db, *_ = library
    assert get_answer(db, "nope") is None
