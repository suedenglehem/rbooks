"""M8 extended per-book summaries ("resumes") and keyword search.

Covers the five M8 seams: the schema migration, generation through the real
worker loop (durable job, scripted model, FTS mirror), the reconcile-style
enqueue backfill, the web-UI API surface, and the resume generation settings.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from fixtures import publish_handbuilt
from library_rag.api import create_app
from library_rag.config import Config, ConfigError, ResumeSettings, Services
from library_rag.db import Database, db_path_for
from library_rag.embeddings import FakeEmbedder
from library_rag.indexing import FakeQdrant
from library_rag.llm import FakeAnswerModel
from library_rag.migrations import MIGRATIONS, migrate
from library_rag.resumes import (
    enqueue_missing_resumes,
    get_resume,
    sample_resume_input,
    search_resumes,
    store_resume,
)
from library_rag.worker import run_worker

# A distinctive token so FTS hits are unambiguous and bm25 ranking is testable.
_MARKER = "phosphorescence"

_SENTENCE = (
    "The lighthouse keeper keeps a ledger of every ship that passes the "
    f"{_MARKER} off the northern shelf, and the ledger runs on without pause."
)


def _scripted_resume(words: int = 780) -> str:
    """A deterministic ~*words*-long generation (well past the 100-word floor)."""
    text = _SENTENCE
    while len(text.split()) < words:
        text += " " + _SENTENCE
    return text


_CHUNKS = [
    "The lighthouse keeper keeps a ledger of every ship that passes.",
    "On certain nights the shelf glows with a pale green light.",
    "The ledger runs on without pause, one ship a line.",
    "By spring the keeper had counted more than three hundred ships.",
    "The northern shelf kept its secret; the ledger kept its count.",
]


def _published(state_db: Database, cfg: Config, *, doc: str, rev: str, run: str) -> FakeQdrant:
    """One published revision with chunks (the real M4 hand-built fixture)."""
    cfg.embedding.fake = True
    q = FakeQdrant(cfg.embedding.dimensions)
    publish_handbuilt(
        state_db, cfg, q, doc_id=doc, rev_id=rev, run_id=run,
        texts=_CHUNKS, title="Resumed Book",
    )
    return q


# --- M8 schema -----------------------------------------------------------------


def test_m8_applies_fresh_and_idempotent(base_config: Config) -> None:
    path = db_path_for(base_config.paths.state_root)
    db = Database.connect(path)
    try:
        latest = len(MIGRATIONS)
        assert migrate(db) == latest
        assert migrate(db) == latest  # re-run is a no-op at the latest version
        names = {
            r["name"]
            for r in db.query(
                "SELECT name FROM sqlite_master "
                "WHERE name IN ('book_resumes', 'resumes_fts')"
            )
        }
        assert names == {"book_resumes", "resumes_fts"}
    finally:
        db.close()


# --- generation through the worker ---------------------------------------------


def test_resume_generation_end_to_end(state_db: Database, base_config: Config) -> None:
    base_config.answer.fake = True
    q = _published(state_db, base_config, doc="docR", rev="revR", run="runR")
    assert enqueue_missing_resumes(state_db, base_config) == 1
    text = _scripted_resume()
    handled = run_worker(
        state_db, base_config, once=True, poll_delay=0,
        qdrant=q, embedder=FakeEmbedder(base_config.embedding.dimensions),
        model=FakeAnswerModel([text]),
    )
    assert handled == 1
    row = state_db.query_one("SELECT * FROM jobs WHERE stage = 'resume'")
    assert row is not None
    assert row["state"] == "succeeded"
    manifest = json.loads(row["output_manifest"])
    assert manifest["rev_id"] == "revR"
    assert manifest["run_id"] == "runR"
    assert manifest["word_count"] == len(text.split())
    rec = get_resume(state_db, "revR")
    assert rec is not None
    assert rec["text"] == text
    assert rec["word_count"] == len(text.split())
    # Title comes from the revision's first_path stem (same rule as /library).
    assert rec["title"] == "revR"
    assert rec["prompt_version"] == "resume-v1"
    # The FTS mirror is in sync: the marker is findable and ranked.
    hits = search_resumes(state_db, _MARKER)
    assert [h["rev_id"] for h in hits] == ["revR"]


# --- enqueue backfill ------------------------------------------------------------


def test_enqueue_missing_resumes_is_idempotent(state_db: Database, base_config: Config) -> None:
    base_config.answer.fake = True
    _published(state_db, base_config, doc="docE", rev="revE", run="runE")
    assert enqueue_missing_resumes(state_db, base_config) == 1
    assert enqueue_missing_resumes(state_db, base_config) == 0
    row = state_db.query_one("SELECT * FROM jobs WHERE stage = 'resume'")
    assert row is not None
    assert row["state"] == "pending"
    assert row["input_id"] == "revE"
    assert row["input_version"] == "resume-v1:runE"


def test_enqueue_missing_resumes_gates(
    state_db: Database, base_config: Config
) -> None:
    # Answer model unconfigured: nothing to generate with, nothing enqueued.
    _published(state_db, base_config, doc="docG", rev="revG", run="runG")
    assert base_config.answer.is_configured is False
    assert enqueue_missing_resumes(state_db, base_config) == 0
    assert state_db.query_one("SELECT 1 AS x FROM jobs WHERE stage = 'resume'") is None
    # Resume stage disabled: same silence.
    base_config.answer.fake = True
    base_config.resume.enabled = False
    assert enqueue_missing_resumes(state_db, base_config) == 0
    # Both gates open: exactly one job.
    base_config.resume.enabled = True
    assert enqueue_missing_resumes(state_db, base_config) == 1


# --- API --------------------------------------------------------------------------


def test_resume_search_api(state_db: Database, base_config: Config) -> None:
    q = _published(state_db, base_config, doc="docS", rev="revS", run="runS")
    _published(state_db, base_config, doc="docS2", rev="revS2", run="runS2")
    dense = f"The {_MARKER} of the shelf. " * 4 + "The keeper counted ships."
    sparse = "The keeper counted ships. A note on " + _MARKER + "."
    store_resume(state_db, rev_id="revS", doc_id="docS", run_id="runS",
                 title="Searched Book", text=dense,
                 model_revision="m@1", prompt_version="resume-v1")
    store_resume(state_db, rev_id="revS2", doc_id="docS2", run_id="runS2",
                 title="Searched Book 2", text=sparse,
                 model_revision="m@1", prompt_version="resume-v1")
    app = create_app(
        base_config, state_db, qdrant=q,
        embedder=FakeEmbedder(base_config.embedding.dimensions),
        model=FakeAnswerModel(),
    )
    client = TestClient(app)

    # bm25 ranking: the resume that mentions the term more ranks first.
    r = client.post("/resumes/search", json={"query": _MARKER})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert [x["rev_id"] for x in results] == ["revS", "revS2"]
    assert results[0]["score"] < results[1]["score"]  # lower bm25 = better
    assert results[0]["doc_id"] == "docS"
    assert _MARKER in results[0]["excerpt"]

    # A single resume's full record.
    r2 = client.get("/resumes/revS")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["text"] == dense
    assert body["word_count"] == len(dense.split())
    assert body["model_revision"] == "m@1"
    assert body["prompt_version"] == "resume-v1"

    # Unknown revision → 404; empty query → 422.
    assert client.get("/resumes/nope").status_code == 404
    assert client.post("/resumes/search", json={"query": ""}).status_code == 422

    # The ingestion dashboard reports the stored resume count.
    assert client.get("/ingest/status").json()["resumes"] == 2


def test_resume_search_query_validation(state_db: Database, base_config: Config) -> None:
    q = _published(state_db, base_config, doc="docV", rev="revV", run="runV")
    app = create_app(base_config, state_db, qdrant=q,
                     embedder=FakeEmbedder(base_config.embedding.dimensions),
                     model=FakeAnswerModel())
    client = TestClient(app)
    assert client.post("/resumes/search", json={"query": ""}).status_code == 422
    assert client.post("/resumes/search", json={"query": "x" * 401}).status_code == 422
    # A query with no stored match returns an empty list, not an error.
    r = client.post("/resumes/search", json={"query": "zzzqqq"})
    assert r.status_code == 200
    assert r.json()["results"] == []


# --- sampling ----------------------------------------------------------------------


def test_sample_resume_input_uses_head_middle_tail(
    state_db: Database, base_config: Config
) -> None:
    _published(state_db, base_config, doc="docP", rev="revP", run="runP")
    sample = sample_resume_input(state_db, "runP", char_budget=10_000)
    # The budget is far above the fixture's total text, so head, middle and
    # tail must all be present and in reading order.
    assert _CHUNKS[0][: len(_CHUNKS[0])] in sample
    assert _CHUNKS[-1] in sample
    assert sample.index(_CHUNKS[0][:4]) < sample.index(_CHUNKS[-1])
    assert sample_resume_input(state_db, "no-such-run", char_budget=10_000) == ""


def test_sample_resume_input_tiny_books_do_not_crash(
    state_db: Database, base_config: Config
) -> None:
    # Regression: a run with exactly three chunks killed the worker —
    # _even_indices(3) is [] and the mid budget was divided by len([]).
    # Books with one, two, or three chunks degrade to head/tail instead.
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    for n, run in ((1, "runT1"), (2, "runT2"), (3, "runT3")):
        publish_handbuilt(
            state_db, base_config, q,
            doc_id=f"docT{n}", rev_id=f"revT{n}", run_id=run,
            texts=_CHUNKS[:n], title="Tiny Book",
        )
        sample = sample_resume_input(state_db, run, char_budget=10_000)
        assert sample, f"{n}-chunk run produced an empty sample"
        assert _CHUNKS[0][:4] in sample


# --- config ------------------------------------------------------------------------


def test_resume_settings_defaults(base_config: Config) -> None:
    r = base_config.resume
    assert r.enabled is True
    assert r.max_tokens == 2400
    assert r.temperature == 0.3
    assert r.timeout_seconds == 300.0
    assert r.prompt_version == "resume-v1"
    assert r.input_char_budget == 16000


def test_resume_settings_invalid_values_raise(base_config: Config) -> None:
    with pytest.raises(ConfigError, match=r"resume\.max_tokens"):
        ResumeSettings(max_tokens=0)
    with pytest.raises(ConfigError, match=r"resume\.temperature"):
        ResumeSettings(temperature=2.0)
    with pytest.raises(ConfigError, match=r"resume\.timeout_seconds"):
        ResumeSettings(timeout_seconds=0)
    with pytest.raises(ConfigError, match=r"resume\.input_char_budget"):
        ResumeSettings(input_char_budget=100)
    # A fully valid override is accepted by the top-level Config.
    cfg = Config(
        paths=base_config.paths,
        services=Services(),
        resume=ResumeSettings(max_tokens=3000, temperature=0.5),
    )
    assert cfg.resume.max_tokens == 3000
