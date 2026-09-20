"""M6 pilot tooling: latency measurement, question bank, capacity report.

Unit-safe (FakeQdrant + FakeEmbedder, no Docker/GPU): the latency driver
against a real worker thread over a one-page PDF, the bank's write-path
invariants and determinism, and the report's aggregation/projection math on
synthetic inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from fixtures import ingest_and_publish, make_pdf
from library_rag.config import Config
from library_rag.db import Database
from library_rag.embeddings import make_embedder
from library_rag.indexing import FakeQdrant
from library_rag.latency import build_probe_queries, run_latency
from library_rag.pilot import SurveyRecord
from library_rag.pilot_report import REPORT_SCHEMA, build_report, render_markdown, write_report
from library_rag.questions import (
    QUESTION_BANK_SCHEMA,
    QuestionEntry,
    add_question,
    export_dataset,
    load_bank,
    next_question_id,
    save_bank,
    suggest_candidates,
)

_EN_PAGE = "The cat sat on the mat and the dog was there."
_EN_PAGE_2 = "More words on this second page of the book."
_LONG_PAGE = "The extraordinary protagonist discovered the laboratory at dawn."


def _seed_chunks(db: Database, n: int = 12) -> None:
    """Minimal catalog + chunk rows so probe sampling has distinct text to rank.

    ``build_probe_queries`` only reads ``chunks``; the parent rows satisfy the
    foreign keys without running the pipeline.
    """
    db.execute(
        "INSERT INTO documents (doc_id, anchor_sha256, created_at, updated_at) "
        "VALUES ('d1', ?, 0.0, 0.0)",
        ("a" * 64,),
    )
    db.execute(
        "INSERT INTO source_revisions (rev_id, doc_id, sha256, size_bytes, format, "
        "archive_relpath, first_path, is_active, created_at) "
        "VALUES ('r1', 'd1', ?, 1, 'pdf', 'a/x.pdf', '/books/x.pdf', 0, 0.0)",
        ("b" * 64,),
    )
    db.execute(
        "INSERT INTO extraction_runs (run_id, rev_id, doc_id, parser_version, settings_sha, "
        "unit_count, state, created_at, updated_at) "
        "VALUES ('run1', 'r1', 'd1', 'test', 'test', ?, 'succeeded', 0.0, 0.0)",
        (n,),
    )
    for i in range(n):
        db.execute(
            "INSERT INTO chunks (chunk_id, run_id, rev_id, position, text, token_count, "
            "spans, created_at) VALUES (?, 'run1', 'r1', ?, ?, 10, '[]', 0.0)",
            (f"c{i:02d}", i,
             f"Sentence {i} on page {i} of the laboratory manual describes experiment {i} in detail."),
        )


# --- question bank ------------------------------------------------------------


def _entry(
    id: str = "q001",
    category: str = "exact_term",
    answerable: bool = True,
    question: str = "What does the text say about the laboratory?",
    expected_chunks: tuple[str, ...] = (),
) -> QuestionEntry:
    return QuestionEntry(
        id=id,
        question=question,
        category=category,
        expected_chunks=expected_chunks,
        answerable=answerable,
    )


def test_bank_roundtrip_is_sorted_by_id(tmp_path: Path) -> None:
    p = tmp_path / "bank.json"
    save_bank(p, [_entry("q002"), _entry("q001")])
    loaded = load_bank(p)
    assert [e.id for e in loaded] == ["q001", "q002"]
    assert loaded[0].question == "What does the text say about the laboratory?"
    assert loaded[0].category == "exact_term"


def test_add_question_assigns_sequential_ids(tmp_path: Path) -> None:
    p = tmp_path / "bank.json"
    e1 = add_question(p, _entry(id=""))
    e2 = add_question(p, _entry(id=""))
    assert e1.id == "q001"
    assert e2.id == "q002"
    assert [e.id for e in load_bank(p)] == ["q001", "q002"]


def test_add_question_rejects_duplicate_id(tmp_path: Path) -> None:
    p = tmp_path / "bank.json"
    add_question(p, _entry("q001"))
    with pytest.raises(ValueError, match="already in the bank"):
        add_question(p, _entry("q001"))


def test_entry_invariants_reject_bad_combos() -> None:
    with pytest.raises(ValueError, match="unknown category"):
        QuestionEntry.from_dict({"id": "q1", "question": "Q", "category": "mystery"})
    with pytest.raises(ValueError, match="unanswerable requires answerable=false"):
        QuestionEntry.from_dict(
            {"id": "q1", "question": "Q", "category": "unanswerable", "answerable": True}
        )
    with pytest.raises(ValueError, match="answerable=false requires category=unanswerable"):
        QuestionEntry.from_dict(
            {"id": "q1", "question": "Q", "category": "conceptual", "answerable": False}
        )
    good = QuestionEntry.from_dict(
        {"id": "q1", "question": "Q", "category": "unanswerable", "answerable": False}
    )
    assert good.answerable is False


def test_load_bank_rejects_bad_files(tmp_path: Path) -> None:
    p = tmp_path / "bank.json"
    p.write_text(json.dumps({"schema": "nope", "questions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_bank(p)
    p.write_text(
        json.dumps(
            {
                "schema": QUESTION_BANK_SCHEMA,
                "questions": [
                    {"id": "q1", "question": "Q", "category": "conceptual"},
                    {"id": "q1", "question": "R", "category": "conceptual"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_bank(p)
    p.write_text(
        json.dumps(
            {"schema": QUESTION_BANK_SCHEMA, "questions": [{"id": "q1", "question": "  ",
                                                            "category": "conceptual"}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty"):
        load_bank(p)
    assert load_bank(tmp_path / "absent.json") == []  # absent file is an empty bank


def test_export_dataset_matches_evaluate_schema() -> None:
    entries = [_entry("q002", category="conceptual"), _entry("q001", expected_chunks=("c1", "c2"))]
    ds = export_dataset(entries)
    questions = cast("list[dict[str, Any]]", ds["questions"])
    assert [q["id"] for q in questions] == ["q001", "q002"]  # sorted by id
    q1, q2 = questions
    assert q1["expected_chunks"] == ["c1", "c2"]
    assert "expected_chunks" not in q2  # omitted when empty
    assert all(set(q) <= {"id", "question", "expected_chunks", "answerable"} for q in questions)


def test_next_question_id() -> None:
    assert next_question_id([]) == "q001"
    assert next_question_id([_entry("q001"), _entry("q003")]) == "q004"
    assert next_question_id([_entry("zz")]) == "q001"  # non qNNN ids are ignored


def test_suggest_candidates_are_deterministic_and_unlabeled(
    state_db: Database, base_config: Config
) -> None:
    base_config.embedding.fake = True
    src_root = base_config.paths.source_roots[0]
    src_root.mkdir(parents=True, exist_ok=True)
    pdf = src_root / "suggest.pdf"
    make_pdf(pdf, [_LONG_PAGE])
    ingest_and_publish(state_db, base_config, pdf, qdrant=FakeQdrant(base_config.embedding.dimensions))

    c1 = suggest_candidates(state_db, limit=5, seed=42)
    c2 = suggest_candidates(state_db, limit=5, seed=42)
    assert c1 == c2
    assert c1
    for c in c1:
        assert c["status"] == "candidate"  # prompts, never labels
        assert c["suggested_category"] in ("exact_term", "conceptual")
        assert c["chunk_id"]
        assert c["book"].endswith("suggest.pdf")


# --- capacity report -----------------------------------------------------------


def _rec(
    path: str,
    fmt: str = "pdf",
    ocr: str = "native",
    lang: str = "en",
    pages: int | None = 20,
    size: int = 1000,
    error: str | None = None,
) -> SurveyRecord:
    return SurveyRecord(path, size, 0.0, fmt, pages, ocr, lang, 16, 500, error)


def _records(n: int = 10) -> list[SurveyRecord]:
    return [_rec(f"/books/{i:03d}.pdf", pages=20) for i in range(n)]


def _run(tmp_path: Path | str = "/tmp/sandbox", **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "sandbox_root": str(tmp_path),
        "manifest_sha256": "x" * 64,
        "page_cap": None,
        "seconds": 60.0,
        "registration_seconds": 5.0,
        "worker_seconds": 55.0,
        "completed_jobs": 100,
        "documents": 5,
        "revisions": 5,
        "units": 100,
        "chunks": 250,
        "chunk_tokens": 10000,
        "chunk_chars": 50000,
        "publications_active": 5,
        "extraction_failures": 0,
        "pending_jobs": 0,
        "peak_rss_mb": 100.0,
        "embed_model_revision": "fake-v1",
        "embed_dimensions": 1024,
        "stages": {
            "extract": {"jobs": 5, "succeeded": 5, "failed": 0.0, "seconds": 10.0,
                        "seconds_max": 3.0},
            "embed": {"jobs": 5, "succeeded": 5, "failed": 0.0, "seconds": 40.0,
                      "seconds_max": 9.0},
        },
        "per_book": [
            {"rev_id": f"r{i}", "format": "pdf", "units": 20, "chars": 10000,
             "chunks": 50, "tokens": 2000}
            for i in range(5)
        ],
        "disk": {
            "qdrant": {"bytes": 1_000_000, "delta_bytes": 1_000_000},
            "state": {"bytes": 1_000_000, "delta_bytes": 500_000},
            "artifacts": {"bytes": 100_000, "delta_bytes": 100_000},
            "archive": {"bytes": 5_000_000, "delta_bytes": 5_000_000},
        },
        "failed_jobs": [],
    }
    base.update(over)
    return base


def _manifest() -> dict[str, Any]:
    return {
        "schema": "pilot-manifest/1",
        "target": 300,
        "seed": 42,
        "page_cap": 32,
        "survey_sha256": "y" * 64,
        "totals": {"candidates": 1000, "strata": 5, "sampled": 300},
        "strata": {"pdf|native|en": {"population": 800, "sampled": 200}},
    }


def test_report_with_no_inputs_is_empty_but_valid() -> None:
    report = build_report()
    assert report["schema"] == REPORT_SCHEMA
    assert report["corpus"] is None
    assert report["sample"] is None
    assert report["measured"] is None
    assert report["projections"] is None
    assert report["storage"] is None
    assert report["frozen_config"] is None
    assert report["questions"]["total"] == 0
    assert "not started" in report["questions"]["status"]


def test_corpus_section_counts_profiled_and_unprofiled() -> None:
    records = [*_records(4), _rec("/books/bad.pdf", error="open: no"),
               _rec("/books/ghost.pdf", pages=None)]
    corpus = build_report(survey=records)["corpus"]
    assert corpus["candidates"] == 6
    assert corpus["profiled"] == 4
    assert corpus["unprofiled"] == 2
    assert corpus["est_units"] == 80
    assert corpus["est_units_books"] == 4
    assert corpus["size_bytes"] == 6000
    assert corpus["by_format"] == {"pdf": 6}


def test_projections_scale_pilot_to_full_corpus(tmp_path: Path) -> None:
    report = build_report(survey=_records(10), run=_run(tmp_path))
    proj = report["projections"]
    assert proj is not None
    # 100 pilot units -> 250 chunks / 10000 tokens; full corpus = 200 units.
    assert proj["full_units_est"] == 200
    assert proj["full_chunks_est"] == 500
    assert proj["full_tokens_est"] == 20000
    assert proj["per_stage"]["extract"]["full_seconds_est"] == 20
    assert proj["per_stage"]["embed"]["full_seconds_est"] == 80
    assert proj["full_run_seconds_est"] == 100
    assert proj["embed_tokens_per_s"] == 250.0
    # All per-book ratios are identical (50/20), so the bootstrap collapses
    # to the point estimate.
    assert proj["bootstrap_ci_95"]["chunks"] == [500, 500]
    assert proj["bootstrap_ci_95"]["tokens"] == [20000, 20000]


def test_projections_require_both_survey_and_run(tmp_path: Path) -> None:
    assert build_report(run=_run(tmp_path))["projections"] is None
    assert build_report(survey=_records())["projections"] is None
    # A corpus with no profiled books gives no unit base to scale from.
    records = [_rec("/books/bad.pdf", error="open: no")]
    assert build_report(survey=records, run=_run(tmp_path))["projections"] is None
    # Bootstrap needs at least five per-book outcomes.
    small = _run(tmp_path, per_book=_run()["per_book"][:2])
    assert build_report(survey=_records(), run=small)["projections"]["bootstrap_ci_95"]["chunks"] is None


def test_storage_section_scales_disk_growth(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()  # the free-space probe statvfs <sandbox>/state
    storage = build_report(survey=_records(10), run=_run(tmp_path))["storage"]
    assert storage is not None
    # qdrant: (1e6 B / 250 chunks) * 500 est chunks * 2-generation headroom.
    assert storage["components"]["qdrant index (2-generation headroom)"] == 4_000_000
    # state: (5e5 B / 5 docs) * 10 profiled books.
    assert storage["components"]["state database"] == 1_000_000
    # artifacts: (1e5 B / 250 chunks) * 500 est chunks.
    assert storage["components"]["artifacts"] == 200_000
    assert storage["components"]["archive (one copy per book)"] == 10_000
    assert storage["total_bytes"] == 5_210_000
    assert storage["total_gb"] == round(5_210_000 / 1e9, 1)
    # The sandbox state volume exists (tmp_path) so the free-space probe works.
    assert storage["free_bytes_on_sandbox_state_volume"]
    # Without a run there is nothing to scale.
    assert build_report(survey=_records())["storage"] is None


def test_frozen_section_recommends_single_worker(tmp_path: Path) -> None:
    assert build_report()["frozen_config"] is None
    frozen = build_report(run=_run(tmp_path))["frozen_config"]
    assert frozen is not None
    assert frozen["workers"] == 1
    assert frozen["embed_model_revision"] == "fake-v1"
    assert frozen["embed_dimensions"] == 1024
    assert len(frozen["notes"]) == 2
    assert "chunking" not in frozen  # only added when a config is supplied


def test_frozen_section_includes_chunking_from_config(tmp_path: Path, base_config: Config) -> None:
    frozen = build_report(run=_run(tmp_path), config=base_config)["frozen_config"]
    assert frozen is not None
    c = frozen["chunking"]
    assert c["target_tokens"] == base_config.chunking.target_tokens
    assert c["overlap_tokens"] == base_config.chunking.overlap_tokens
    assert c["tokenizer"] == base_config.chunking.tokenizer


def test_questions_section_statuses() -> None:
    def status(n: int) -> str:
        entries = [
            QuestionEntry(id=f"q{i:03d}", question="Q?", category="conceptual") for i in range(n)
        ]
        return str(build_report(questions=entries)["questions"]["status"])

    assert "not started" in status(0)
    assert status(50).startswith("in progress")
    assert status(150) == "PRD target met"
    # by_category is zero-filled across all six categories.
    q = build_report(questions=[QuestionEntry(id="q001", question="Q?",
                                              category="unanswerable", answerable=False)])["questions"]
    assert q["by_category"]["unanswerable"] == 1
    assert q["by_category"]["exact_term"] == 0
    assert q["answerable"] == 0


def test_render_markdown_and_write_report(tmp_path: Path) -> None:
    entries = [QuestionEntry(id="q001", question="Q?", category="conceptual")]
    report = build_report(
        survey=_records(10),
        manifest=_manifest(),
        run=_run(tmp_path),
        questions=entries,
    )
    md = render_markdown(report)
    for header in (
        "# Pilot Capacity Report (PRD §12)",
        "## 1. Corpus",
        "## 2. Sample",
        "## 3. Measured pilot run",
        "## 4. Full-corpus projections (estimates)",
        "## 5. Storage estimate (full corpus)",
        "## 6. Question bank (operator-labeled)",
        "## 7. Recommended frozen config",
        "## 8. Uncertainty and assumptions",
    ):
        assert header in md
    out = tmp_path / "report.md"
    write_report(report, out)
    assert out.read_text(encoding="utf-8") == md


def test_render_markdown_without_inputs(tmp_path: Path) -> None:
    md = render_markdown(build_report())
    assert "# Pilot Capacity Report (PRD §12)" in md
    assert "Inputs: none" in md
    assert "## 1. Corpus" not in md


# --- latency -------------------------------------------------------------------


def _indexed(base_config: Config, state_db: Database) -> FakeQdrant:
    """One two-page PDF in the source root, drained to an active publication."""
    base_config.embedding.fake = True
    src_root = base_config.paths.source_roots[0]
    src_root.mkdir(parents=True, exist_ok=True)
    pdf = src_root / "indexed.pdf"
    make_pdf(pdf, [_EN_PAGE, _EN_PAGE_2])
    qdrant = FakeQdrant(base_config.embedding.dimensions)
    ingest_and_publish(state_db, base_config, pdf, qdrant=qdrant)
    return qdrant


def test_build_probe_queries_verbatim_needs_no_index(state_db: Database) -> None:
    assert build_probe_queries(state_db, questions=["A?", "", "B"], n=10) == ["A?", "B"]
    assert build_probe_queries(state_db, questions=["A?", "B"], n=1) == ["A?"]
    # No questions and no indexed chunks: an honest empty probe list.
    assert build_probe_queries(state_db) == []


def test_build_probe_queries_sample_is_deterministic(state_db: Database) -> None:
    _seed_chunks(state_db, n=12)
    p1 = build_probe_queries(state_db, n=10, seed=42)
    p2 = build_probe_queries(state_db, n=10, seed=42)
    p3 = build_probe_queries(state_db, n=10, seed=7)
    assert p1 == p2
    assert len(p1) == 10  # capped at n
    assert len(set(p1)) == 10  # distinct sentences, so the order is observable
    assert p1 != p3  # the seed changes the chunk order


def test_run_latency_idle_phase_only(state_db: Database, base_config: Config) -> None:
    qdrant = _indexed(base_config, state_db)
    embedder = make_embedder(base_config)
    result = run_latency(
        state_db, base_config, qdrant, embedder,
        probe_queries=["cat mat dog", "second page words"], limit=5,
    )
    assert result["probes"] == 2
    assert result["enqueued"] == 0
    assert result["ingesting"] is None
    idle = cast("dict[str, Any]", result["idle"])
    assert idle["phase"] == "idle"
    assert idle["n"] == 2
    assert idle["p50_ms"] is not None and idle["p95_ms"] is not None
    assert idle["max_ms"] >= idle["p50_ms"]


def test_run_latency_backlog_drains_while_measuring(
    state_db: Database, base_config: Config, tmp_path: Path
) -> None:
    qdrant = _indexed(base_config, state_db)
    embedder = make_embedder(base_config)
    backlog = tmp_path / "backlog"
    backlog.mkdir()
    make_pdf(backlog / "new.pdf", [_LONG_PAGE])

    result = run_latency(
        state_db, base_config, qdrant, embedder,
        probe_queries=["cat mat dog", "second page words"],
        limit=5, ingest_root=backlog, ingest_count=2,
    )
    assert result["enqueued"] == 1
    busy = cast("dict[str, Any] | None", result["ingesting"])
    assert busy is not None
    assert busy["phase"] == "ingesting"
    assert busy["n"] == 2

    # The book is now registered: a second pass enqueues nothing and skips
    # the busy phase entirely instead of pretending to measure load.
    again = run_latency(
        state_db, base_config, qdrant, embedder,
        probe_queries=["cat mat dog", "second page words"],
        limit=5, ingest_root=backlog, ingest_count=2,
    )
    assert again["enqueued"] == 0
    assert again["ingesting"] is None


def test_run_latency_rejects_bad_arguments(
    state_db: Database, base_config: Config, tmp_path: Path
) -> None:
    qdrant = _indexed(base_config, state_db)
    embedder = make_embedder(base_config)
    with pytest.raises(ValueError, match="no probe"):
        run_latency(state_db, base_config, qdrant, embedder, probe_queries=[])
    with pytest.raises(ValueError, match="reps"):
        run_latency(state_db, base_config, qdrant, embedder,
                    probe_queries=["q"], reps=0)
    with pytest.raises(ValueError, match="not a directory"):
        run_latency(state_db, base_config, qdrant, embedder, probe_queries=["q"],
                    ingest_root=tmp_path / "nope", ingest_count=1)
