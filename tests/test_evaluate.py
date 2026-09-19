"""M5 evaluation (PRD §12): the JSON question dataset with strict field
validation, the end-to-end ``evaluate`` run over a published fake library
(recall@k, abstention on unanswerable questions, citation validity, latencies,
per-question rows), the ``format_report`` text, and the error paths — a
dataset error is a validation failure, a down index is an unavailable
failure (never a silent zero).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from fixtures import publish_handbuilt
from library_rag.config import Config
from library_rag.db import Database
from library_rag.embeddings import FakeEmbedder
from library_rag.evaluate import (
    evaluate,
    format_report,
    load_dataset,
)
from library_rag.indexing import FakeQdrant
from library_rag.llm import FakeAnswerModel
from library_rag.retrieval import IndexUnavailableError

_TEXTS_A = [
    "The zebra grazes on the open savannah at dawn.",
    "Striped zebras run in tight family herds.",
    "A zebra's stripes are unique, like fingerprints.",
    "Zebra mule hybrids are called zorses.",
    "Plain zebra calves nurse within hours of birth.",
    "Equus quagga is the common name for the zebra.",
]


Library = tuple[Database, Config, FakeQdrant, FakeEmbedder, list[str]]


class _DownQdrant(FakeQdrant):
    def ping(self) -> bool:
        return False


@pytest.fixture()
def library(state_db: Database, base_config: Config) -> Library:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    _, _, chunks = publish_handbuilt(
        state_db,
        base_config,
        q,
        doc_id="docA",
        rev_id="revA",
        run_id="runA",
        texts=_TEXTS_A,
        title="Book A",
    )
    return state_db, base_config, q, emb, chunks


# --- dataset loading ---------------------------------------------------------------


def test_load_dataset_round_trip_and_defaults() -> None:
    ds = load_dataset(
        {
            "questions": [
                {"id": "q1", "question": "what stripes?"},
                {"id": "q2", "question": "nope", "expected_chunks": ["runA:chunk-0"]},
            ]
        }
    )
    assert [q.id for q in ds.questions] == ["q1", "q2"]
    assert ds.questions[0].expected_chunks == ()
    assert ds.questions[0].answerable is True  # default
    assert ds.questions[1].expected_chunks == ("runA:chunk-0",)


@pytest.mark.parametrize(
    "payload,match",
    [
        ([1, 2], "dataset must be an object with a 'questions' list"),
        ({"questions": "nope"}, "dataset must be an object with a 'questions' list"),
        ({"questions": [42]}, "question 0 must be an object"),
        ({"questions": [{"question": "q"}]}, "question 0: 'id' must be a non-empty string"),
        ({"questions": [{"id": "q", "question": ""}]}, "'question' must be a non-empty string"),
        (
            {"questions": [{"id": "q", "question": "a"}, {"id": "q", "question": "b"}]},
            "duplicate question id: q",
        ),
        (
            {"questions": [{"id": "q", "question": "a", "expected_chunks": "x"}]},
            "'expected_chunks' must be a list of strings",
        ),
        (
            {"questions": [{"id": "q", "question": "a", "answerable": "yes"}]},
            "'answerable' must be a boolean",
        ),
    ],
    ids=[
        "not-object",
        "questions-not-list",
        "question-not-object",
        "missing-id",
        "empty-question",
        "duplicate-id",
        "bad-expected-chunks",
        "bad-answerable",
    ],
)
def test_load_dataset_rejects_invalid_datasets(payload: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        # The payloads are deliberately ill-typed (a bare list, a string
        # where a list is due); the validator must reject them by value.
        load_dataset(cast("Path | str | dict[str, object]", payload))


def test_load_dataset_reads_json_file(tmp_path: Path) -> None:
    f = tmp_path / "ds.json"
    f.write_text(json.dumps({"questions": [{"id": "q1", "question": "hi"}]}))
    ds = load_dataset(f)
    assert [q.id for q in ds.questions] == ["q1"]


# --- evaluate -----------------------------------------------------------------------


def _write_dataset(path: Path, questions: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"questions": questions}))
    return path


def test_evaluate_report_fields(library: Library, tmp_path: Path) -> None:
    db, cfg, q, emb, chunks = library
    # First question is unanswerable: the scripted ABSTAIN reply consumes the
    # script pop, the second question gets the default answered reply.
    ds = _write_dataset(
        tmp_path / "ds.json",
        [
            {"id": "q-no", "question": "quantum chromodynamics of zebras", "answerable": False},
            {"id": "q-yes", "question": "zebra stripes", "expected_chunks": [chunks[0]]},
        ],
    )
    model = FakeAnswerModel(["ABSTAIN\nThe library does not cover this."])
    report = evaluate(db, cfg, q, emb, model, load_dataset(ds), k=20)

    assert report["k"] == 20
    assert report["questions"] == 2
    assert report["labeled_questions"] == 1
    assert report["recall_at_k"] == 1.0  # the labeled chunk is in the top 20 of 6
    assert report["abstention"] == {"unanswerable": 1, "abstained": 1, "rate": 1.0}
    assert report["citations"] == {"answered": 1, "all_valid": 1, "rate": 1.0}
    for key in ("retrieval_ms", "answer_ms"):
        assert set(report["latency_ms"][key]) == {"p50", "p95"}
        assert report["latency_ms"][key]["p50"] >= 0
        assert report["latency_ms"][key]["p95"] >= report["latency_ms"][key]["p50"]

    per = report["per_question"]
    assert [e["id"] for e in per] == ["q-no", "q-yes"]  # dataset order
    assert per[0]["status"] == "abstained"
    assert per[0]["recall"] is None
    assert per[1]["status"] == "answered"
    assert per[1]["citations"] == ["E1"]
    assert per[1]["citations_valid"] is True
    assert per[1]["recall"] == 1.0
    # Both answers are persisted as first-class outcomes.
    from library_rag.answers import list_answers

    assert len(list_answers(db)) == 2


def test_evaluate_without_labels_reports_recall_none(library: Library, tmp_path: Path) -> None:
    db, cfg, q, emb, _ = library
    ds = _write_dataset(tmp_path / "ds.json", [{"id": "q1", "question": "zebra stripes"}])
    report = evaluate(db, cfg, q, emb, FakeAnswerModel(), load_dataset(ds), k=5)
    assert report["labeled_questions"] == 0
    assert report["recall_at_k"] is None
    text = format_report(report)
    assert "n/a (no labels)" in text
    assert "n/a (none marked unanswerable)" in text


def test_format_report_lines(library: Library, tmp_path: Path) -> None:
    db, cfg, q, emb, chunks = library
    ds = _write_dataset(
        tmp_path / "ds.json",
        [
            {"id": "q-no", "question": "quantum chromodynamics of zebras", "answerable": False},
            {"id": "q-yes", "question": "zebra stripes", "expected_chunks": [chunks[0]]},
        ],
    )
    model = FakeAnswerModel(["ABSTAIN\nThe library does not cover this."])
    report = evaluate(db, cfg, q, emb, model, load_dataset(ds), k=20)
    text = format_report(report)
    lines = text.splitlines()
    assert lines[0] == "evaluation: 2 questions (k=20, labeled 1)"
    assert "recall@20: 1.000" in text
    assert "abstention on unanswerable: 1/1 (1.000)" in text
    assert "citations valid on answered: 1/1 (1.000)" in text
    assert "retrieval p50/p95 ms:" in text
    assert "answer    p50/p95 ms:" in text
    assert "[abstained] q-no" in lines
    assert "recall=1.00" in text  # the labeled, answered question


def test_evaluate_down_qdrant_raises_unavailable(library: Library) -> None:
    db, cfg, _, emb, _ = library
    down = _DownQdrant(cfg.embedding.dimensions)
    with pytest.raises(IndexUnavailableError):
        evaluate(db, cfg, down, emb, FakeAnswerModel(), load_dataset({"questions": []}), k=5)
