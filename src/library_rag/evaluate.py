"""Retrieval and answering evaluation harness (PRD §13).

A dataset is a JSON file with explicit human-labeled questions::

    {"questions": [
        {"id": "q1", "question": "...",
         "expected_chunks": ["<chunk_id>", ...],   # optional label
         "answerable": true}]}                     # optional, default true

Metrics, all computed only over the labels that exist (nothing is invented):

* ``recall@k`` — mean per-question fraction of ``expected_chunks`` present
  among the top-k *candidates* (retrieval quality, independent of the model);
* abstention — for ``answerable: false`` questions, the share answered
  ``abstained`` (a correct refusal, not a failed answer);
* citation validity — for ``answered`` questions, the share whose cited IDs
  are all in the frozen manifest (must be 1.0 by construction; a regression
  here means the validation path was bypassed);
* latency — p50/p95 of retrieval and answer-model milliseconds.

Evaluation runs the real pipeline (``search_candidates`` for recall,
``answer_query`` for the rest), so persisted answers from an evaluation are
visible in the history like any other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from .answers import answer_query
from .config import Config
from .db import Database
from .embeddings import Embedder
from .indexing import QdrantOps
from .llm import AnswerModel
from .retrieval import IndexUnavailableError, search_candidates

__all__ = ["Dataset", "Question", "evaluate", "format_report", "load_dataset"]


@dataclass(frozen=True)
class Question:
    """One labeled dataset entry."""

    id: str
    question: str
    expected_chunks: tuple[str, ...] = ()
    answerable: bool = True


@dataclass(frozen=True)
class Dataset:
    questions: tuple[Question, ...]


def load_dataset(source: Path | str | dict[str, Any]) -> Dataset:
    """Load and validate an evaluation dataset from a JSON file or a parsed object."""
    if isinstance(source, (str, Path)):
        raw: Any = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        raw = source
    if not isinstance(raw, dict) or not isinstance(raw.get("questions"), list):
        raise ValueError("dataset must be an object with a 'questions' list")
    seen: set[str] = set()
    questions: list[Question] = []
    for i, item in enumerate(raw["questions"]):
        if not isinstance(item, dict):
            raise ValueError(f"question {i} must be an object")
        qid = item.get("id")
        text = item.get("question")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"question {i}: 'id' must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"question {qid}: 'question' must be a non-empty string")
        if qid in seen:
            raise ValueError(f"duplicate question id: {qid}")
        seen.add(qid)
        expected = item.get("expected_chunks", [])
        if not isinstance(expected, list) or not all(isinstance(c, str) for c in expected):
            raise ValueError(f"question {qid}: 'expected_chunks' must be a list of strings")
        answerable = item.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"question {qid}: 'answerable' must be a boolean")
        questions.append(Question(qid, text, tuple(expected), answerable))
    return Dataset(tuple(questions))


# --- metrics -----------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def _share(hits: int, total: int) -> float | None:
    return (hits / total) if total else None


def evaluate(
    db: Database,
    cfg: Config,
    qdrant: QdrantOps,
    embedder: Embedder | None,
    model: AnswerModel | None,
    dataset: Dataset,
    *,
    k: int = 20,
) -> dict[str, Any]:
    """Run the full pipeline per question and aggregate the report.

    Raises :class:`library_rag.retrieval.IndexUnavailableError` when Qdrant is
    unreachable (retrieval cannot be measured honestly).
    """
    per_question: list[dict[str, Any]] = []
    recalls: list[float] = []
    unanswerable = correct_abstentions = 0
    answered = all_citations_valid = 0
    retrieval_ms: list[float] = []
    model_ms: list[float] = []

    # Preflight: with Qdrant down no retrieval metric is honest — fail before
    # touching the dataset (also covers the empty-dataset case, where the
    # per-question loop below would never detect it).
    if not qdrant.ping():
        raise IndexUnavailableError(
            "Qdrant is unreachable; search is unavailable (refusing to fabricate results)"
        )

    for q in dataset.questions:
        # IndexUnavailableError propagates: with Qdrant down no retrieval
        # metric is honest, and the CLI maps that to an error exit.
        candidates, _counts = search_candidates(db, cfg, qdrant, embedder, q.question, limit=k)
        retrieved = [p.chunk_id for p in candidates]
        recall = None
        if q.expected_chunks:
            expected = set(q.expected_chunks)
            recall = len(set(retrieved) & expected) / len(expected)
            recalls.append(recall)
        result = answer_query(db, cfg, qdrant, embedder, model, q.question)
        retrieval_ms.append(result.retrieval_ms)
        model_ms.append(result.model_ms)
        entry: dict[str, Any] = {
            "id": q.id,
            "question": q.question,
            "status": result.status,
            "citations": list(result.citations),
            "retrieval_ms": round(result.retrieval_ms, 2),
            "model_ms": round(result.model_ms, 2),
            "recall": recall,
        }
        manifest_ids = {e.evidence_id for e in result.manifest.evidence}
        entry["citations_valid"] = bool(manifest_ids) or not result.citations
        if result.status == "answered":
            answered += 1
            if all(c in manifest_ids for c in result.citations):
                all_citations_valid += 1
        if not q.answerable:
            unanswerable += 1
            if result.status == "abstained":
                correct_abstentions += 1
        per_question.append(entry)

    report: dict[str, Any] = {
        "k": k,
        "questions": len(dataset.questions),
        "labeled_questions": len(recalls),
        "recall_at_k": (sum(recalls) / len(recalls)) if recalls else None,
        "abstention": {
            "unanswerable": unanswerable,
            "abstained": correct_abstentions,
            "rate": _share(correct_abstentions, unanswerable),
        },
        "citations": {
            "answered": answered,
            "all_valid": all_citations_valid,
            "rate": _share(all_citations_valid, answered),
        },
        "latency_ms": {
            "retrieval_ms": {"p50": _percentile(retrieval_ms, 50), "p95": _percentile(retrieval_ms, 95)},
            "answer_ms": {"p50": _percentile(model_ms, 50), "p95": _percentile(model_ms, 95)},
        },
        "per_question": per_question,
    }
    return report


def format_report(report: dict[str, Any]) -> str:
    """A human-readable one-screen summary (the JSON is the full report)."""
    recall = report["recall_at_k"]
    lines = [
        f"evaluation: {report['questions']} questions (k={report['k']}, "
        f"labeled {report['labeled_questions']})",
        f"  recall@{report['k']}: "
        + (f"{recall:.3f}" if recall is not None else "n/a (no labels)"),
    ]
    a = report["abstention"]
    lines.append(
        "  abstention on unanswerable: "
        + (
            f"{a['abstained']}/{a['unanswerable']} ({a['rate']:.3f})"
            if a["rate"] is not None
            else "n/a (none marked unanswerable)"
        )
    )
    c = report["citations"]
    lines.append(
        "  citations valid on answered: "
        + (f"{c['all_valid']}/{c['answered']} ({c['rate']:.3f})" if c["rate"] is not None else "n/a")
    )
    lat = report["latency_ms"]
    lines.append(
        f"  retrieval p50/p95 ms: {_ms(lat['retrieval_ms']['p50'])} / {_ms(lat['retrieval_ms']['p95'])}"
    )
    lines.append(
        f"  answer    p50/p95 ms: {_ms(lat['answer_ms']['p50'])} / {_ms(lat['answer_ms']['p95'])}"
    )
    for q in report["per_question"]:
        recall = f" recall={q['recall']:.2f}" if q.get("recall") is not None else ""
        lines.append(f"[{q['status']:>9}] {q['id']}{recall}")
    return "\n".join(lines)


def _ms(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "n/a"
