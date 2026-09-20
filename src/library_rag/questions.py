"""Pilot question bank: the operator's manually labeled evaluation questions.

PRD §12 asks for 100-200 human-labeled questions across six categories
(exact terms, conceptual, OCR-heavy, EPUB, cross-book conflicts, and
unanswerable). This module is the *annotation tooling*; the labels themselves
are the operator's work. The hard rule: nothing in here invents a label.

* :func:`add_question` is the only write path, and it records only what the
  operator supplied (plus provenance: who, when, which books);
* :func:`suggest_candidates` emits machine-generated *candidates* — raw text
  from the index that the operator may turn into questions. Candidates are
  clearly marked unlabeled and never enter the bank unreviewed;
* :func:`export_dataset` projects the bank onto the ``evaluate`` dataset
  schema (``id``/``question``/``expected_chunks``/``answerable``), so the
  labeled bank is exactly what ``library-rag evaluate`` consumes.

The bank is a JSON file::

    {"schema": "pilot-questions/1", "questions": [
        {"id": "q001", "question": "...", "category": "exact_term",
         "expected_chunks": ["<chunk_id>"], "answerable": true,
         "source_paths": ["/mnt/.../book.pdf"], "notes": "",
         "labeled_by": "operator", "labeled_at": 1234.5}]}
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import Database

__all__ = [
    "QUESTION_BANK_SCHEMA",
    "QUESTION_CATEGORIES",
    "QuestionEntry",
    "add_question",
    "export_dataset",
    "load_bank",
    "next_question_id",
    "save_bank",
    "suggest_candidates",
]

QUESTION_BANK_SCHEMA = "pilot-questions/1"

#: The six PRD categories. ``unanswerable`` is consistent with the
#: ``answerable: false`` flag the evaluate harness uses for abstention.
QUESTION_CATEGORIES = ("exact_term", "conceptual", "ocr", "epub", "conflict", "unanswerable")


@dataclass(frozen=True)
class QuestionEntry:
    """One bank row: an operator-labeled question with provenance."""

    id: str
    question: str
    category: str
    expected_chunks: tuple[str, ...] = ()
    answerable: bool = True
    source_paths: tuple[str, ...] = ()
    notes: str = ""
    labeled_by: str = ""
    labeled_at: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "expected_chunks": list(self.expected_chunks),
            "answerable": self.answerable,
            "source_paths": list(self.source_paths),
            "notes": self.notes,
            "labeled_by": self.labeled_by,
            "labeled_at": self.labeled_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuestionEntry:
        def _str_list(key: str) -> tuple[str, ...]:
            raw = d.get(key, [])
            if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
                raise ValueError(f"question {d.get('id')!r}: {key} must be a list of strings")
            return tuple(raw)

        category = d.get("category")
        if not isinstance(category, str) or category not in QUESTION_CATEGORIES:
            raise ValueError(f"question {d.get('id')!r}: unknown category {category!r}")
        answerable = d.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"question {d.get('id')!r}: answerable must be a boolean")
        if category == "unanswerable" and answerable:
            raise ValueError(f"question {d.get('id')!r}: unanswerable requires answerable=false")
        if answerable is False and category != "unanswerable":
            raise ValueError(f"question {d.get('id')!r}: answerable=false requires category=unanswerable")
        labeled_at = d.get("labeled_at")
        return cls(
            id=str(d["id"]),
            question=str(d["question"]),
            category=category,
            expected_chunks=_str_list("expected_chunks"),
            answerable=answerable,
            source_paths=_str_list("source_paths"),
            notes=str(d.get("notes", "")),
            labeled_by=str(d.get("labeled_by", "")),
            labeled_at=float(labeled_at) if isinstance(labeled_at, (int, float)) else None,
        )


def load_bank(path: Path | str) -> list[QuestionEntry]:
    """Load a question bank file (an absent file is an empty bank)."""
    p = Path(path)
    if not p.is_file():
        return []
    raw: Any = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != QUESTION_BANK_SCHEMA:
        raise ValueError(f"{p}: schema must be {QUESTION_BANK_SCHEMA!r}")
    rows = raw.get("questions")
    if not isinstance(rows, list):
        raise ValueError(f"{p}: 'questions' must be a list")
    seen: set[str] = set()
    out: list[QuestionEntry] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"{p}: every question needs a non-empty 'id'")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"{p}: question {row['id']!r} needs a non-empty 'question'")
        if row["id"] in seen:
            raise ValueError(f"{p}: duplicate question id {row['id']!r}")
        seen.add(row["id"])
        out.append(QuestionEntry.from_dict(row))
    return out


def save_bank(path: Path | str, entries: list[QuestionEntry]) -> None:
    """Atomically write the bank (sorted by id for stable diffs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema": QUESTION_BANK_SCHEMA,
        "questions": [e.to_dict() for e in sorted(entries, key=lambda e: e.id)],
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def next_question_id(entries: list[QuestionEntry]) -> str:
    """The next zero-padded id (``q001`` style) after the existing ones."""
    highest = 0
    for e in entries:
        if e.id.startswith("q") and e.id[1:].isdigit():
            highest = max(highest, int(e.id[1:]))
    return f"q{highest + 1:03d}"


def add_question(
    path: Path | str,
    entry: QuestionEntry,
) -> QuestionEntry:
    """Append *entry* to the bank (rejecting duplicate ids); return it.

    If ``entry.id`` is empty, :func:`next_question_id` fills it in.
    """
    entries = load_bank(path)
    qid = entry.id or next_question_id(entries)
    if any(e.id == qid for e in entries):
        raise ValueError(f"question id already in the bank: {qid!r}")
    if qid != entry.id:
        entry = QuestionEntry(
            id=qid,
            question=entry.question,
            category=entry.category,
            expected_chunks=entry.expected_chunks,
            answerable=entry.answerable,
            source_paths=entry.source_paths,
            notes=entry.notes,
            labeled_by=entry.labeled_by,
            labeled_at=entry.labeled_at,
        )
    entries.append(entry)
    save_bank(path, entries)
    return entry


def export_dataset(entries: list[QuestionEntry]) -> dict[str, object]:
    """Project the bank onto the evaluate-harness dataset schema."""
    return {
        "questions": [
            {
                "id": e.id,
                "question": e.question,
                **({"expected_chunks": list(e.expected_chunks)} if e.expected_chunks else {}),
                "answerable": e.answerable,
            }
            for e in sorted(entries, key=lambda e: e.id)
        ]
    }


def suggest_candidates(db: Database, *, limit: int = 40, seed: int = 42) -> list[dict[str, Any]]:
    """Machine-generated, *unlabeled* question candidates from the index.

    For a deterministic, seed-shuffled sample of chunks: one ``exact_term``
    candidate (the rarest distinctive token in the chunk) and one
    ``conceptual`` candidate (the longest sentence). Each candidate carries
    the book path and chunk id so the operator can verify before running
    ``pilot annotate add``. Candidates are status-marked ``"candidate"`` —
    they are prompts, never labels.
    """
    rows = db.query(
        """
        SELECT c.chunk_id, c.text, r.first_path
        FROM chunks c
        JOIN source_revisions r ON r.rev_id = c.rev_id
        """
    )

    def _rank(cid: str) -> str:
        return hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest()

    sample = sorted(rows, key=lambda r: (_rank(str(r["chunk_id"])), str(r["chunk_id"])))[: limit]
    out: list[dict[str, Any]] = []
    for row in sample:
        text = str(row["text"])
        book = str(row["first_path"] or "")
        tokens = [t for t in text.lower().split() if len(t) > 5 and t.isalpha()]
        term = max(tokens, key=len) if tokens else None
        if term is not None:
            out.append(
                {
                    "status": "candidate",
                    "suggested_category": "exact_term",
                    "term": term,
                    "book": book,
                    "chunk_id": str(row["chunk_id"]),
                    "hint": f"e.g. What does the text say about '{term}'?",
                }
            )
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.strip()) >= 40]
        if sentences:
            sent = max(sentences, key=len)
            if len(sent) > 200:
                sent = sent[:200].rsplit(" ", 1)[0] + "..."
            out.append(
                {
                    "status": "candidate",
                    "suggested_category": "conceptual",
                    "sentence": sent,
                    "book": book,
                    "chunk_id": str(row["chunk_id"]),
                }
            )
    return out
