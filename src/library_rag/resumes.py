"""Extended per-book summaries ("resumes") and their keyword search.

After a revision is published, the ``resume`` job stage samples the book's
text (head, evenly-spaced middle passages, tail of the chunked run) and asks
the *answer* model — the only LLM on the box — for a 700-1000 word English
summary, using the generation-specific parameters in ``cfg.resume`` (longer
and looser than cited answering). The stored summary lives in
``book_resumes`` with a mirror row in the standalone ``resumes_fts`` FTS5
table, so keyword search is bm25-ranked in SQLite without touching Qdrant.
"""

from __future__ import annotations

import re
import time

from .config import Config
from .db import Database
from .identity import make_task_key
from .jobs import Jobs
from .llm import (
    AnswerModel,
    FakeAnswerModel,
    LlamaCppAnswerModel,
    Message,
)
from .scan import STAGE_RESUME

__all__ = [
    "MIN_RESUME_WORDS",
    "build_resume_messages",
    "enqueue_missing_resumes",
    "get_resume",
    "make_resume_model",
    "resume_input_version",
    "resume_task_key",
    "sample_resume_input",
    "search_resumes",
    "store_resume",
]

# A resume shorter than this is a truncated generation, not a summary; the
# job fails permanently so the operator bumps resume.max_tokens and retries.
MIN_RESUME_WORDS = 100

# Job states that still count as "work in flight" for the enqueue idempotency
# check (mirrors jobs._NOT_TERMINAL).
_OPEN_JOB_STATES = "('pending', 'running', 'retryable_failed')"

_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def make_resume_model(cfg: Config) -> AnswerModel | None:
    """Build the answer model with resume-generation parameters.

    Mirrors :func:`library_rag.llm.make_answer_model` — the model identity
    (host, port, name, revision) always comes from ``cfg.answer`` — but the
    generation parameters come from ``cfg.resume``: a resume is a longer,
    looser generation than a cited answer, and
    :class:`LlamaCppAnswerModel` fixes those at construction. Returns
    ``None`` when no answer model is configured; the caller then fails the
    job per-job (``answer_model_not_configured``), not at config load.
    """
    a = cfg.answer
    if not a.is_configured:
        return None
    if a.fake:
        return FakeAnswerModel()
    assert a.model_revision is not None
    r = cfg.resume
    return LlamaCppAnswerModel(
        host=cfg.services.answer_host,
        port=cfg.services.answer_port,
        model_name=a.model_name or a.model_revision,
        model_revision=a.model_revision,
        timeout_seconds=r.timeout_seconds,
        max_tokens=r.max_tokens,
        temperature=r.temperature,
    )


def resume_input_version(cfg: Config, run_id: str) -> str:
    """The job ``input_version`` for a resume: prompt contract + source run.

    Both parts change the generated text, so both belong in the idempotency
    key: re-chunking (a new run) or a prompt-version bump mints a fresh job
    without touching the stored resume until the new one succeeds.
    """
    return f"{cfg.resume.prompt_version}:{run_id}"


def resume_task_key(cfg: Config, rev_id: str, run_id: str) -> str:
    return make_task_key(STAGE_RESUME, rev_id, resume_input_version(cfg, run_id))


def _even_indices(n: int, max_points: int = 16) -> list[int]:
    """Positions strictly between the first and last of *n*, evenly spaced."""
    count = min(max_points, (n - 2) // 2)
    if count <= 0:
        return []
    return [1 + round((n - 1) * (i + 1) / (count + 1)) for i in range(count)]


def sample_resume_input(db: Database, run_id: str, *, char_budget: int) -> str:
    """Sample a book's text for resume generation, bounded to *char_budget*.

    Takes the opening chunk, up to 16 evenly-spaced middle chunks, and the
    ending chunk (from the end), joined in reading order so the model sees
    the shape of the book, not an arbitrary window.
    """
    rows = db.query(
        "SELECT text FROM chunks WHERE run_id = ? ORDER BY position", (run_id,)
    )
    if not rows:
        return ""
    texts = [str(r["text"]) for r in rows]
    end_budget = char_budget // 4
    head = texts[0][:end_budget]
    parts = [head]
    if len(texts) > 2:
        mid_budget = char_budget - len(head) - end_budget
        indices = _even_indices(len(texts))
        per = max(128, mid_budget // len(indices))
        for i in indices:
            parts.append(texts[i][:per])
    if len(texts) > 1:
        parts.append(texts[-1][-end_budget:])
    return "\n\n".join(parts)[:char_budget]


def build_resume_messages(title: str, sample: str) -> list[Message]:
    """The resume prompt contract (version ``cfg.resume.prompt_version``)."""
    system = (
        "You write extended, factual summaries of books for a personal "
        "research library. You summarize strictly from the supplied text and "
        "never invent content that is not in it."
    )
    user = (
        f"Write an extended English summary (700-1000 words) of the book "
        f'titled "{title}". The text below is a sample of the book: its '
        "opening, evenly-spaced middle passages, and its ending. Cover what "
        "the book is about, its structure and contents, its main arguments, "
        "plot, or themes, and why it matters. Use plain prose without "
        "headings or lists, and do not mention that you worked from a "
        f"sample.\n\n---\n\n{sample}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def store_resume(
    db: Database,
    *,
    rev_id: str,
    doc_id: str,
    run_id: str,
    title: str | None,
    text: str,
    model_revision: str,
    prompt_version: str,
    now: float | None = None,
) -> int:
    """Upsert the stored resume and its FTS mirror in one transaction.

    Returns the word count recorded. The FTS row is deleted and re-inserted
    (standalone table) so regeneration never leaves a stale index row.
    """
    ts = time.time() if now is None else now
    word_count = len(text.split())
    with db.transaction():
        db.execute("DELETE FROM resumes_fts WHERE rev_id = ?", (rev_id,))
        db.execute(
            """
            INSERT INTO book_resumes (rev_id, doc_id, run_id, title, text, word_count,
                                      model_revision, prompt_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rev_id) DO UPDATE SET
                doc_id = excluded.doc_id,
                run_id = excluded.run_id,
                title = excluded.title,
                text = excluded.text,
                word_count = excluded.word_count,
                model_revision = excluded.model_revision,
                prompt_version = excluded.prompt_version,
                updated_at = excluded.updated_at
            """,
            (
                rev_id,
                doc_id,
                run_id,
                title,
                text,
                word_count,
                model_revision,
                prompt_version,
                ts,
                ts,
            ),
        )
        db.execute(
            "INSERT INTO resumes_fts (rev_id, text) VALUES (?, ?)", (rev_id, text)
        )
    return word_count


def enqueue_missing_resumes(db: Database, cfg: Config) -> int:
    """Enqueue resume jobs for published revisions without a stored resume.

    The backfill / gap-closer: every *active* revision with an *active*
    publication and no ``book_resumes`` row gets one job, keyed on the latest
    succeeded extraction run. Idempotent — a rev with a stored resume or an
    open resume job is never re-enqueued. Gated on ``cfg.resume.enabled``
    and a configured answer model: without one there is nothing to generate
    with, so nothing is enqueued (matching how reconcile skips embedding
    work when no embedder is configured).
    """
    if not cfg.resume.enabled or not cfg.answer.is_configured:
        return 0
    jobs = Jobs(db)
    enqueued = 0
    rows = db.query(
        """
        SELECT r.rev_id AS rev_id
        FROM source_revisions r
        JOIN publications p ON p.rev_id = r.rev_id AND p.state = 'active'
        WHERE r.is_active = 1
          AND NOT EXISTS (SELECT 1 FROM book_resumes b WHERE b.rev_id = r.rev_id)
        """
    )
    for row in rows:
        rev_id = str(row["rev_id"])
        run = db.query_one(
            "SELECT run_id FROM extraction_runs WHERE rev_id = ? AND state = 'succeeded' "
            "ORDER BY created_at DESC",
            (rev_id,),
        )
        if run is None:
            continue  # published but no settled run; publish would defer anyway
        task_key = resume_task_key(cfg, rev_id, str(run["run_id"]))
        if db.query_one(
            "SELECT 1 AS x FROM jobs WHERE task_key = ? AND state IN "
            + _OPEN_JOB_STATES,
            (task_key,),
        ):
            continue
        jobs.enqueue(task_key, STAGE_RESUME, input_id=rev_id,
                     input_version=resume_input_version(cfg, str(run["run_id"])))
        enqueued += 1
    return enqueued


def get_resume(db: Database, rev_id: str) -> dict[str, object] | None:
    """The stored resume for *rev_id*, or None."""
    row = db.query_one("SELECT * FROM book_resumes WHERE rev_id = ?", (rev_id,))
    if row is None:
        return None
    return {
        "rev_id": row["rev_id"],
        "doc_id": row["doc_id"],
        "title": row["title"],
        "text": row["text"],
        "word_count": int(row["word_count"]),
        "model_revision": row["model_revision"],
        "prompt_version": row["prompt_version"],
        "updated_at": float(row["updated_at"]),
    }


def _fts_match(query: str) -> str:
    """An OR-of-quoted-terms FTS5 MATCH expression for a keyword query.

    ``\\w`` tokens never contain a double quote, so the quoting is safe.
    Deduplicated in first-seen order; empty when the query has no tokens.
    """
    tokens = _TOKEN.findall(query)
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))


def search_resumes(db: Database, query: str, limit: int = 20) -> list[dict[str, object]]:
    """bm25-ranked keyword search over stored resumes (lower score = better).

    Returns result dicts with ``rev_id``, ``doc_id``, ``title``, ``score``
    and a ~280-char ``excerpt`` of the resume text.
    """
    match = _fts_match(query)
    if not match:
        return []
    rows = db.query(
        """
        SELECT b.rev_id AS rev_id, b.doc_id AS doc_id, b.title AS title,
               b.text AS text, bm25(resumes_fts) AS score
        FROM resumes_fts f
        JOIN book_resumes b ON b.rev_id = f.rev_id
        WHERE resumes_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (match, limit),
    )
    return [
        {
            "rev_id": r["rev_id"],
            "doc_id": r["doc_id"],
            "title": r["title"] or r["rev_id"],
            "score": float(r["score"]),
            "excerpt": str(r["text"])[:280],
        }
        for r in rows
    ]
