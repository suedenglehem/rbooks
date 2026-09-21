"""Schema migrations.

The schema is introduced as a sequence of numbered migrations, each applied
exactly once and recorded in ``schema_migrations``. M1 introduces the catalog
(documents, source_revisions, path_aliases), the durable job table (with
lease/fencing columns), and a small ``meta`` table (used for the pause flag).

M2 adds the pipeline slice it needs: ``scan_state`` (scan fast-check cache),
``extraction_runs`` and ``source_units``; M3 adds ``chunks``; M4 adds
``embedding_batches``, ``index_generations``, ``sparse_corpus_stats`` and
``publications``; M5 adds ``answers`` (cited answers with frozen evidence
manifests). The point of the runner is that the schema *evolves*, so each
milestone stays a minimal, independently-testable slice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .db import Database

__all__ = ["MIGRATIONS", "current_version", "migrate"]

_SCHEMA_MIGRATIONS = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    applied_at REAL NOT NULL
);
"""

# One DDL statement per entry; the runner applies them in order inside a single
# transaction (``Database.execute`` accepts only one statement at a time).
_MIGRATION_0001: tuple[str, ...] = (
    """
    CREATE TABLE documents (
        doc_id        TEXT PRIMARY KEY,
        anchor_sha256 TEXT NOT NULL,
        created_at    REAL NOT NULL,
        updated_at    REAL NOT NULL
    )
    """,
    """
    CREATE TABLE source_revisions (
        rev_id          TEXT PRIMARY KEY,
        doc_id          TEXT NOT NULL REFERENCES documents(doc_id),
        sha256          TEXT NOT NULL,
        size_bytes      INTEGER NOT NULL,
        format          TEXT NOT NULL CHECK (format IN ('pdf', 'epub')),
        archive_relpath TEXT NOT NULL,
        first_path      TEXT NOT NULL,
        is_active       INTEGER NOT NULL DEFAULT 0,
        created_at      REAL NOT NULL,
        UNIQUE (doc_id, sha256)
    )
    """,
    "CREATE INDEX idx_revisions_doc ON source_revisions(doc_id)",
    "CREATE INDEX idx_revisions_sha ON source_revisions(sha256)",
    """
    CREATE TABLE path_aliases (
        path       TEXT PRIMARY KEY,
        doc_id     TEXT NOT NULL REFERENCES documents(doc_id),
        rev_id     TEXT NOT NULL REFERENCES source_revisions(rev_id),
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    "CREATE INDEX idx_aliases_doc ON path_aliases(doc_id)",
    """
    CREATE TABLE jobs (
        job_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        task_key          TEXT NOT NULL UNIQUE,
        stage             TEXT NOT NULL,
        input_id          TEXT,
        input_version     TEXT,
        range_spec        TEXT,
        state             TEXT NOT NULL CHECK (state IN (
                            'pending', 'running', 'succeeded',
                            'retryable_failed', 'permanent_failed', 'cancelled')),
        attempts          INTEGER NOT NULL DEFAULT 0,
        max_attempts      INTEGER NOT NULL DEFAULT 3,
        lease_owner       TEXT,
        lease_token       TEXT,
        lease_expires_at  REAL,
        last_heartbeat_at REAL,
        next_attempt_at   REAL,
        error_category    TEXT,
        error_detail      TEXT,
        output_manifest   TEXT,
        created_at        REAL NOT NULL,
        updated_at        REAL NOT NULL
    )
    """,
    "CREATE INDEX idx_jobs_state ON jobs(state)",
    """
    CREATE TABLE meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
)


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


_MIGRATION_0002: tuple[str, ...] = (
    # Scan fast-check cache (PRD §8A): size/mtime recorded so a rescan does not
    # re-hash unchanged files.
    """
    CREATE TABLE scan_state (
        path         TEXT PRIMARY KEY,
        size_bytes   INTEGER NOT NULL,
        mtime        REAL NOT NULL,
        sha256       TEXT,
        format       TEXT,
        rev_id       TEXT,
        last_seen_at REAL NOT NULL
    )
    """,
    # One row per (revision, parser, settings) extraction attempt.
    """
    CREATE TABLE extraction_runs (
        run_id         TEXT PRIMARY KEY,
        rev_id         TEXT NOT NULL REFERENCES source_revisions(rev_id),
        doc_id         TEXT NOT NULL REFERENCES documents(doc_id),
        parser_version TEXT NOT NULL,
        settings_sha   TEXT NOT NULL,
        unit_count     INTEGER,
        state          TEXT NOT NULL CHECK (state IN ('running', 'succeeded', 'failed')),
        error_category TEXT,
        error_detail   TEXT,
        created_at     REAL NOT NULL,
        updated_at     REAL NOT NULL,
        UNIQUE (rev_id, parser_version, settings_sha)
    )
    """,
    "CREATE INDEX idx_runs_rev ON extraction_runs(rev_id)",
    # Source units: one per physical PDF page or EPUB spine item (PRD §6).
    # Text/geometry live in the compressed JSON artifact; the row carries the
    # metadata and the artifact pointer so citations can be resolved without
    # reading the archive.
    """
    CREATE TABLE source_units (
        unit_id          TEXT PRIMARY KEY,
        run_id           TEXT NOT NULL REFERENCES extraction_runs(run_id),
        rev_id           TEXT NOT NULL REFERENCES source_revisions(rev_id),
        kind             TEXT NOT NULL CHECK (kind IN ('page', 'section')),
        position         INTEGER NOT NULL,
        ref              TEXT,
        char_count       INTEGER,
        rotation         INTEGER,
        width            REAL,
        height           REAL,
        quality_flags    TEXT,
        artifact_relpath TEXT NOT NULL,
        artifact_sha256  TEXT NOT NULL,
        created_at       REAL NOT NULL,
        UNIQUE (run_id, kind, position)
    )
    """,
    "CREATE INDEX idx_units_rev ON source_units(rev_id)",
    "CREATE INDEX idx_units_run ON source_units(run_id)",
)


# M3: selective OCR (per-page state on the unit row) and chunks (PRD §8C/§8E).
_MIGRATION_0003: tuple[str, ...] = (
    # Which OCR decision was made for a page unit at extraction time.
    # 'ocr' pages get an ocr job; 'reuse'/'skip' never do. Null for sections.
    "ALTER TABLE source_units ADD COLUMN route TEXT",
    # OCR lifecycle of a unit: 'none' (not applicable / not started), 'pending'
    # (job enqueued), 'done', 'failed'. Tracked on the row so a crashed resume
    # can tell what still has to happen without reading artifacts.
    "ALTER TABLE source_units ADD COLUMN ocr_state TEXT NOT NULL DEFAULT 'none'",
    # Pipeline fingerprint (units + normalization + chunker settings) of the
    # last completed chunk pass. Equality with the freshly computed value is
    # the no-op test for chunk jobs.
    "ALTER TABLE extraction_runs ADD COLUMN chunk_fingerprint TEXT",
    """
    CREATE TABLE chunks (
        chunk_id      TEXT PRIMARY KEY,
        run_id        TEXT NOT NULL REFERENCES extraction_runs(run_id),
        rev_id        TEXT NOT NULL REFERENCES source_revisions(rev_id),
        position      INTEGER NOT NULL,
        text          TEXT NOT NULL,
        token_count   INTEGER NOT NULL,
        title         TEXT,
        spans         TEXT NOT NULL,
        prev_chunk_id TEXT,
        next_chunk_id TEXT,
        created_at    REAL NOT NULL,
        UNIQUE (run_id, position)
    )
    """,
    "CREATE INDEX idx_chunks_run ON chunks(run_id)",
    "CREATE INDEX idx_chunks_rev ON chunks(rev_id)",
)


# M4: embeddings, index generations, and publications (PRD §6/§8F).
_MIGRATION_0004: tuple[str, ...] = (
    # Checkpointed dense-vector batches: non-pickle numeric file + chunk manifest
    # (PRD §5). Keyed on (run, encoding, batch index) so a replay reuses the
    # file instead of re-embedding, and a model/encoding change mints new keys.
    """
    CREATE TABLE embedding_batches (
        batch_id         TEXT PRIMARY KEY,
        run_id           TEXT NOT NULL REFERENCES extraction_runs(run_id),
        model_revision   TEXT NOT NULL,
        embedding_sha    TEXT NOT NULL,
        batch_index      INTEGER NOT NULL,
        chunk_ids        TEXT NOT NULL,
        vector_sha256    TEXT NOT NULL,
        artifact_relpath TEXT NOT NULL,
        created_at       REAL NOT NULL,
        UNIQUE (run_id, embedding_sha, batch_index)
    )
    """,
    "CREATE INDEX idx_emb_batches_run ON embedding_batches(run_id)",
    # One row per (run, encoding, corpus-statistics epoch) point set. Carries the
    # persisted model/dtype/dimension/normalization settings (PRD §8F).
    """
    CREATE TABLE index_generations (
        gen_id           TEXT PRIMARY KEY,
        run_id           TEXT NOT NULL REFERENCES extraction_runs(run_id),
        rev_id           TEXT NOT NULL REFERENCES source_revisions(rev_id),
        model_revision   TEXT NOT NULL,
        embedding_sha    TEXT NOT NULL,
        sparse_stats_sha TEXT NOT NULL,
        dimensions       INTEGER NOT NULL,
        dtype            TEXT NOT NULL,
        normalized       INTEGER NOT NULL,
        point_count      INTEGER,
        state            TEXT NOT NULL CHECK (state IN ('ready', 'abandoned')),
        created_at       REAL NOT NULL,
        updated_at       REAL NOT NULL,
        UNIQUE (run_id, embedding_sha, sparse_stats_sha)
    )
    """,
    "CREATE INDEX idx_gen_rev ON index_generations(rev_id)",
    # Corpus-wide BM25 statistics, one row per statistics epoch (PRD §8F IDF).
    # The query path loads the stats of the generation epoch it searches; the
    # canonical df map is stored so any process can re-encode without state.
    """
    CREATE TABLE sparse_corpus_stats (
        stats_sha   TEXT PRIMARY KEY,
        doc_count   INTEGER NOT NULL,
        avg_doc_len REAL NOT NULL,
        df_json     TEXT NOT NULL,
        created_at  REAL NOT NULL
    )
    """,
    # A publication makes one generation the visible evidence for its revision.
    # At most one row per rev is ever 'active' (enforced in the switch
    # transaction); 'staged' means points are upserted inactive and verified,
    # awaiting the activate/reconcile step. The SQLite row, not the Qdrant
    # flags, is the source of truth (PRD §8F).
    """
    CREATE TABLE publications (
        pub_id          TEXT PRIMARY KEY,
        rev_id          TEXT NOT NULL REFERENCES source_revisions(rev_id),
        doc_id          TEXT NOT NULL REFERENCES documents(doc_id),
        gen_id          TEXT NOT NULL REFERENCES index_generations(gen_id),
        run_id          TEXT NOT NULL,
        expected_points INTEGER NOT NULL,
        state           TEXT NOT NULL CHECK (state IN ('staged', 'active', 'superseded')),
        created_at      REAL NOT NULL,
        activated_at    REAL,
        UNIQUE (rev_id, gen_id)
    )
    """,
    "CREATE INDEX idx_pubs_rev ON publications(rev_id)",
    "CREATE INDEX idx_pubs_doc ON publications(doc_id)",
    "CREATE INDEX idx_pubs_state ON publications(state)",
)


# M5 adds the answers table. The load-bearing column is ``evidence_manifest``:
# a frozen snapshot (schema 1) of every cited passage — text, unit locations,
# page/section geometry, quality flags — captured at answer time. Citations
# resolve from this snapshot, so re-chunking or re-publishing the library
# cannot change what a saved answer points at (PRD §12).
_MIGRATION_0005: tuple[str, ...] = (
    """
    CREATE TABLE answers (
        answer_id         TEXT PRIMARY KEY,
        created_at        REAL NOT NULL,
        query             TEXT NOT NULL,
        doc_id            TEXT,
        rev_id            TEXT,
        status            TEXT NOT NULL CHECK (status IN ('answered', 'abstained', 'failed')),
        model_revision    TEXT,
        prompt_version    TEXT NOT NULL,
        evidence_manifest TEXT NOT NULL,
        answer_text       TEXT,
        abstain_reason    TEXT,
        failure_reason    TEXT,
        citations         TEXT NOT NULL,
        search_counts     TEXT,
        retrieval_ms      REAL,
        model_ms          REAL
    )
    """,
    "CREATE INDEX idx_answers_created ON answers(created_at DESC)",
    "CREATE INDEX idx_answers_status ON answers(status)",
)


# M7 slice 7: when a publication was superseded. The B4 switch that marks a
# row ``superseded`` is atomic (the replacement is active in the same
# transaction), so the state alone is definitive; the timestamp exists so
# ``gc`` can apply its grace window to superseded index points and the
# coverage report can show how long a superseded generation has lingered.
# Rows superseded before this column existed keep NULL and are left to a
# later supersede cycle (conservative: never collected on unknown age).
_MIGRATION_0006: tuple[str, ...] = (
    "ALTER TABLE publications ADD COLUMN superseded_at REAL",
)


# M7 slice 9: the software version that enqueued each job. NULL
# marks jobs created before version signatures existed ("legacy"): they were
# enqueued by some code version we never recorded, and the worker-start gate
# treats them conservatively as foreign work requiring explicit confirmation.
_MIGRATION_0007: tuple[str, ...] = (
    "ALTER TABLE jobs ADD COLUMN created_by_version TEXT",
)


MIGRATIONS: list[Migration] = [
    Migration(1, "catalog_and_jobs", _MIGRATION_0001),
    Migration(2, "pipeline_tables", _MIGRATION_0002),
    Migration(3, "ocr_and_chunks", _MIGRATION_0003),
    Migration(4, "embeddings_and_publication", _MIGRATION_0004),
    Migration(5, "answers", _MIGRATION_0005),
    Migration(6, "publications_superseded_at", _MIGRATION_0006),
    Migration(7, "jobs_created_by_version", _MIGRATION_0007),
]


def current_version(db: Database) -> int:
    """The highest applied migration version (0 if none)."""
    row = db.query_one("SELECT MAX(version) AS v FROM schema_migrations")
    return int(row["v"] or 0) if row is not None else 0


def migrate(db: Database, now: float | None = None) -> int:
    """Apply any pending migrations in order; return the resulting version.

    Each migration runs in its own short transaction, so a crash mid-migration
    leaves the schema at the last fully-applied version and re-runs cleanly.
    """
    ts = time.time() if now is None else now
    db.execute(_SCHEMA_MIGRATIONS)
    applied = {int(r["version"]) for r in db.query("SELECT version FROM schema_migrations")}
    version = max(applied, default=0)
    for m in MIGRATIONS:
        if m.version in applied:
            continue
        with db.transaction():
            for statement in m.statements:
                db.execute(statement)
            db.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (m.version, m.name, ts),
            )
        version = m.version
    return version
