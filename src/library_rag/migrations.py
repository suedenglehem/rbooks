"""Schema migrations.

The schema is introduced as a sequence of numbered migrations, each applied
exactly once and recorded in ``schema_migrations``. M1 introduces the catalog
(documents, source_revisions, path_aliases), the durable job table (with
lease/fencing columns), and a small ``meta`` table (used for the pause flag).

Later milestones add the pipeline tables (extraction_runs, source_units,
chunks, embedding_batches, index_generations, publications) as further
migrations — the point of the runner is that the schema *evolves*, so M1 stays
a minimal, independently-testable slice.
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


MIGRATIONS: list[Migration] = [
    Migration(1, "catalog_and_jobs", _MIGRATION_0001),
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
