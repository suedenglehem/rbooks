"""Migration runner: idempotent, ordered, durable settings."""

from __future__ import annotations

import sqlite3

import pytest

from library_rag.config import Config
from library_rag.db import Database, db_path_for
from library_rag.migrations import MIGRATIONS, current_version, migrate


def test_migrate_is_idempotent(base_config: Config) -> None:
    path = db_path_for(base_config.paths.state_root)
    db = Database.connect(path)
    try:
        expected = len(MIGRATIONS)  # latest schema version (2 as of M2)
        assert migrate(db) == expected
        # Running again must not error and must report the same version.
        assert migrate(db) == expected
        assert current_version(db) == expected
    finally:
        db.close()


def test_all_expected_tables_exist(state_db: Database) -> None:
    rows = state_db.query(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = {r["name"] for r in rows}
    expected = (
        "schema_migrations",
        "documents",
        "source_revisions",
        "path_aliases",
        "jobs",
        "scan_state",
        "extraction_runs",
        "source_units",
    )
    for name in expected:
        assert name in tables


def test_migrations_recorded_in_order(state_db: Database) -> None:
    rows = state_db.query("SELECT version, name FROM schema_migrations ORDER BY version")
    assert [r["version"] for r in rows] == [m.version for m in MIGRATIONS]
    assert [r["name"] for r in rows] == [m.name for m in MIGRATIONS]


def test_durability_pragmas_active(base_config: Config) -> None:
    db = Database.connect(db_path_for(base_config.paths.state_root))
    try:
        journal = db.query_one("PRAGMA journal_mode;")
        fk = db.query_one("PRAGMA foreign_keys;")
        sync = db.query_one("PRAGMA synchronous;")
        assert journal is not None and fk is not None and sync is not None
        assert str(journal["journal_mode"]).lower() == "wal"
        assert int(fk["foreign_keys"]) == 1
        # synchronous=FULL -> 2 (FULL); NORMAL is 1, OFF is 0.
        assert int(sync["synchronous"]) == 2
    finally:
        db.close()


def test_foreign_keys_enforced(state_db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError), state_db.transaction():
        state_db.execute(
            "INSERT INTO path_aliases (path, doc_id, rev_id, created_at, updated_at) "
            "VALUES ('/x', 'no-such-doc', 'no-such-rev', 0, 0)"
        )
