"""Command-line interface.

``doctor`` (M0) and the recovery-foundation commands (M1) are implemented:
``init``, ``pause``, ``resume``, ``status``, ``retry``, ``reconcile``. The
remaining commands are registered with their help text and milestone so that
``library-rag --help`` documents the whole intended surface (PRD §4). Each stub
prints an explicit "not yet implemented (milestone Mx)" message and returns a
non-zero exit code rather than pretending to do the work.

Destructive commands require an explicit ``--yes`` non-interactive flag; that
convention is implemented as each command lands (M1+).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

from . import __version__
from .catalog import source_file_count
from .config import Config, ConfigError, load_config
from .db import Database, db_path_for
from .doctor import render_json, render_text, run_doctor
from .identity import normalize_path
from .jobs import Jobs
from .log import setup_logging
from .migrations import current_version, migrate
from .reconcile import Mount, reconcile_catalog

# Exit codes.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_IMPLEMENTED = 2


# --- helpers ---------------------------------------------------------------
def _doctor(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    # Load an explicit config if given; otherwise run best-effort. When no usable
    # config exists, pass None so run_doctor falls back to its built-in default.
    try:
        cfg = load_config(args.config)
    except ConfigError:
        cfg = None
    report = run_doctor(cfg)
    if args.json:
        print(render_json(report))
    else:
        print(render_text(report))
    return EXIT_OK


def _require_config(args: argparse.Namespace) -> Config:
    """Load a usable config or exit non-zero (state commands need a state root)."""
    try:
        return load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_ERROR) from exc


def _open_state(cfg: Config) -> Database:
    """Open (and migrate, idempotently) the state database for *cfg*."""
    db = Database.connect(db_path_for(cfg.paths.state_root))
    migrate(db)
    return db


def _not_implemented(milestone: str) -> Callable[..., int]:
    def _run(_args: argparse.Namespace) -> int:
        print(f"command not yet implemented (planned: milestone {milestone})", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    return _run


def _count(db: Database, sql: str) -> int:
    row = db.query_one(sql)
    return int(row["n"]) if row is not None else 0


# --- M1 recovery commands --------------------------------------------------
def _init(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    cfg.paths.state_root.mkdir(parents=True, exist_ok=True)
    db = Database.connect(db_path_for(cfg.paths.state_root))
    version = migrate(db)
    db.close()
    print(f"initialized state database at {db_path_for(cfg.paths.state_root)} (schema v{version})")
    return EXIT_OK


def _pause(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    try:
        Jobs(db).pause(reason=args.reason)
    finally:
        db.close()
    print("paused: no new jobs will be claimed (active jobs may finish)")
    return EXIT_OK


def _resume(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    try:
        Jobs(db).resume()
    finally:
        db.close()
    print("resumed: job claiming re-enabled")
    return EXIT_OK


def _status(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    try:
        jobs = Jobs(db)
        job_counts = jobs.counts()
        data: dict[str, object] = {
            "schema_version": current_version(db),
            "paused": jobs.is_paused(),
            "jobs": job_counts,
            "documents": _count(db, "SELECT COUNT(*) AS n FROM documents"),
            "revisions": _count(db, "SELECT COUNT(*) AS n FROM source_revisions"),
            "source_files": source_file_count(db),
        }
    finally:
        db.close()
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"schema v{data['schema_version']}; paused={data['paused']}")
        print(f"documents={data['documents']} revisions={data['revisions']} "
              f"source_files={data['source_files']}")
        if job_counts:
            counts = " ".join(f"{k}={v}" for k, v in sorted(job_counts.items()))
            print(f"jobs: {counts}")
        else:
            print("jobs: (none)")
    return EXIT_OK


def _retry(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    try:
        requeued = Jobs(db).retry(include_permanent=args.include_permanent)
    finally:
        db.close()
    print(f"requeued {requeued} job(s)")
    return EXIT_OK


def _reconcile(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    try:
        # For each configured source root, reconcile its aliases. A root that
        # does not exist is treated as an unavailable mount and left untouched.
        mounts = [Mount(label=str(r), root=r, sentinel=None) for r in cfg.paths.source_roots]
        visible = set()
        for r in cfg.paths.source_roots:
            if r.exists():
                for p in r.rglob("*"):
                    if p.is_file():
                        visible.add(normalize_path(p))
        report = reconcile_catalog(db, mounts, visible)
    finally:
        db.close()
    if args.json:
        print(json.dumps(
            {
                "unavailable_mounts": report.unavailable_mounts,
                "pruned_aliases": report.pruned_aliases,
                "deleted_content": report.deleted_content,
            },
            indent=2,
            sort_keys=True,
        ))
    else:
        for msg in report.messages:
            print(msg)
        print(f"pruned {report.pruned_aliases} stale alias(es); "
              f"deleted content: {report.deleted_content}")
    return EXIT_OK


# --- parser ----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="library-rag",
        description="Local-first RAG research application for a personal book library.",
    )
    parser.add_argument("--version", action="version", version=f"library-rag {__version__}")
    parser.add_argument("--log-level", default="INFO", help="log level (DEBUG/INFO/WARNING)")
    parser.add_argument("--log-format", default="human", choices=["human", "json"])

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    doctor = sub.add_parser(
        "doctor", help="read-only environment diagnostics (distro, CPU, RAM, GPU, disks, OCR, services)"
    )
    doctor.add_argument("--config", help="path to config YAML")
    doctor.add_argument("--json", action="store_true", help="emit the report as JSON")
    doctor.set_defaults(_func=_doctor)

    init = sub.add_parser("init", help="create/verify the state directory and database (M1)")
    init.add_argument("--config", help="path to config YAML")
    init.set_defaults(_func=_init)

    pause = sub.add_parser("pause", help="persist pause; stop new job claims (M1)")
    pause.add_argument("--config", help="path to config YAML")
    pause.add_argument("--reason", default="", help="operator note for the pause")
    pause.set_defaults(_func=_pause)

    resume = sub.add_parser("resume", help="clear a persisted pause (M1)")
    resume.add_argument("--config", help="path to config YAML")
    resume.set_defaults(_func=_resume)

    status = sub.add_parser("status", help="show ingestion/pipeline status and counts (M1)")
    status.add_argument("--config", help="path to config YAML")
    status.add_argument("--json", action="store_true", help="emit status as JSON")
    status.set_defaults(_func=_status)

    retry = sub.add_parser("retry", help="requeue retryable-failed jobs (M1)")
    retry.add_argument("--config", help="path to config YAML")
    retry.add_argument(
        "--include-permanent",
        action="store_true",
        help="also requeue permanently-failed jobs (resets their attempt count)",
    )
    retry.set_defaults(_func=_retry)

    reconcile = sub.add_parser("reconcile", help="reconcile durable outputs and index/DB state (M1)")
    reconcile.add_argument("--config", help="path to config YAML")
    reconcile.add_argument("--json", action="store_true", help="emit the report as JSON")
    reconcile.set_defaults(_func=_reconcile)

    _register_stub(sub, "scan", "discover PDFs/EPUBs and register source revisions (M2)")
    _register_stub(sub, "ingest", "run the extraction->OCR->chunk->embed->publish pipeline (M2-M4)")
    _register_stub(sub, "search", "lexical + semantic search over the library (M4)")
    _register_stub(sub, "serve", "run the FastAPI research app + reader (M5)")
    _register_stub(sub, "evaluate", "run retrieval/answer evaluation on a labeled set (M6)")
    _register_stub(sub, "backup", "consistent backup of DB, Qdrant, and manifests (M7)")
    _register_stub(sub, "restore", "restore a backup into an isolated directory (M7)")
    _register_stub(sub, "verify", "verify source links and search after restore (M7)")
    return parser


def _register_stub(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], name: str, help: str
) -> None:
    milestone = help.rsplit("(", 1)[-1].rstrip(")")
    p = sub.add_parser(name, help=help)
    p.add_argument("--config", help="path to config YAML")
    p.set_defaults(_func=_not_implemented(milestone))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "_func", None)
    if func is None:  # pragma: no cover - argparse enforces a subcommand
        parser.print_help()
        return EXIT_ERROR
    result: int = func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
