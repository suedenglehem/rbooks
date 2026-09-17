"""Command-line interface.

M0 implements ``doctor`` fully. The remaining commands are registered with their
help text and milestone so that ``library-rag --help`` documents the whole
intended surface (PRD §4: "CLI help must document each command"). Each stub
prints an explicit "not yet implemented (milestone Mx)" message and returns a
non-zero exit code rather than pretending to do the work.

Destructive commands require an explicit ``--yes`` non-interactive flag; that
convention is implemented as each command lands (M1+).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from . import __version__
from .config import ConfigError, load_config
from .doctor import render_json, render_text, run_doctor
from .log import setup_logging

# Exit codes.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_IMPLEMENTED = 2


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


def _not_implemented(milestone: str) -> Callable[..., int]:
    def _run(_args: argparse.Namespace) -> int:
        print(f"command not yet implemented (planned: milestone {milestone})", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    return _run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="library-rag",
        description="Local-first RAG research application for a personal book library.",
    )
    parser.add_argument("--version", action="version", version=f"library-rag {__version__}")
    parser.add_argument("--log-level", default="INFO", help="log level (DEBUG/INFO/WARNING)")
    parser.add_argument("--log-format", default="human", choices=["human", "json"])

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    doctor = sub.add_parser("doctor", help="read-only environment diagnostics (distro, CPU, RAM, GPU, disks, OCR, services)")
    doctor.add_argument("--config", help="path to config YAML")
    doctor.add_argument("--json", action="store_true", help="emit the report as JSON")
    doctor.set_defaults(_func=_doctor)
    _register_stub(sub, "init", "create/verify the state directory and database (M1)")
    _register_stub(sub, "scan", "discover PDFs/EPUBs and register source revisions (M1)")
    _register_stub(sub, "ingest", "run the extraction->OCR->chunk->embed->publish pipeline (M1-M4)")
    _register_stub(sub, "pause", "persist pause; stop new job claims (M1)")
    _register_stub(sub, "resume", "clear a persisted pause (M1)")
    _register_stub(sub, "status", "show ingestion/pipeline status and counts (M1)")
    _register_stub(sub, "retry", "requeue retryable-failed jobs (M1)")
    _register_stub(sub, "search", "lexical + semantic search over the library (M4)")
    _register_stub(sub, "reconcile", "reconcile durable outputs and index/DB state (M1)")
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
