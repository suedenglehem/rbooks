"""Command-line interface.

``doctor`` (M0), the recovery-foundation commands (M1: ``init``, ``pause``,
``resume``, ``status``, ``retry``, ``reconcile``), the M2 pipeline commands
``scan`` (discover + register + enqueue) and ``ingest`` (worker loop over the
extraction queue), the M4 ``search`` command (fused dense+sparse search over
the published index, with sparse-only degraded mode when embedding inference
is unavailable), and the M5 commands ``answer`` (cited answering with the
persisted frozen evidence manifest), ``serve`` (the FastAPI research app),
and ``evaluate`` (labeled retrieval/answer metrics, PRD §13) are
implemented. The remaining commands are registered with their help text and
milestone so that ``library-rag --help`` documents the whole intended surface
(PRD §4). Each stub prints an explicit "not yet implemented (milestone Mx)"
message and returns a non-zero exit code rather than pretending to do the
work.

Destructive commands require an explicit ``--yes`` non-interactive flag; that
convention is implemented as each command lands (M1+).
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import signal
import sys
from collections.abc import Callable
from dataclasses import asdict

from . import __version__
from .answers import answer_query, get_answer
from .api import create_app
from .catalog import source_file_count
from .config import Config, ConfigError, load_config
from .db import Database, db_path_for
from .doctor import render_json, render_text, run_doctor
from .embeddings import Embedder, make_embedder
from .evaluate import evaluate, format_report, load_dataset
from .identity import normalize_path
from .indexing import QdrantOps, RealQdrantOps, reconcile_publications
from .jobs import Jobs
from .llm import AnswerModel, make_answer_model
from .log import setup_logging
from .migrations import current_version, migrate
from .reconcile import Mount, reconcile_catalog
from .retrieval import IndexUnavailableError, search
from .scan import scan_roots
from .worker import DEFAULT_LEASE_TTL, run_worker

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
            "ocr_state": {
                r["ocr_state"]: int(r["n"])
                for r in db.query(
                    "SELECT ocr_state, COUNT(*) AS n FROM source_units GROUP BY ocr_state"
                )
            },
            "chunks": _count(db, "SELECT COUNT(*) AS n FROM chunks"),
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
        ocr_state = data["ocr_state"]
        if isinstance(ocr_state, dict):
            ocr_counts = " ".join(f"{k}={v}" for k, v in sorted(ocr_state.items()))
            print(f"ocr: {ocr_counts}")
        print(f"chunks={data['chunks']}")
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


# --- M2 pipeline commands ----------------------------------------------------
def _scan(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    try:
        reports = scan_roots(db, cfg, Jobs(db))
    finally:
        db.close()
    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2, sort_keys=True))
    else:
        for r in reports:
            if r.mount_unavailable:
                print(f"mount unavailable: {r.root} (skipped, nothing deleted)")
                continue
            print(
                f"{r.root}: discovered={r.discovered} unchanged={r.unchanged} "
                f"new_documents={r.new_documents} new_revisions={r.new_revisions} "
                f"aliases={r.aliases} jobs={r.jobs_enqueued}"
            )
            if r.invalid:
                print(f"  invalid (content mismatch): {', '.join(r.invalid)}")
            if r.missing:
                print(f"  missing (reported, not deleted): {', '.join(r.missing)}")
            if r.changed_during_scan:
                print(f"  changed during scan (retried next scan): {', '.join(r.changed_during_scan)}")
    return EXIT_OK


def _ingest(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    stop = False

    def _graceful_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True  # finish the in-flight unit, then stop claiming (PRD §7)

    try:
        if not args.once:
            signal.signal(signal.SIGTERM, _graceful_stop)
            signal.signal(signal.SIGINT, _graceful_stop)
        try:
            completed = run_worker(
                db, cfg, once=args.once, lease_ttl=args.lease_ttl, stop_event=lambda: stop
            )
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
    finally:
        db.close()
    print(f"ingest: {completed} job(s) completed")
    return EXIT_OK


# --- M4 search ---------------------------------------------------------------
def _search(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    qdrant = RealQdrantOps(cfg)
    try:
        if not qdrant.ping():
            print("error: Qdrant is unreachable; search is unavailable", file=sys.stderr)
            return EXIT_ERROR
        # Best effort: repair partial publication flag changes from a crash
        # before serving evidence (PRD §8F: reconcile after restart). Search
        # must survive a failed reconciliation pass.
        with contextlib.suppress(Exception):
            reconcile_publications(db, cfg, qdrant)
        # Embedding inference optional: unconfigured -> sparse-only degraded
        # mode with an explicit status (PRD §9).
        embedder = None
        with contextlib.suppress(ConfigError):
            embedder = make_embedder(cfg)
        result = search(
            db, cfg, qdrant, embedder, args.query, doc_id=args.doc, rev_id=args.rev
        )
    except IndexUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        db.close()
    passages = result.passages[: args.limit] if args.limit else result.passages
    if args.json:
        print(json.dumps(
            {
                "query": result.query,
                "degraded": result.degraded,
                "degraded_reason": result.degraded_reason,
                "counts": result.counts,
                "passages": [
                    {
                        "chunk_id": p.chunk_id,
                        "doc_id": p.doc_id,
                        "rev_id": p.rev_id,
                        "title": p.title,
                        "text": p.text,
                        "score": p.score,
                        "dense_rank": p.dense_rank,
                        "sparse_rank": p.sparse_rank,
                        "spans": p.spans,
                    }
                    for p in passages
                ],
            },
            indent=2,
            sort_keys=True,
        ))
    else:
        if result.degraded:
            print(f"warning: degraded search: {result.degraded_reason}", file=sys.stderr)
        if not passages:
            print("no passages found")
        for i, p in enumerate(passages, start=1):
            print(f"{i}. {p.title or '(untitled)'}  "
                  f"[doc={p.doc_id} score={p.score:.4f}]")
            print(f"   {p.text}")
    return EXIT_OK


# --- M5 answer / serve / evaluate -------------------------------------------
def _index_and_models(
    cfg: Config, db: Database
) -> tuple[QdrantOps, Embedder | None, AnswerModel | None]:
    """Qdrant + optional embedder/answer-model wiring shared by answer & co.

    Returns ``(qdrant, embedder, model)``; raises
    :class:`IndexUnavailableError` when Qdrant is unreachable.
    """
    qdrant = RealQdrantOps(cfg)
    if not qdrant.ping():
        raise IndexUnavailableError("Qdrant is unreachable")
    # Best effort: repair partial publication flag changes from a crash
    # before serving evidence (PRD §8F: reconcile after restart).
    with contextlib.suppress(Exception):
        reconcile_publications(db, cfg, qdrant)
    # Embedding inference optional: unconfigured -> sparse-only degraded
    # mode with an explicit status (PRD §9).
    embedder = None
    with contextlib.suppress(ConfigError):
        embedder = make_embedder(cfg)
    model = make_answer_model(cfg)  # optional; never raises
    return qdrant, embedder, model


def _answer(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    data: dict[str, object] | None = None
    try:
        qdrant, embedder, model = _index_and_models(cfg, db)
        result = answer_query(
            db, cfg, qdrant, embedder, model, args.query,
            doc_id=args.doc, rev_id=args.rev,
        )
        # Serve the persisted row: what history and citation resolution show.
        data = get_answer(db, result.answer_id)
    except IndexUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        db.close()
    if data is None:  # pragma: no cover - the row was written above
        print("error: persisted answer row missing", file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"status: {data['status']}  (answer {data['answer_id']})")
        if data["abstain_reason"]:
            print(f"abstained: {data['abstain_reason']}")
        if data["failure_reason"]:
            print(f"failed: {data['failure_reason']}")
        if data["answer_text"]:
            print(data["answer_text"])
        citations = data.get("citations")
        if isinstance(citations, list) and citations:
            print("citations: " + " ".join(str(c) for c in citations))
        evidence = data.get("evidence")
        if isinstance(evidence, list):
            for e in evidence:
                if not isinstance(e, dict):
                    continue
                loc = e.get("location")
                if isinstance(loc, dict):
                    kind = str(loc.get("kind"))
                    where = (
                        f"page {loc.get('page')}"
                        if kind == "page"
                        else f"section {loc.get('ref')}" if kind == "section"
                        else "location unknown"
                    )
                else:
                    where = "location unknown"
                print(f"  {e['evidence_id']}: {e['source_title']} ({where})")
    return EXIT_OK if data["status"] in ("answered", "abstained") else EXIT_ERROR


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.strip("[]").lower() in {"localhost", "::1"}


def _serve(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    host, port = cfg.services.app_host, cfg.services.app_port
    if not _is_loopback(host) and not cfg.services.api_token:
        print(
            "error: binding to a non-loopback address requires services.api_token "
            "(PRD §12: bearer auth before any network exposure)",
            file=sys.stderr,
        )
        return EXIT_ERROR
    db = _open_state(cfg)
    try:
        qdrant, embedder, model = _index_and_models(cfg, db)
        app = create_app(cfg, db, qdrant=qdrant, embedder=embedder, model=model)
    except IndexUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        db.close()
        return EXIT_ERROR
    import uvicorn  # local import: keeps the other commands import-light

    uvicorn.run(app, host=host, port=port, log_level=args.log_level.lower())
    db.close()
    return EXIT_OK


def _evaluate(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    try:
        dataset = load_dataset(args.dataset)
    except (OSError, ValueError) as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    db = _open_state(cfg)
    try:
        qdrant, embedder, model = _index_and_models(cfg, db)
        report = evaluate(db, cfg, qdrant, embedder, model, dataset, k=args.k)
    except IndexUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        db.close()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
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

    scan = sub.add_parser(
        "scan", help="discover PDFs/EPUBs, register source revisions, enqueue extraction (M2)"
    )
    scan.add_argument("--config", help="path to config YAML")
    scan.add_argument("--json", action="store_true", help="emit the scan reports as JSON")
    scan.set_defaults(_func=_scan)

    ingest = sub.add_parser(
        "ingest", help="run the worker loop over the pipeline queue (M2-M4)"
    )
    ingest.add_argument("--config", help="path to config YAML")
    ingest.add_argument(
        "--once",
        action="store_true",
        help="drain the queue and exit (default: run until SIGTERM/SIGINT)",
    )
    ingest.add_argument(
        "--lease-ttl",
        type=float,
        default=DEFAULT_LEASE_TTL,
        help=f"lease TTL in seconds (default {DEFAULT_LEASE_TTL:.0f})",
    )
    ingest.set_defaults(_func=_ingest)

    search_p = sub.add_parser(
        "search", help="lexical + semantic search over the published index (M4)"
    )
    search_p.add_argument("query", help="the query text")
    search_p.add_argument("--config", help="path to config YAML")
    search_p.add_argument("--json", action="store_true", help="emit the result as JSON")
    search_p.add_argument(
        "--limit", type=int, default=None, help="at most this many passages (default: all selected)"
    )
    search_p.add_argument("--doc", default=None, help="restrict to one document id")
    search_p.add_argument("--rev", default=None, help="restrict to one revision id")
    search_p.set_defaults(_func=_search)

    answer_p = sub.add_parser(
        "answer", help="cited answering over the published index (M5)"
    )
    answer_p.add_argument("query", help="the question to answer")
    answer_p.add_argument("--config", help="path to config YAML")
    answer_p.add_argument("--json", action="store_true", help="emit the answer as JSON")
    answer_p.add_argument("--doc", default=None, help="restrict to one document id")
    answer_p.add_argument("--rev", default=None, help="restrict to one revision id")
    answer_p.set_defaults(_func=_answer)

    serve = sub.add_parser(
        "serve", help="run the FastAPI research app + reader (M5)"
    )
    serve.add_argument("--config", help="path to config YAML")
    serve.set_defaults(_func=_serve)

    evaluate_p = sub.add_parser(
        "evaluate", help="run retrieval/answer evaluation on a labeled set (M5, PRD §13)"
    )
    evaluate_p.add_argument("--dataset", required=True, help="path to the labeled dataset JSON")
    evaluate_p.add_argument("--config", help="path to config YAML")
    evaluate_p.add_argument("--json", action="store_true", help="emit the report as JSON")
    evaluate_p.add_argument("--k", type=int, default=20, help="candidates per question (default 20)")
    evaluate_p.set_defaults(_func=_evaluate)

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
