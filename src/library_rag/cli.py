"""Command-line interface.

``doctor`` (M0), the recovery-foundation commands (M1: ``init``, ``pause``,
``resume``, ``status``, ``retry``, ``reconcile``), the M2 pipeline commands
``scan`` (discover + register + enqueue) and ``ingest`` (worker loop over the
extraction queue), the M4 ``search`` command (fused dense+sparse search over
the published index, with sparse-only degraded mode when embedding inference
is unavailable), and the M5 commands ``answer`` (cited answering with the
persisted frozen evidence manifest), ``serve`` (the FastAPI research app),
``evaluate`` (labeled retrieval/answer metrics, PRD §13), and the M6
``pilot`` group (read-only corpus survey and deterministic stratified sample
manifest, PRD §12), and the M7 ``coverage`` (pipeline funnel report,
PRD §2/§14) and ``discover`` (scheduled discovery, PRD lines 15/83/181)
commands are implemented. The remaining commands are registered
with their help text and
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
import hashlib
import ipaddress
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

from . import __version__
from .answers import answer_query, get_answer
from .api import create_app
from .backup import BackupError, create_backup, restore_backup, verify_system
from .catalog import source_file_count
from .config import MOUNT_SENTINEL_ENV, Config, ConfigError, load_config
from .coverage import CoverageReport, coverage_report
from .db import Database, db_path_for
from .discover import DEFAULT_DISCOVER_INTERVAL, run_discovery
from .doctor import render_json, render_text, run_doctor
from .embeddings import Embedder, make_embedder
from .evaluate import evaluate, format_report, load_dataset
from .gc import GcError, run_gc
from .identity import normalize_path
from .indexing import QdrantOps, RealQdrantOps, reconcile_publications
from .jobs import Jobs
from .latency import build_probe_queries, run_latency
from .llm import AnswerModel, make_answer_model
from .log import setup_logging
from .migrate import MigrationError, MigrationReport, run_migration
from .migrations import current_version, migrate
from .pilot import (
    MANIFEST_SCHEMA,
    PILOT_REPORT_SCHEMA,
    format_manifest_summary,
    load_survey,
    run_pilot,
    sample_manifest,
    sandbox_config,
    survey_sources,
    write_manifest,
    write_pilot_report,
)
from .pilot_report import build_report, write_report
from .questions import (
    QUESTION_BANK_SCHEMA,
    QUESTION_CATEGORIES,
    QuestionEntry,
    add_question,
    export_dataset,
    load_bank,
    suggest_candidates,
)
from .reconcile import Mount, reconcile_catalog
from .removal import RemovalError, remove_document, resolve_target
from .retrieval import IndexUnavailableError, search
from .scan import scan_roots
from .worker import DEFAULT_LEASE_TTL, run_worker

# Exit codes.
EXIT_OK = 0
EXIT_ERROR = 1


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
        # does not exist — or whose configured mount sentinel is missing — is
        # treated as an unavailable mount and left untouched, so an empty but
        # present mount point can never prune its aliases.
        mounts = [
            Mount(
                label=str(r),
                root=r,
                sentinel=Path(sp) if (sp := cfg.mount_sentinels.get(str(r))) is not None else None,
            )
            for r in cfg.paths.source_roots
        ]
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


def _discover(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    stop = False

    def _graceful_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True  # finish the in-flight pass, then stop (PRD §7, like ingest)

    try:
        signal.signal(signal.SIGTERM, _graceful_stop)
        signal.signal(signal.SIGINT, _graceful_stop)
        try:
            passes = run_discovery(
                db, cfg, Jobs(db), interval_seconds=args.interval, stop_event=lambda: stop
            )
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
    finally:
        db.close()
    print(f"discover: {passes} pass(es) completed")
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


# --- M7 backup / restore / verify --------------------------------------------


def _config_file_path(cli_value: str | None) -> Path | None:
    """The config file in use: CLI flag, then $LIBRARY_RAG_CONFIG, else None."""
    p = cli_value or os.environ.get(MOUNT_SENTINEL_ENV)
    return Path(p) if p else None


def _backup(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    dest = Path(args.to) if args.to else cfg.paths.backup_root
    if dest is None:
        print(
            "error: no backup destination: pass --to or set paths.backup_root",
            file=sys.stderr,
        )
        return EXIT_ERROR
    db = _open_state(cfg)
    try:
        manifest = create_backup(
            db,
            cfg,
            dest,
            include_contents=not args.state_only,
            config_source=_config_file_path(args.config),
        )
    except BackupError as exc:
        print(f"backup error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        db.close()
    total = sum(f["size"] for f in manifest["files"])
    print(f"backup complete: {dest}")
    print(f"  files: {len(manifest['files'])} ({total / 2**30:.2f} GiB in {manifest['elapsed_seconds']} s)")
    print(f"  paused: {manifest['paused']}  qdrant: {manifest['qdrant']}")
    print(f"  counts: {json.dumps(manifest['counts'], sort_keys=True)}")
    print(f"verify: library-rag verify --config <config> --backup {dest}")
    return EXIT_OK


def _restore(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    try:
        manifest = restore_backup(Path(args.backup), Path(args.target), force=args.force)
    except BackupError as exc:
        print(f"restore error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    restore = manifest["restore"]
    print(f"restored to {restore['target']}")
    print(f"  files: linked {restore['files_linked']}, copied {restore['files_copied']}")
    print(f"  counts: {json.dumps(manifest['counts'], sort_keys=True)}")
    print(
        f"verify: library-rag verify --config {restore['target']}/config.yaml "
        f"--backup {restore['target']}"
    )
    return EXIT_OK


def _verify(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    qdrant: RealQdrantOps | None = None
    try:
        embedder: Embedder | None = None
        with contextlib.suppress(ConfigError):
            embedder = make_embedder(cfg)
        # A running worker holds the local Qdrant storage lock for its whole
        # run; verify must still work while the system is up, so the index
        # and search checks degrade instead of failing to open the client.
        try:
            candidate = RealQdrantOps(cfg)
        except Exception:
            candidate = None
        qdrant = candidate if candidate is not None and candidate.ping() else None
        results = verify_system(
            cfg,
            db,
            qdrant,
            search_query=args.search,
            backup_dir=Path(args.backup) if args.backup else None,
            full_checksums=args.full_checksums,
            embedder=embedder,
        )
    finally:
        if qdrant is not None:
            qdrant.close()
        db.close()
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2, sort_keys=True))
    else:
        for r in results:
            print(f"{'ok  ' if r.ok else 'FAIL'} {r.name}: {r.detail}")
    return EXIT_OK if all(r.ok for r in results) else EXIT_ERROR


def _gc(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    qdrant: RealQdrantOps | None = None
    try:
        # The points kind needs a Qdrant client; in local mode a running
        # worker holds the storage lock, so the client degrades to None and
        # the file kinds still run (the report carries a note).
        try:
            candidate = RealQdrantOps(cfg)
        except Exception:
            candidate = None
        qdrant = candidate if candidate is not None and candidate.ping() else None
        report = run_gc(
            db,
            cfg,
            execute=args.execute,
            grace_seconds=args.grace_seconds,
            qdrant=qdrant,
        )
    except GcError as exc:
        print(f"gc error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if qdrant is not None:
            qdrant.close()
        db.close()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        total = sum(c.size_bytes for c in report.candidates)
        mode = "execute" if report.executed else "dry-run"
        print(f"gc ({mode}): {len(report.candidates)} candidate(s), {total / 2**20:.1f} MiB")
        by_kind: dict[str, int] = {}
        for c in report.candidates:
            by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
        for kind in ("archive", "artifact", "job_log", "points"):
            if by_kind.get(kind):
                print(f"  {kind}: {by_kind[kind]}")
        for c in report.candidates[:20]:
            print(f"  [{c.kind}] {c.relpath}  ({c.size_bytes} B, {c.reason})")
        if len(report.candidates) > 20:
            print(f"  ... and {len(report.candidates) - 20} more")
        if report.executed:
            print(
                f"  reclaimed: {report.deleted} file(s), "
                f"{report.bytes_reclaimed / 2**20:.1f} MiB, "
                f"{report.directories_removed} dir(s)"
            )
        for e in report.errors:
            print(f"  error: {e}", file=sys.stderr)
        for note in report.notes:
            print(f"  note: {note}")
        if not report.executed and report.candidates:
            print("dry-run: nothing deleted; re-run with --execute to delete")
    return EXIT_OK if not report.errors else EXIT_ERROR


def _migrate(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    try:
        report = run_migration(
            db,
            cfg,
            execute=args.execute,
            accept_maintenance_window=args.accept_maintenance_window,
        )
    except MigrationError as exc:
        print(f"migrate error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        db.close()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _migrate_text(report)
    return EXIT_OK


def _migrate_text(r: MigrationReport) -> None:
    g = 2**30
    mode = "execute" if r.executed else "dry-run"
    print(f"migrate ({mode})")
    print(
        f"  rechunk runs: {r.rechunk_runs} (active-revision: {r.rechunk_active_runs})  "
        f"reembed runs: {r.reembed_runs}"
    )
    print(
        f"  publications to supersede: {r.publications_to_supersede}  "
        f"migrating points: {r.migrating_points}  active points: {r.active_points}"
    )
    if r.qdrant_storage_bytes is not None:
        line = f"  qdrant store: {r.qdrant_storage_bytes / g:.1f} GiB"
        if r.free_bytes is not None:
            line += f"  free: {r.free_bytes / g:.1f} GiB"
        print(line)
        if r.second_generation_bytes:
            print(f"  estimated second generation: {r.second_generation_bytes / g:.1f} GiB")
    if r.fits is None:
        print("  two-generation fit: UNKNOWN (capacity unverifiable — see notes)")
    else:
        print(f"  two-generation fit: {'yes' if r.fits else 'NO'}")
    if r.maintenance_window_required:
        print(
            "  maintenance window: REQUIRED (stop the worker, run gc --execute, "
            "re-check; or accept explicitly with --accept-maintenance-window)"
        )
    else:
        print("  maintenance window: not required")
    for note in r.notes:
        print(f"  note: {note}")
    if r.executed:
        print(f"  enqueued jobs: {r.enqueued_jobs}")
        if r.enqueued_jobs == 0:
            print("  (nothing drifted — no jobs enqueued)")


def _remove(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    qdrant: RealQdrantOps | None = None
    try:
        # A running worker holds the local Qdrant storage lock; a document
        # with no index points must still be removable, so the client degrades
        # to None instead of failing the command.
        try:
            candidate = RealQdrantOps(cfg)
        except Exception:
            candidate = None
        qdrant = candidate if candidate is not None and candidate.ping() else None
        doc_id = resolve_target(db, args.target)
        report = remove_document(db, cfg, qdrant, doc_id, execute=args.execute)
    except RemovalError as exc:
        print(f"remove error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if qdrant is not None:
            qdrant.close()
        db.close()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        mode = "execute" if report.executed else "dry-run"
        print(f"remove ({mode}): {report.doc_id}")
        print(
            f"  points: {report.points}  publications: {report.publications}  "
            f"generations: {report.generations}  revisions: {report.revisions}"
        )
        print(
            f"  runs: {report.extraction_runs}  units: {report.source_units}  "
            f"batches: {report.embedding_batches}  "
            f"jobs cancelled: {report.jobs_cancelled}"
        )
        if report.archive_objects:
            total = sum(int(o["size_bytes"]) for o in report.archive_objects)
            print(
                f"  archive: {len(report.archive_objects)} object(s), "
                f"{total / 2**20:.1f} MiB"
            )
        for sha in report.archive_kept:
            print(f"  archive kept (still referenced): {sha}")
        if report.executed and report.verified:
            print("  verified: all catalog rows and index points removed")
        for e in report.errors:
            print(f"  error: {e}", file=sys.stderr)
        if not report.executed:
            print("dry-run: nothing deleted; re-run with --execute to remove")
    return EXIT_OK if not report.errors else EXIT_ERROR


def _coverage(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    qdrant: RealQdrantOps | None = None
    try:
        # A running worker holds the local Qdrant storage lock for its whole
        # run; the report must still work while the system is up, so the
        # point counts degrade to None instead of failing the command.
        try:
            candidate = RealQdrantOps(cfg)
        except Exception:
            candidate = None
        qdrant = candidate if candidate is not None and candidate.ping() else None
        report = coverage_report(db, cfg, qdrant=qdrant)
    finally:
        if qdrant is not None:
            qdrant.close()
        db.close()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _coverage_text(report)
    return EXIT_OK


def _coverage_text(report: CoverageReport) -> None:
    s = report.stages
    f = report.failures
    discovered = sum(rc.discovered for rc in report.roots if not rc.mount_unavailable)
    print(
        f"coverage: {s.documents} document(s), {s.revisions} revision(s), "
        f"{s.published} active publication(s)"
    )
    print(
        f"  funnel: discovered {discovered} -> archived {s.archived} -> "
        f"extracted {s.extracted} -> chunked {s.chunked} -> embedded {s.embedded} -> "
        f"indexed {s.indexed} -> published {s.published}"
    )
    print(
        f"  ocr units: routed {s.ocr_routed}, done {s.ocr_done}, "
        f"pending {s.ocr_pending}, failed {s.ocr_failed}"
    )
    if report.points_active is None:
        print(f"  points: catalog {report.points_expected}, index: unavailable")
    else:
        print(
            f"  points: catalog {report.points_expected}, "
            f"active {report.points_active}, total {report.points_total}"
        )
    failed_jobs = f.failed_jobs_by_stage
    detail = f" ({', '.join(f'{k} {v}' for k, v in sorted(failed_jobs.items()))})" if failed_jobs else ""
    print(
        f"  failures: {sum(failed_jobs.values())} failed job(s){detail}, "
        f"{f.failed_extractions} failed extraction(s), {f.ocr_failed_units} failed OCR unit(s)"
    )
    if s.archive_missing:
        print(
            f"  archive missing: {len(s.archive_missing)} object(s): "
            f"{', '.join(s.archive_missing[:10])}"
        )
    for rc in report.roots:
        if rc.mount_unavailable:
            print(f"  root {rc.root}: mount unavailable")
            continue
        print(
            f"  root {rc.root}: discovered {rc.discovered}, registered {rc.registered}, "
            f"unindexed {len(rc.unindexed)}, orphaned {len(rc.orphaned)}, "
            f"stale {len(rc.stale)}, invalid {len(rc.invalid)}"
        )
        for label, paths in (
            ("unindexed", rc.unindexed),
            ("orphaned", rc.orphaned),
            ("stale", rc.stale),
            ("invalid", rc.invalid),
        ):
            for p in paths[:10]:
                print(f"    {label}: {p}")
            if len(paths) > 10:
                print(f"    ... and {len(paths) - 10} more {label}")
    if report.stalled:
        by_stage: dict[str, int] = {}
        for sd in report.stalled:
            by_stage[sd.stage] = by_stage.get(sd.stage, 0) + 1
        summary = ", ".join(f"{k} {v}" for k, v in sorted(by_stage.items()))
        print(f"  stalled (no active publication): {len(report.stalled)} document(s): {summary}")
        for sd in report.stalled[:10]:
            print(f"    {sd.stage}: {sd.doc_id}")
        if len(report.stalled) > 10:
            print(f"    ... and {len(report.stalled) - 10} more")


# --- M6 pilot ---------------------------------------------------------------


def _pilot_survey(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    for root in cfg.paths.source_roots:
        if not root.is_dir():
            print(f"warning: source root missing: {root}", file=sys.stderr)
    out = Path(args.out)
    state = {"total": 0}

    def on_progress(done: int, total: int) -> None:
        state["total"] = total
        print(f"  survey: {done}/{total} profiled", file=sys.stderr)

    try:
        summary = survey_sources(cfg, out, limit=args.limit, on_progress=on_progress)
    except OSError as exc:
        print(f"survey error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"survey: {summary.files_scanned} candidates recorded in {out} (this pass: {summary.seconds}s)")
    for fmt, n in sorted(summary.by_format.items()):
        print(f"  format {fmt}: {n}")
    print(f"  strata: {len(summary.by_stratum)}  errored: {summary.errors}")
    return EXIT_OK


def _pilot_sample(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    survey_path = Path(args.survey)
    if not survey_path.is_file():
        print(f"survey error: no such file: {survey_path}", file=sys.stderr)
        return EXIT_ERROR
    if not (200 <= args.target <= 500) and not args.allow_out_of_range:
        print(
            f"error: target {args.target} is outside the PRD pilot range 200-500 "
            "(pass --allow-out-of-range for sandbox smoke tests)",
            file=sys.stderr,
        )
        return EXIT_ERROR
    if args.page_cap is not None and args.page_cap < 1:
        print("error: --page-cap must be >= 1 (or omit it for no cap)", file=sys.stderr)
        return EXIT_ERROR
    try:
        records = load_survey(survey_path)
        survey_sha = hashlib.sha256(survey_path.read_bytes()).hexdigest()
        manifest = sample_manifest(
            records, seed=args.seed, target=args.target, page_cap=args.page_cap, survey_sha=survey_sha
        )
        write_manifest(manifest, Path(args.out))
    except (OSError, ValueError) as exc:
        print(f"sample error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        print(f"manifest written: {args.out}")
        for line in format_manifest_summary(manifest):
            print(line)
    return EXIT_OK


def _pilot_run(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"run error: no such manifest: {manifest_path}", file=sys.stderr)
        return EXIT_ERROR
    try:
        metrics = run_pilot(cfg, manifest_path, Path(args.sandbox_root))
    except (OSError, ValueError, ConfigError) as exc:
        print(f"run error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    out = Path(args.out) if args.out else Path(args.sandbox_root) / "scratch" / "pilot_run.json"
    try:
        write_pilot_report(metrics, out)
    except OSError as exc:
        print(f"run error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(
            f"pilot run: {metrics.documents} documents, {metrics.chunks} chunks "
            f"({metrics.chunk_tokens} tokens) in {metrics.seconds:.1f}s -> {out}"
        )
        for stage, s in sorted(metrics.stages.items()):
            print(
                f"  {stage}: jobs={s['jobs']} ok={s['succeeded']} failed={s['failed']} "
                f"{s['seconds']:.1f}s (max {s['seconds_max']:.1f}s)"
            )
        if metrics.failed_jobs:
            print(f"  failed jobs: {len(metrics.failed_jobs)} (details in the report)")
        for name, d in sorted(metrics.disk.items()):
            print(f"  disk {name}: +{d['delta_bytes'] / 1048576:.1f} MiB")
    return EXIT_OK


def _pilot_latency(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    sandbox_root = Path(args.sandbox_root)
    try:
        sandbox_cfg = sandbox_config(cfg, sandbox_root, page_cap=args.page_cap)
    except (OSError, ValueError, ConfigError) as exc:
        print(f"latency error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if not db_path_for(sandbox_cfg.paths.state_root).is_file():
        print(
            f"latency error: no sandbox index at {sandbox_root} "
            "(run `pilot run` against this sandbox first)",
            file=sys.stderr,
        )
        return EXIT_ERROR
    questions = None
    if args.questions:
        try:
            questions = [e.question for e in load_bank(args.questions)]
        except (OSError, ValueError) as exc:
            print(f"latency error: {exc}", file=sys.stderr)
            return EXIT_ERROR
    db = _open_state(sandbox_cfg)
    try:
        qdrant = RealQdrantOps(sandbox_cfg)
        embedder = make_embedder(sandbox_cfg)
        probes = build_probe_queries(db, questions=questions, n=args.probes)
        result = run_latency(
            db,
            sandbox_cfg,
            qdrant,
            embedder,
            probe_queries=probes,
            limit=args.limit,
            reps=args.reps,
            ingest_root=args.ingest_root,
            ingest_count=args.ingest_count,
        )
    except (OSError, ValueError, ConfigError) as exc:
        print(f"latency error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        db.close()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"latency: {result['probes']} probes x {args.reps} reps, limit {args.limit}")
        for phase in ("idle", "ingesting"):
            s = cast("dict[str, object] | None", result[phase])
            if s is None:
                continue
            print(
                f"  {phase}: n={s['n']} mean={s['mean_ms']}ms p50={s['p50_ms']}ms "
                f"p95={s['p95_ms']}ms max={s['max_ms']}ms"
            )
        if result["enqueued"]:
            print(f"  backlog enqueued: {result['enqueued']} books")
    return EXIT_OK


def _pilot_report(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    survey = None
    if args.survey:
        try:
            survey = load_survey(Path(args.survey))
        except (OSError, ValueError) as exc:
            print(f"report error: survey: {exc}", file=sys.stderr)
            return EXIT_ERROR
    manifest = None
    if args.manifest:
        try:
            raw = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"report error: manifest: {exc}", file=sys.stderr)
            return EXIT_ERROR
        if not isinstance(raw, dict) or raw.get("schema") != MANIFEST_SCHEMA:
            print(
                f"report error: manifest: schema must be {MANIFEST_SCHEMA!r}", file=sys.stderr
            )
            return EXIT_ERROR
        manifest = raw
    run = None
    if args.run:
        try:
            raw = json.loads(Path(args.run).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"report error: run: {exc}", file=sys.stderr)
            return EXIT_ERROR
        if not isinstance(raw, dict) or raw.get("schema") != PILOT_REPORT_SCHEMA:
            print(
                f"report error: run: schema must be {PILOT_REPORT_SCHEMA!r}", file=sys.stderr
            )
            return EXIT_ERROR
        run = raw
    questions = None
    if args.questions:
        try:
            questions = load_bank(args.questions)
        except (OSError, ValueError) as exc:
            print(f"report error: questions: {exc}", file=sys.stderr)
            return EXIT_ERROR
    cfg = None
    if args.config:
        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            print(f"report error: config: {exc}", file=sys.stderr)
            return EXIT_ERROR
    if survey is None and manifest is None and run is None and questions is None:
        print(
            "report error: give at least one of --survey --manifest --run --questions",
            file=sys.stderr,
        )
        return EXIT_ERROR
    try:
        report = build_report(
            survey=survey, manifest=manifest, run=run, questions=questions, config=cfg
        )
    except (OSError, ValueError) as exc:
        print(f"report error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.out:
        out = Path(args.out)
    elif run and run.get("sandbox_root"):
        out = Path(str(run["sandbox_root"])) / "scratch" / "pilot_report.md"
    else:
        out = Path("pilot_report.md")
    try:
        write_report(report, out)
    except OSError as exc:
        print(f"report error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"report written: {out}")
        proj = report.get("projections")
        if proj:
            print(
                f"  full corpus: ~{proj['full_chunks_est']:,} chunks, "
                f"ETA {proj['full_run_duration_est']} (single worker)"
            )
        storage = report.get("storage")
        if storage:
            print(f"  storage estimate: {storage['total_gb']} GB")
        q = report.get("questions") or {}
        if q:
            print(f"  question bank: {q['total']} — {q['status']}")
    return EXIT_OK


def _pilot_annotate_add(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    # The from_dict round-trip enforces the bank's invariants (category/
    # answerable consistency) at write time, not only at read time.
    entry = QuestionEntry.from_dict(
        {
            "id": args.id or "",
            "question": args.question,
            "category": args.category,
            "expected_chunks": list(args.expected_chunk or []),
            "answerable": not args.unanswerable,
            "source_paths": list(args.source_path or []),
            "notes": args.notes or "",
            "labeled_by": args.labeled_by or "",
            "labeled_at": time.time(),
        }
    )
    try:
        added = add_question(args.file, entry)
    except (OSError, ValueError) as exc:
        print(f"annotate error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(json.dumps(added.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"added {added.id} [{added.category}] to {args.file}")
    return EXIT_OK


def _pilot_annotate_list(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    try:
        entries = load_bank(args.file)
    except (OSError, ValueError) as exc:
        print(f"annotate error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(
            json.dumps(
                {"schema": QUESTION_BANK_SCHEMA, "questions": [e.to_dict() for e in entries]},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        for e in entries:
            flag = "answerable" if e.answerable else "UNANSWERABLE"
            print(f"  {e.id} [{e.category}] {flag}  {e.question}")
        print(f"{len(entries)} questions in {args.file}")
    return EXIT_OK


def _pilot_annotate_export(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    try:
        entries = load_bank(args.file)
        payload = export_dataset(entries)
    except (OSError, ValueError) as exc:
        print(f"annotate error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.out:
        try:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"annotate error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"exported {len(entries)} questions -> {out}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return EXIT_OK


def _pilot_annotate_suggest(args: argparse.Namespace) -> int:
    setup_logging(args.log_level, args.log_format)
    cfg = _require_config(args)
    db = _open_state(cfg)
    try:
        candidates = suggest_candidates(db, limit=args.limit, seed=args.seed)
    except (OSError, ValueError) as exc:
        print(f"annotate error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        db.close()
    for c in candidates:
        print(json.dumps(c, ensure_ascii=False))
    print(
        f"# {len(candidates)} candidates (unlabeled — review before `pilot annotate add`)",
        file=sys.stderr,
    )
    return EXIT_OK


# --- parser ----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="library-rag",
        description="Local-first RAG research application for a personal book library.",
    )
    parser.add_argument("--version", action="version", version=f"library-rag {__version__}")
    parser.add_argument(
        "--log-level",
        default=None,
        help="log level (DEBUG/INFO/WARNING/ERROR); overrides config logging.level",
    )
    parser.add_argument(
        "--log-format",
        default=None,
        choices=["human", "json"],
        help="log format; overrides config logging.format",
    )

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

    discover = sub.add_parser(
        "discover",
        help="scheduled discovery: re-scan source roots on an interval (M7)",
    )
    discover.add_argument("--config", help="path to config YAML")
    discover.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_DISCOVER_INTERVAL,
        help=f"seconds between discovery passes (default {DEFAULT_DISCOVER_INTERVAL:.0f})",
    )
    discover.set_defaults(_func=_discover)

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

    backup_p = sub.add_parser(
        "backup",
        help="consistent backup of state DB, Qdrant storage, archive, artifacts (M7, PRD §14)",
    )
    backup_p.add_argument("--config", help="path to config YAML")
    backup_p.add_argument(
        "--to", help="backup destination directory (default paths.backup_root)"
    )
    backup_p.add_argument(
        "--state-only",
        action="store_true",
        help="skip archive+artifacts (fast state+index consistency backup)",
    )
    backup_p.set_defaults(_func=_backup)

    restore_p = sub.add_parser(
        "restore", help="restore a backup into an isolated directory (M7)"
    )
    restore_p.add_argument("--backup", required=True, help="backup directory to restore")
    restore_p.add_argument(
        "--target", required=True, help="fresh target directory (created if absent)"
    )
    restore_p.add_argument(
        "--force", action="store_true", help="allow an existing EMPTY target directory"
    )
    restore_p.set_defaults(_func=_restore)

    verify_p = sub.add_parser(
        "verify",
        help="verify system/backup integrity: DB, source links, artifacts, index, checksums (M7)",
    )
    verify_p.add_argument("--config", help="path to config YAML")
    verify_p.add_argument(
        "--backup", help="also verify this backup directory's manifest checksums"
    )
    verify_p.add_argument("--search", help="run a smoke search with this query")
    verify_p.add_argument(
        "--full-checksums",
        action="store_true",
        help="re-hash every backed-up file (default trusts content-addressed paths)",
    )
    verify_p.add_argument("--json", action="store_true", help="emit results as JSON")
    verify_p.set_defaults(_func=_verify)

    gc_p = sub.add_parser(
        "gc",
        help="garbage-collect unreferenced archive/artifact/log objects (M7, dry-run by default)",
    )
    gc_p.add_argument("--config", help="path to config YAML")
    gc_p.add_argument(
        "--execute",
        action="store_true",
        help="delete candidates (default is a dry-run report)",
    )
    gc_p.add_argument(
        "--grace-seconds",
        type=float,
        default=600.0,
        help="skip files modified within this window (default 600)",
    )
    gc_p.add_argument("--json", action="store_true", help="emit the report as JSON")
    gc_p.set_defaults(_func=_gc)

    remove_p = sub.add_parser(
        "remove",
        help="explicitly remove a book: index points, catalog rows, archive object (M7)",
    )
    remove_p.add_argument("target", help="source path or doc_id of the book to remove")
    remove_p.add_argument("--config", help="path to config YAML")
    remove_p.add_argument(
        "--execute",
        action="store_true",
        help="perform the removal (default is a dry-run report)",
    )
    remove_p.add_argument("--json", action="store_true", help="emit the report as JSON")
    remove_p.set_defaults(_func=_remove)

    coverage_p = sub.add_parser(
        "coverage",
        help="coverage report: pipeline funnel vs source roots (M7, read-only)",
    )
    coverage_p.add_argument("--config", help="path to config YAML")
    coverage_p.add_argument("--json", action="store_true", help="emit the report as JSON")
    coverage_p.set_defaults(_func=_coverage)

    migrate_p = sub.add_parser(
        "migrate",
        help="plan/execute a generation migration after a chunker or embedding change (M7)",
    )
    migrate_p.add_argument("--config", help="path to config YAML")
    migrate_p.add_argument(
        "--execute",
        action="store_true",
        help="enqueue the re-chunk/re-embed/publish jobs (default is a plan report)",
    )
    migrate_p.add_argument(
        "--accept-maintenance-window",
        action="store_true",
        help="allow execution even though the new generation does not fit under two generations",
    )
    migrate_p.add_argument("--json", action="store_true", help="emit the report as JSON")
    migrate_p.set_defaults(_func=_migrate)

    pilot_p = sub.add_parser(
        "pilot",
        help="M6 pilot tooling: corpus survey, stratified sample, run, latency, report, "
        "question bank (PRD §12)",
    )
    pilot_sub = pilot_p.add_subparsers(dest="pilot_command", required=True, metavar="STAGE")
    ps = pilot_sub.add_parser(
        "survey", help="read-only corpus survey of candidate books (JSONL, resumable)"
    )
    ps.add_argument("--config", help="path to config YAML")
    ps.add_argument("--out", required=True, help="survey JSONL output path")
    ps.add_argument(
        "--limit", type=int, default=None, help="profile at most N files this pass (testing)"
    )
    ps.set_defaults(_func=_pilot_survey)
    pm = pilot_sub.add_parser(
        "sample", help="deterministic stratified manifest from a survey (200-500 books)"
    )
    pm.add_argument("--survey", required=True, help="survey JSONL produced by `pilot survey`")
    pm.add_argument("--out", required=True, help="manifest JSON output path")
    pm.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    pm.add_argument("--target", type=int, default=300, help="books to sample (PRD: 200-500)")
    pm.add_argument("--page-cap", type=int, default=None, help="per-book page cap (omit for no cap)")
    pm.add_argument("--json", action="store_true", help="emit the full manifest as JSON")
    pm.add_argument(
        "--allow-out-of-range",
        action="store_true",
        help="permit a target outside 200-500 (sandbox smoke tests only)",
    )
    pm.set_defaults(_func=_pilot_sample)
    pr = pilot_sub.add_parser(
        "run",
        help="ingest exactly the manifest's books into an isolated sandbox and measure it",
    )
    pr.add_argument("--config", help="path to config YAML")
    pr.add_argument("--manifest", required=True, help="manifest JSON from `pilot sample`")
    pr.add_argument(
        "--sandbox-root",
        required=True,
        help="isolated writable root (archive/state/qdrant/scratch live under it; the real index is never touched)",
    )
    pr.add_argument(
        "--out", default=None, help="metrics JSON output (default <sandbox>/scratch/pilot_run.json)"
    )
    pr.add_argument("--json", action="store_true", help="emit the full metrics as JSON")
    pr.set_defaults(_func=_pilot_run)
    pl = pilot_sub.add_parser(
        "latency",
        help="p50/p95 retrieval latency on the sandbox index, idle vs during ingestion",
    )
    pl.add_argument("--config", help="path to config YAML")
    pl.add_argument(
        "--sandbox-root",
        required=True,
        help="sandbox from `pilot run` (its state DB + qdrant directory are measured)",
    )
    pl.add_argument("--questions", default=None, help="question bank JSON (probes verbatim)")
    pl.add_argument("--probes", type=int, default=10, help="probe queries (default 10)")
    pl.add_argument("--reps", type=int, default=1, help="repeat the measurement pass (default 1)")
    pl.add_argument("--limit", type=int, default=20, help="candidates per probe (default 20)")
    pl.add_argument(
        "--page-cap", type=int, default=None, help="per-book page cap applied to the ingest backlog"
    )
    pl.add_argument(
        "--ingest-root",
        default=None,
        help="directory of extra books to queue while measuring (omitted: idle only)",
    )
    pl.add_argument(
        "--ingest-count",
        type=int,
        default=0,
        help="max books from --ingest-root to enqueue (default 0: idle only)",
    )
    pl.add_argument("--json", action="store_true", help="emit the full result as JSON")
    pl.set_defaults(_func=_pilot_latency)
    pr2 = pilot_sub.add_parser(
        "report",
        help="aggregate survey + manifest + run (+ question bank) into the capacity report",
    )
    pr2.add_argument("--survey", default=None, help="survey JSONL from `pilot survey`")
    pr2.add_argument("--manifest", default=None, help="manifest JSON from `pilot sample`")
    pr2.add_argument("--run", default=None, help="pilot run report JSON from `pilot run`")
    pr2.add_argument("--questions", default=None, help="question bank JSON from `pilot annotate`")
    pr2.add_argument("--config", default=None, help="config YAML (cited in the frozen-config section)")
    pr2.add_argument(
        "--out", default=None, help="markdown output (default <sandbox>/scratch/pilot_report.md)"
    )
    pr2.add_argument("--json", action="store_true", help="also emit the report as JSON")
    pr2.set_defaults(_func=_pilot_report)
    pa = pilot_sub.add_parser(
        "annotate",
        help="question-bank tooling (add/list/export/suggest; labels are the operator's work)",
    )
    pa_sub = pa.add_subparsers(dest="annotate_command", required=True, metavar="ACTION")
    paa = pa_sub.add_parser(
        "add", help="append one operator-labeled question (the bank's only write path)"
    )
    paa.add_argument("--file", required=True, help="question bank JSON path")
    paa.add_argument("--question", required=True, help="the question text (verbatim)")
    paa.add_argument("--category", required=True, choices=list(QUESTION_CATEGORIES))
    paa.add_argument("--id", default="", help="question id (default: auto qNNN)")
    paa.add_argument("--expected-chunk", action="append", default=[], help="expected chunk id (repeatable)")
    paa.add_argument("--unanswerable", action="store_true", help="mark unanswerable (forces category=unanswerable)")
    paa.add_argument("--source-path", action="append", default=[], help="source book path (repeatable)")
    paa.add_argument("--notes", default="", help="free-form labeler notes")
    paa.add_argument("--labeled-by", default="", help="who labeled this")
    paa.add_argument("--json", action="store_true", help="emit the stored entry as JSON")
    paa.set_defaults(_func=_pilot_annotate_add)
    pal = pa_sub.add_parser("list", help="show the bank")
    pal.add_argument("--file", required=True, help="question bank JSON path")
    pal.add_argument("--json", action="store_true", help="emit the bank as JSON")
    pal.set_defaults(_func=_pilot_annotate_list)
    pae = pa_sub.add_parser(
        "export", help="project the bank onto the `evaluate` dataset schema"
    )
    pae.add_argument("--file", required=True, help="question bank JSON path")
    pae.add_argument("--out", default=None, help="output JSON (default: stdout)")
    pae.set_defaults(_func=_pilot_annotate_export)
    pas = pa_sub.add_parser(
        "suggest",
        help="print machine-generated, UNLABELED question candidates from the index",
    )
    pas.add_argument("--config", help="path to config YAML")
    pas.add_argument("--limit", type=int, default=40, help="chunks to sample (default 40)")
    pas.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    pas.set_defaults(_func=_pilot_annotate_suggest)

    return parser


def _apply_config_logging(args: argparse.Namespace) -> None:
    """Resolve the short-log level/format from config.yaml, flags as overrides.

    ``--log-level`` / ``--log-format`` default to ``None`` so we can tell an
    explicit operator flag from "not set". The per-subcommand ``--config``
    names the YAML; we load it best-effort here and fill in
    ``args.log_level`` / ``args.log_format`` so every handler's existing
    ``setup_logging(args.log_level, args.log_format)`` call picks the resolved
    values up without change. Precedence: explicit flag > config > built-in
    default (INFO / human). The config load is best-effort: a missing or
    invalid file just falls back to the flag/defaults, and the handler's own
    ``_require_config`` reports the real error.
    """
    level = args.log_level
    fmt = args.log_format
    cfg_path = getattr(args, "config", None)
    if cfg_path:
        try:
            cfg = load_config(cfg_path)
        except ConfigError:
            cfg = None
        if cfg is not None:
            if level is None:
                level = cfg.logging.level
            if fmt is None:
                fmt = cfg.logging.format
    args.log_level = "INFO" if level is None else level
    args.log_format = "human" if fmt is None else fmt


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "_func", None)
    if func is None:  # pragma: no cover - argparse enforces a subcommand
        parser.print_help()
        return EXIT_ERROR
    _apply_config_logging(args)
    result: int = func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
