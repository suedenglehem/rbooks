"""Operational diagnostics for the research app (the Rag-page Diagnostics block).

Pure, read-only probes plus a few safe filesystem reads: catalog size and
drive headroom, serve/job log tails, LLM/embedder endpoint health, and
GPU/CPU state. Every probe is defensive — a missing file, a dead endpoint, or
no ``nvidia-smi`` degrades to an empty/"unavailable" field rather than raising,
so one broken probe never blanks the whole Diagnostics panel.

Like :mod:`browse` and :mod:`resumes`, this module has no FastAPI dependency:
the functions take a :class:`~.db.Database` / :class:`~.config.Config` and
return plain dicts the API layer serializes directly. The only side effect is
reading files that this process (or the worker) owns — the serve log and the
per-job failure logs — and those reads are containment-checked.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from psutil import virtual_memory

from .config import Config
from .db import Database, db_path_for
from .scan import iter_candidate_paths

__all__ = [
    "catalog_stats",
    "cpu_status",
    "gpu_status",
    "job_log_read",
    "job_logs_list",
    "serve_log_tail",
    "service_status",
]

# Per-job failure logs are named ``<job_id>.attempt<N>.log``. Anything else in
# the directory is not ours to serve; the strict pattern is the traversal guard.
_JOB_LOG_NAME = re.compile(r"^(?P<job_id>\d+)\.attempt(?P<attempt>\d+)\.log$")

# --- catalog -----------------------------------------------------------------


def catalog_stats(db: Database, cfg: Config) -> dict[str, Any]:
    """Catalog size, storage footprint, and the state DB's drive headroom.

    "Processed" means the book's active revision has been extracted (it has
    chunks); "published" means it is in the live searchable index (an active
    publication). Both are reported so the operator can see the gap between
    extracted and searchable.

    The ``disk_*`` fields count book files actually present in the configured
    source roots — the on-disk repository total (with a per-extension
    breakdown), walked with the scanner's own pruning and the global
    ``file_types`` so it is exactly what a scan would discover.
    ``disk_unprocessed`` is the on-disk total minus the processed count.

    Storage sizes are best-effort: the originals come from the catalog (exact,
    one query), the derived roots (state / qdrant / artifacts) are measured
    with ``du`` in parallel and fall back to 0.
    """
    books = _count(db, "SELECT COUNT(DISTINCT doc_id) AS n FROM documents")
    processed = _count(
        db,
        """
        SELECT COUNT(DISTINCT d.doc_id) AS n
        FROM documents d
        JOIN source_revisions r ON r.doc_id = d.doc_id AND r.is_active = 1
        JOIN chunks c ON c.rev_id = r.rev_id
        """,
    )
    published = _count(
        db, "SELECT COUNT(DISTINCT doc_id) AS n FROM publications WHERE state = 'active'"
    )
    chunks = _count(db, "SELECT COUNT(*) AS n FROM chunks")
    books_bytes = _count(
        db, "SELECT COALESCE(SUM(size_bytes), 0) AS n FROM source_revisions WHERE is_active = 1"
    )

    db_path = db_path_for(cfg.paths.state_root)
    db_size = _file_size(db_path)
    drive = _disk_usage(db_path)

    # Derived data lives in the state root (SQLite + job logs), the Qdrant
    # storage dir, and the artifact root. ``services.qdrant_path`` is the
    # embedded store when set; otherwise the configured qdrant_root.
    qdrant_dir = (
        Path(cfg.services.qdrant_path)
        if cfg.services.qdrant_path
        else cfg.paths.qdrant_root
    )
    # All four probes are blocking (du walks, os.walk over the source roots),
    # so they run in one pool: total cost ≈ the slowest, not the sum.
    du_targets = {
        "state": cfg.paths.state_root,
        "qdrant": qdrant_dir,
        "artifacts": cfg.paths.artifact_root,
    }
    with ThreadPoolExecutor(max_workers=len(du_targets) + 1) as pool:
        disk_fut = pool.submit(_disk_book_counts, cfg)
        du_futs = {name: pool.submit(_du, p) for name, p in du_targets.items()}
        derived = {name: fut.result() for name, fut in du_futs.items()}
        disk = disk_fut.result()
    return {
        "books": books,
        "processed_books": processed,
        "published_books": published,
        "chunks": chunks,
        "books_bytes": books_bytes,
        "disk_books": disk["total"],
        "disk_by_ext": disk["by_ext"],
        "disk_unprocessed": max(0, disk["total"] - processed),
        "derived_bytes": derived,
        "space_occupied_bytes": books_bytes + sum(derived.values()),
        "db_path": str(db_path),
        "db_size_bytes": db_size,
        "db_drive_total_bytes": drive["total"],
        "db_drive_free_bytes": drive["free"],
    }


def _count(db: Database, sql: str) -> int:
    row = db.query_one(sql)
    return int(row["n"]) if row is not None else 0


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _disk_usage(path: Path) -> dict[str, int]:
    try:
        du = shutil.disk_usage(str(path) if path.exists() else os.sep)
        return {"total": du.total, "free": du.free, "used": du.used}
    except OSError:
        return {"total": 0, "free": 0, "used": 0}


def _du(path: Path) -> int:
    """Apparent on-disk size of *path* via ``du -sb``; 0 when it fails.

    ``du`` walks the tree in C and is far faster than a Python ``os.walk`` over
    a 35k-book tree; it is also the number the operator expects ("how much
    space does this occupy"). Missing tool or unreadable path degrades to 0.
    """
    if not path.is_dir():
        return 0
    proc = _run(["du", "-sb", str(path)], timeout=20.0)
    if proc is None or proc.returncode != 0:
        return 0
    first = proc.stdout.split(maxsplit=1)
    try:
        return int(first[0])
    except (ValueError, IndexError):
        return 0


def _disk_book_counts(cfg: Config) -> dict[str, Any]:
    """Count book files on disk, in the configured source roots.

    Walks with the scanner's own pruning (ignore sets, symlinks never
    followed) and the global ``file_types``, so the total is exactly what a
    scan would discover. ``{"total": n, "by_ext": {".pdf": a, ".epub": b}}``;
    a missing or unmounted root contributes nothing.
    """
    by_ext: dict[str, int] = {}
    total = 0
    suffixes = frozenset(cfg.file_types)
    for root in cfg.paths.source_roots:
        if not root.is_dir():
            continue
        for p in iter_candidate_paths(
            root, cfg.scan.ignore_dirs, cfg.scan.ignore_files, suffixes
        ):
            ext = p.suffix.lower() or "(none)"
            by_ext[ext] = by_ext.get(ext, 0) + 1
            total += 1
    return {"total": total, "by_ext": by_ext}


# --- logs --------------------------------------------------------------------


def serve_log_tail(log_path: Path, lines: int = 1000) -> dict[str, Any]:
    """The last *lines* of the serve log, newest last.

    ``{"path", "exists", "line_count", "lines"}``. A missing file is not an
    error — the serve may log only to stderr (``exists: False, lines: []``).
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        return {"path": str(log_path), "exists": False, "line_count": 0, "lines": []}
    got = _tail_lines(log_path, lines)
    return {
        "path": str(log_path),
        "exists": True,
        "line_count": len(got),
        "lines": got,
    }


def job_logs_list(
    db: Database, job_logs_dir: Path, limit: int = 200
) -> dict[str, Any]:
    """List the per-job failure logs, newest first, annotated from the queue.

    Each entry carries the job's current state and ``error_category`` when the
    job is still in the table (so the UI can show "job 42 · ocr_failed"), which
    is what makes the list actionable. Jobs whose log file outlived a queue
    reset simply have ``state: null``.
    """
    job_logs_dir = Path(job_logs_dir)
    entries: list[dict[str, Any]] = []
    if job_logs_dir.is_dir():
        for p in job_logs_dir.iterdir():
            m = _JOB_LOG_NAME.match(p.name)
            if not m or not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append(
                {
                    "name": p.name,
                    "job_id": int(m.group("job_id")),
                    "attempt": int(m.group("attempt")),
                    "size_bytes": st.st_size,
                    "mtime": st.st_mtime,
                }
            )
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    entries = entries[:limit]

    # One query for the state/error of every listed job.
    job_ids = sorted({e["job_id"] for e in entries})
    states: dict[int, dict[str, Any]] = {}
    if job_ids:
        marks = ",".join("?" * len(job_ids))
        rows = db.query(
            f"""
            SELECT job_id, state, error_category, error_detail, stage, input_id
            FROM jobs WHERE job_id IN ({marks})
            """,
            tuple(job_ids),
        )
        for row in rows:
            states[int(row["job_id"])] = {
                "state": row["state"],
                "error_category": row["error_category"],
                "error_detail": (row["error_detail"] or "")[:300] or None,
                "stage": row["stage"],
                "input_id": row["input_id"],
            }
    for e in entries:
        meta = states.get(e["job_id"])
        e.update(meta if meta is not None else {"state": None})
    return {"dir": str(job_logs_dir), "count": len(entries), "logs": entries}


def job_log_read(
    job_logs_dir: Path, name: str, lines: int = 4000
) -> dict[str, Any] | None:
    """Read the last *lines* of one per-job failure log, or None.

    Returns None (→ 404) when *name* does not match the strict
    ``<job_id>.attempt<N>.log`` shape or does not name a file inside
    *job_logs_dir*. The containment check is on the realpath, so a crafted name
    cannot reach a sibling file.
    """
    job_logs_dir = Path(job_logs_dir)
    m = _JOB_LOG_NAME.match(name)
    if not m:
        return None
    base = os.path.realpath(job_logs_dir)
    target = os.path.realpath(os.path.join(base, name))
    if not _is_within(target, base) or not os.path.isfile(target):
        return None
    got = _tail_lines(Path(target), lines)
    return {"name": name, "line_count": len(got), "lines": got}


def _is_within(path: str, base: str) -> bool:
    """True when *path* is *base* or nested under it (both realpaths)."""
    try:
        return os.path.commonpath([path, base]) == base
    except ValueError:  # drives differ, or one path is empty
        return False


def _tail_lines(path: Path, n: int) -> list[str]:
    """Last *n* lines of *path* without loading the whole file.

    Reads 64 KiB blocks from the end and keeps only the tail, so a multi-MB
    job log costs a handful of reads regardless of its size. Returns ``[]`` for
    a missing or empty file. Multi-byte UTF-8 is kept as bytes until the final
    decode so a character split across a block boundary is never mangled.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size == 0 or n <= 0:
        return []
    block = 64 * 1024
    data = b""
    remaining = size
    with open(path, "rb") as f:
        while remaining > 0:
            step = min(block, remaining)
            remaining -= step
            f.seek(remaining)
            data = f.read(step) + data
            # n newlines guarantee n complete lines; one more read of the tail
            # cannot add a line we don't already have.
            if data.count(b"\n") >= n:
                break
    text = data.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":  # file ended with a newline
        lines.pop()
    # A multibyte character can only be split at the oldest edge of ``data``
    # (mid-line, not at a "\n"), and that edge line is always among the lines
    # dropped by the slice below — the kept lines are intact.
    return lines[-n:]


# --- services ----------------------------------------------------------------


def service_status(cfg: Config, timeout: float = 2.0) -> dict[str, Any]:
    """Health of the LLM answer endpoints and the embedding endpoints.

    Reports exactly what the app is configured to call: the primary answer
    endpoint plus ``answer.extra_endpoints``, and the embed pool
    (``embed_ports``). All pings run in parallel with a short
    connect/read timeout, so a fully-dead fleet costs ~*timeout* seconds, not
    *timeout* times the endpoint count.
    """
    svc = cfg.services
    llm = [
        ("primary", svc.answer_host, svc.answer_port)
    ] + [(f"extra-{i + 1}", e.host, e.port) for i, e in enumerate(cfg.answer.extra_endpoints)]
    ports = svc.embed_ports
    embed = [(f"embed-{p}", svc.embed_host, p) for p in ports]

    def ping(label: str, host: str, port: int) -> dict[str, Any]:
        up, detail = _ping(host, port, timeout)
        return {"label": label, "host": host, "port": port, "up": up, "detail": detail}

    with ThreadPoolExecutor(max_workers=max(1, len(llm) + len(embed))) as pool:
        results = list(
            pool.map(lambda t: ping(*t), [*llm, *embed])
        )
    split = len(llm)
    return {"llm": results[:split], "embedders": results[split:]}


def _ping(host: str, port: int, timeout: float) -> tuple[bool, str]:
    """GET-probe an endpoint; try /health then /v1/models. Never raises."""
    base = f"http://{host}:{port}"
    last = "no response"
    for suffix in ("/health", "/v1/models"):
        try:
            r = httpx.get(f"{base}{suffix}", timeout=timeout)
            return (r.status_code < 500, f"HTTP {r.status_code}")
        except httpx.HTTPError as exc:
            last = str(exc).splitlines()[0][:160]
    return (False, last)


# --- hardware ----------------------------------------------------------------


def gpu_status() -> dict[str, Any]:
    """Per-GPU make, memory, utilization, and temperature via nvidia-smi.

    ``{"available", "note", "gpus": [{index, name, mem_total_mib,
    mem_used_mib, mem_free_mib, util_pct, temp_c}]}``. No driver/CLI →
    ``available: False`` with a note; never raises.
    """
    if _run(["nvidia-smi", "--version"]) is None:
        return {
            "available": False,
            "note": "nvidia-smi not found (no NVIDIA driver/CLI on PATH)",
            "gpus": [],
        }
    proc = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,"
            "utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc is None or proc.returncode != 0:
        note = (proc.stderr.strip()[:200] if proc else "nvidia-smi failed")
        return {"available": False, "note": note, "gpus": []}
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "mem_total_mib": int(parts[2]),
                    "mem_used_mib": int(parts[3]),
                    "mem_free_mib": int(parts[4]),
                    "util_pct": int(parts[5]),
                    "temp_c": int(parts[6]),
                }
            )
        except ValueError:
            continue
    if not gpus:
        return {"available": False, "note": "nvidia-smi returned no GPUs", "gpus": []}
    return {"available": True, "note": f"{len(gpus)} GPU(s)", "gpus": gpus}


def cpu_status() -> dict[str, Any]:
    """CPU load average (normalized by core count) plus RAM headroom.

    Uses the 1/5/15-min load averages (fast, non-blocking) rather than a
    sampling ``cpu_percent`` so the probe never stalls the request.
    """
    logical = os.cpu_count() or 1
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = 0.0
    try:
        vm = virtual_memory()
        ram = {
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "used_pct": vm.percent,
        }
    except Exception:  # psutil hiccup must not blank the panel
        ram = {"total_bytes": 0, "available_bytes": 0, "used_pct": 0.0}
    return {
        "logical_cores": logical,
        "load1": round(load1, 2),
        "load5": round(load5, 2),
        "load15": round(load15, 2),
        "load_pct": round(100.0 * load1 / logical, 1),
        "ram": ram,
    }


# --- shared helper -----------------------------------------------------------


def _run(cmd: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str] | None:
    """Run *cmd*, capturing output; None when the binary is absent."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
