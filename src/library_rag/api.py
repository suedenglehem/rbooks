"""Research application API (PRD §12) — the FastAPI app behind ``serve``.

Built on the M2 read-only source reader: the same app instance serves the
reader routes (``/books/...``) plus the research surface:

* ``/health`` (liveness, always open) and ``/ready`` (service status);
* ``/library`` — the book list (active revisions, publication state);
* ``/search`` — fused search over the published index;
* ``/answer`` — cited answering with the persisted frozen evidence manifest;
* ``/answers`` / ``/answers/{id}`` / ``/answers/{id}/citations/{evidence_id}``
  — the answer history and snapshot-based citation resolution;
* ``/resumes/search`` and ``/resumes/{rev_id}`` — bm25 keyword search over the
  stored per-book summaries (M8) and a single resume's full record;
* ``/scan`` and ``/ingest/...`` — the ingestion dashboard controls
  (status, pause, resume, retry, rescan);
* ``/system/...`` — the Rag-page Diagnostics block: catalog stats, the
  serve-log tail, per-job failure logs, LLM/embedder/GPU/CPU status, and
  ``POST /system/shutdown`` (SIGTERM graceful stop of the serve process).

Auth (PRD §12): the app binds loopback by default. The bearer-token
requirement is a configurable master switch, ``services.require_api_token``,
OFF by default (local-machine runs need no token and the web UI never shows
its token field). When it is true, every request except liveness and static
asset delivery must carry ``Authorization: Bearer <services.api_token>`` and
the web UI shows the token field on the first 401. A Content-Security-Policy
header is added to every response so the bundled UI loads nothing from the
network.

Dependencies are injected (``qdrant``/``embedder``/``model``) so tests can
substitute the fakes; ``serve`` wires the real implementations.
"""

from __future__ import annotations

import os
import signal
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import ClientDisconnect

from . import __version__, system_info
from .answers import answer_query, get_answer, list_answers, resolve_citation
from .browse import BrowseError, BrowseNotFound, list_browse_dir
from .config import Config
from .db import Database
from .embeddings import Embedder
from .indexing import QdrantOps
from .jobs import Jobs
from .llm import AnswerModel
from .reader import active_run
from .reader import create_app as create_reader_app
from .resumes import get_resume, search_resumes
from .retrieval import IndexUnavailableError, Passage, search
from .scan import scan_roots

__all__ = ["create_app", "find_web_dist"]

# Content-Security-Policy for the bundled, fully-local UI.
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; worker-src 'self' blob:; font-src 'self'; "
    "connect-src 'self'; frame-src 'self' data:"
)

# Paths that stay open even when a bearer token is configured: liveness and
# static-asset delivery (the browser cannot attach headers to asset fetches).
_OPEN_PATHS = {"/health", "/", "/index.html"}


def _is_static_path(path: str) -> bool:
    segment = path.rsplit("/", 1)[-1]
    return "." in segment  # /assets/app-*.js, /pdf.worker-*.mjs, favicon, ...


def _self_terminate() -> None:
    """Graceful shutdown: SIGTERM to the serve process itself.

    uvicorn's handler drains in-flight requests, the server exits, and
    ``_serve`` closes the state DB and releases the embedded Qdrant lock (so
    the ingestion worker can start). Sent from a BackgroundTask *after* the
    HTTP response is flushed, so the client always sees the ack first.
    """
    os.kill(os.getpid(), signal.SIGTERM)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    doc: str | None = None
    rev: str | None = None
    limit: int | None = Field(default=None, ge=1, le=100)


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    doc: str | None = None
    rev: str | None = None


class PauseRequest(BaseModel):
    reason: str = ""


class RetryRequest(BaseModel):
    include_permanent: bool = False


class ResumeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=400)
    limit: int = Field(default=20, ge=1, le=50)


def find_web_dist() -> Path | None:
    """Locate the built research UI (``web/dist``) or return None.

    Resolution order: ``$LIBRARY_RAG_WEB_DIST`` (operator override), then the
    repo layout ``<package root>/../web/dist`` (``src/library_rag`` -> repo
    root). The app degrades gracefully (API only) when the UI was not built.
    """
    import os

    override = os.environ.get("LIBRARY_RAG_WEB_DIST")
    if override:
        p = Path(override)
        return p if (p / "index.html").is_file() else None
    candidate = Path(__file__).resolve().parent.parent.parent / "web" / "dist"
    return candidate if (candidate / "index.html").is_file() else None


def _passage_dict(p: Passage) -> dict[str, Any]:
    return {
        "chunk_id": p.chunk_id,
        "doc_id": p.doc_id,
        "rev_id": p.rev_id,
        "title": p.title,
        "text": p.text,
        "score": p.score,
        "dense_rank": p.dense_rank,
        "sparse_rank": p.sparse_rank,
        "spans": list(p.spans),
    }


def create_app(
    cfg: Config,
    db: Database,
    *,
    qdrant: QdrantOps,
    embedder: Embedder | None = None,
    model: AnswerModel | None = None,
) -> FastAPI:
    """Build the research app: reader routes + research surface on one instance."""
    app: FastAPI = create_reader_app(cfg, db)
    app.title = "library-rag research app"

    # Browse (M10) availability is soft: enabled AND the configured root
    # actually exists. A pulled mount degrades the feature (404s, tab
    # hidden via /ready) instead of blocking startup — checked here, at
    # app creation, the same layer where find_web_dist() touches the FS.
    browse_root = cfg.browse.root
    browse_available = bool(
        cfg.browse.enabled and browse_root is not None and browse_root.is_dir()
    )

    @app.middleware("http")
    async def _security(
        request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
        # The requirement is a master switch (require_api_token), not a side
        # effect of a value existing: a configured token while the switch is
        # off stays inert, so the default local run is open and the web UI
        # never sees a 401 to prompt on. (load_config rejects a switch-on
        # config without a value, so `token` is non-empty when enforced.)
        token = cfg.services.api_token
        if (
            cfg.services.require_api_token
            and token
            and request.url.path not in _OPEN_PATHS
            and not _is_static_path(request.url.path)
        ):
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {token}":
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        try:
            response = await call_next(request)
        except ClientDisconnect:  # client went away mid-response: not an error
            return JSONResponse({"detail": "client disconnected"}, status_code=499)
        response.headers["Content-Security-Policy"] = _CSP
        return response

    # --- health -----------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {
            "qdrant": qdrant.ping(),
            "embedding_model": embedder is not None,
            "answer_model": model is not None,
            # The web UI uses this to decide whether the token widget (field +
            # "Set" button) exists at all: when the server does not enforce
            # bearer auth, the widget stays hidden, the UI never prompts, and
            # a stored token is forgotten instead of sent.
            "token_required": bool(
                cfg.services.require_api_token and cfg.services.api_token
            ),
            # The web UI shows the Browse nav button only when this is true;
            # the 30 s poll makes a returning mount reappear on its own.
            "browse": browse_available,
        }

    # --- library ----------------------------------------------------------------

    @app.get("/library")
    def library() -> dict[str, Any]:
        published = {
            r["rev_id"]
            for r in db.query("SELECT DISTINCT rev_id FROM publications WHERE state = 'active'")
        }
        books: list[dict[str, Any]] = []
        for row in db.query(
            """
            SELECT d.doc_id, d.updated_at, r.rev_id, r.format, r.size_bytes,
                   r.first_path, r.created_at
            FROM documents d
            JOIN source_revisions r ON r.doc_id = d.doc_id AND r.is_active = 1
            ORDER BY d.doc_id
            """
        ):
            rev_id = str(row["rev_id"])
            run = active_run(db, rev_id)
            chunk_count = 0
            if run is not None:
                n = db.query_one(
                    "SELECT COUNT(*) AS n FROM chunks WHERE run_id = ?", (run["run_id"],)
                )
                chunk_count = int(n["n"]) if n is not None else 0
            books.append(
                {
                    "doc_id": row["doc_id"],
                    "title": Path(str(row["first_path"])).stem,
                    "format": row["format"],
                    "rev_id": rev_id,
                    "size_bytes": int(row["size_bytes"]),
                    "published": rev_id in published,
                    "chunk_count": chunk_count,
                }
            )
        return {"books": books}

    # --- search / answer ---------------------------------------------------------

    @app.post("/search")
    def do_search(body: SearchRequest) -> dict[str, Any]:
        try:
            result = search(
                db, cfg, qdrant, embedder, body.query, doc_id=body.doc, rev_id=body.rev
            )
        except IndexUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        passages = result.passages[: body.limit] if body.limit else result.passages
        return {
            "query": result.query,
            "degraded": result.degraded,
            "degraded_reason": result.degraded_reason,
            "counts": result.counts,
            "passages": [_passage_dict(p) for p in passages],
        }

    @app.post("/answer")
    def do_answer(body: AnswerRequest) -> dict[str, Any]:
        try:
            result = answer_query(
                db, cfg, qdrant, embedder, model, body.query,
                doc_id=body.doc, rev_id=body.rev,
            )
        except IndexUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        # Serve the persisted row: exactly what history and citation
        # resolution will show later (frozen manifest and all).
        data = get_answer(db, result.answer_id)
        assert data is not None, "persisted answer row is missing"
        return data

    @app.get("/answers")
    def answers(limit: int = 20) -> dict[str, Any]:
        return {"answers": list_answers(db, limit=limit)}

    @app.get("/answers/{answer_id}")
    def one_answer(answer_id: str) -> dict[str, Any]:
        data = get_answer(db, answer_id)
        if data is None:
            raise HTTPException(404, "unknown answer")
        return data

    @app.get("/answers/{answer_id}/citations/{evidence_id}")
    def one_citation(answer_id: str, evidence_id: str) -> dict[str, Any]:
        data = get_answer(db, answer_id)
        if data is None:
            raise HTTPException(404, "unknown answer")
        raw = data.get("evidence")
        if not isinstance(raw, list):
            raise HTTPException(404, "unknown evidence id for this answer")
        entry = next(
            (e for e in raw if isinstance(e, dict) and e.get("evidence_id") == evidence_id),
            None,
        )
        if entry is None:
            raise HTTPException(404, "unknown evidence id for this answer")
        from .citations import Evidence

        return resolve_citation(db, cfg, Evidence.from_dict(entry))

    # --- book resumes (M8) -------------------------------------------------------

    @app.post("/resumes/search")
    def do_resume_search(body: ResumeSearchRequest) -> dict[str, Any]:
        return {"results": search_resumes(db, body.query, limit=body.limit)}

    @app.get("/resumes/{rev_id}")
    def one_resume(rev_id: str) -> dict[str, Any]:
        data = get_resume(db, rev_id)
        if data is None:
            raise HTTPException(404, "no stored resume for this revision")
        return data

    # --- browse (M10) -------------------------------------------------------------

    @app.get("/browse/dir")
    def browse_dir(path: str = "") -> dict[str, Any]:
        if not browse_available or browse_root is None:
            # Flag off, root unconfigured, or mount pulled — the SPA hides
            # the tab via /ready; direct calls 404.
            raise HTTPException(404, "browse is not available")
        try:
            return list_browse_dir(db, browse_root, path, cfg.file_types)
        except BrowseNotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except BrowseError as exc:
            raise HTTPException(400, str(exc)) from exc

    # --- ingestion dashboard ------------------------------------------------------

    def _status_data() -> dict[str, Any]:
        jobs = Jobs(db)
        from .migrations import current_version

        return {
            "paused": jobs.is_paused(),
            "jobs": jobs.counts(),
            "schema_version": current_version(db),
            "documents": _count(db, "SELECT COUNT(*) AS n FROM documents"),
            "revisions": _count(db, "SELECT COUNT(*) AS n FROM source_revisions"),
            "chunks": _count(db, "SELECT COUNT(*) AS n FROM chunks"),
            "answers": _count(db, "SELECT COUNT(*) AS n FROM answers"),
            "resumes": _count(db, "SELECT COUNT(*) AS n FROM book_resumes"),
        }

    @app.get("/ingest/status")
    def ingest_status() -> dict[str, Any]:
        return _status_data()

    @app.post("/scan")
    def do_scan() -> dict[str, Any]:
        try:
            reports = scan_roots(db, cfg, Jobs(db))
        except Exception as exc:  # a scan failure must not 500-opaquely
            raise HTTPException(500, f"scan failed: {exc}") from exc
        return {"reports": [asdict(r) for r in reports]}

    @app.post("/ingest/pause")
    def ingest_pause(body: PauseRequest) -> dict[str, Any]:
        Jobs(db).pause(reason=body.reason)
        return {"paused": True}

    @app.post("/ingest/resume")
    def ingest_resume() -> dict[str, Any]:
        Jobs(db).resume()
        return {"paused": False}

    @app.post("/ingest/retry")
    def ingest_retry(body: RetryRequest) -> dict[str, Any]:
        requeued = Jobs(db).retry(include_permanent=body.include_permanent)
        return {"requeued": requeued}

    # --- system diagnostics (Rag-page Diagnostics block) ------------------------
    #
    # Read-only probes (catalog, logs, endpoints, GPU/CPU) plus one
    # self-directed action: graceful shutdown. All of them are outside
    # _OPEN_PATHS, so the bearer middleware protects them automatically.

    state_root = cfg.paths.state_root

    @app.get("/system/catalog")
    def system_catalog() -> dict[str, Any]:
        return system_info.catalog_stats(db, cfg)

    @app.get("/system/log")
    def system_log(lines: int = Query(default=1000, ge=1, le=5000)) -> dict[str, Any]:
        return system_info.serve_log_tail(state_root / "logs" / "serve.log", lines=lines)

    @app.get("/system/job-logs")
    def system_job_logs(limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, Any]:
        return system_info.job_logs_list(db, state_root / "job_logs", limit=limit)

    @app.get("/system/job-logs/{name}")
    def system_job_log(
        name: str, lines: int = Query(default=4000, ge=1, le=20000)
    ) -> dict[str, Any]:
        data = system_info.job_log_read(state_root / "job_logs", name, lines=lines)
        if data is None:
            raise HTTPException(404, "unknown or invalid job log name")
        return data

    @app.get("/system/status")
    def system_status(
        timeout: float = Query(default=2.0, ge=0.5, le=15.0)
    ) -> dict[str, Any]:
        return {
            "services": system_info.service_status(cfg, timeout=timeout),
            "gpu": system_info.gpu_status(),
            "cpu": system_info.cpu_status(),
        }

    @app.post("/system/shutdown")
    def system_shutdown(background: BackgroundTasks) -> dict[str, Any]:
        background.add_task(_self_terminate)
        return {"shutting_down": True}

    # --- static UI (last: the catch-all mount must not shadow the API routes) -----
    dist = find_web_dist()
    if dist is not None:
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")

    return app


def _count(db: Database, sql: str) -> int:
    row = db.query_one(sql)
    return int(row["n"]) if row is not None else 0


