"""Research application API (PRD §12) — the FastAPI app behind ``serve``.

Built on the M2 read-only source reader: the same app instance serves the
reader routes (``/books/...``) plus the research surface:

* ``/health`` (liveness, always open) and ``/ready`` (service status);
* ``/library`` — the book list (active revisions, publication state);
* ``/search`` — fused search over the published index;
* ``/answer`` — cited answering with the persisted frozen evidence manifest;
* ``/answers`` / ``/answers/{id}`` / ``/answers/{id}/citations/{evidence_id}``
  — the answer history and snapshot-based citation resolution;
* ``/scan`` and ``/ingest/...`` — the ingestion dashboard controls
  (status, pause, resume, retry, rescan).

Auth (PRD §12: the app binds loopback by default; a bearer token is required
before any non-loopback exposure): when ``services.api_token`` is set, every
request except liveness and static asset delivery must carry
``Authorization: Bearer <token>``. A Content-Security-Policy header is added
to every response so the bundled UI loads nothing from the network.

Dependencies are injected (``qdrant``/``embedder``/``model``) so tests can
substitute the fakes; ``serve`` wires the real implementations.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import ClientDisconnect

from . import __version__
from .answers import answer_query, get_answer, list_answers, resolve_citation
from .config import Config
from .db import Database
from .embeddings import Embedder
from .indexing import QdrantOps
from .jobs import Jobs
from .llm import AnswerModel
from .reader import active_run
from .reader import create_app as create_reader_app
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

    @app.middleware("http")
    async def _security(
        request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
        token = cfg.services.api_token
        if token and request.url.path not in _OPEN_PATHS and not _is_static_path(
            request.url.path
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

    # --- static UI (last: the catch-all mount must not shadow the API routes) -----
    dist = find_web_dist()
    if dist is not None:
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")

    return app


def _count(db: Database, sql: str) -> int:
    row = db.query_one(sql)
    return int(row["n"]) if row is not None else 0


