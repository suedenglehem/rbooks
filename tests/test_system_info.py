"""Rag-page Diagnostics (system_info): the pure probes and the /system/* routes.

Every probe must be defensive — a missing file, a dead endpoint, or no
nvidia-smi degrades to an empty/"unavailable" field rather than raising.
The route tests use the hand-built published library, so they also pin the
auth behavior (the /system paths are bearer-protected, like the rest of the
API surface).
"""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fixtures import publish_handbuilt
from library_rag import system_info as si
from library_rag.api import create_app
from library_rag.catalog import Format, register_source
from library_rag.config import AnswerEndpoint, Config
from library_rag.db import Database
from library_rag.embeddings import FakeEmbedder
from library_rag.indexing import FakeQdrant
from library_rag.llm import FakeAnswerModel

H1 = "11" * 32
H2 = "22" * 32


# --- catalog -------------------------------------------------------------------


def _seed_processed_book(db: Database, path: str, sha: str, size: int) -> tuple[str, str]:
    """Register a book, then hand-insert a succeeded run + two chunks for it."""
    reg = register_source(db, path, sha, size, Format.PDF)
    run_id = f"run-{sha[:8]}"
    db.execute(
        """
        INSERT INTO extraction_runs
            (run_id, rev_id, doc_id, parser_version, settings_sha, unit_count,
             state, created_at, updated_at)
        VALUES (?, ?, ?, "t", "s", 1, "succeeded", 0, 0)
        """,
        (run_id, reg.rev_id, reg.doc_id),
    )
    for pos in (0, 1):
        db.execute(
            """
            INSERT INTO chunks
                (chunk_id, run_id, rev_id, position, text, token_count,
                 spans, created_at)
            VALUES (?, ?, ?, ?, "chunk", 1, "[]", 0)
            """,
            (f"{run_id}:c{pos}", run_id, reg.rev_id, pos),
        )
    return reg.doc_id, reg.rev_id


def _seed_published(db: Database, doc_id: str, rev_id: str, run_id: str) -> None:
    db.execute(
        """
        INSERT INTO index_generations
            (gen_id, run_id, rev_id, model_revision, embedding_sha,
             sparse_stats_sha, dimensions, dtype, normalized, state,
             created_at, updated_at)
        VALUES ("gen1", ?, ?, "m", "e", "s", 8, "f32", 1, "ready", 0, 0)
        """,
        (run_id, rev_id),
    )
    db.execute(
        """
        INSERT INTO publications
            (pub_id, rev_id, doc_id, gen_id, run_id, expected_points,
             state, created_at, activated_at)
        VALUES ("pub1", ?, ?, "gen1", ?, 2, "active", 0, 0)
        """,
        (rev_id, doc_id, run_id),
    )


def test_catalog_stats_counts(state_db: Database, base_config: Config) -> None:
    doc_a, rev_a = _seed_processed_book(state_db, "/books/a.pdf", H1, 100)
    register_source(state_db, "/books/b.pdf", H2, 200, Format.PDF)  # unprocessed
    _seed_published(state_db, doc_a, rev_a, f"run-{H1[:8]}")

    stats = si.catalog_stats(state_db, base_config)
    assert stats["books"] == 2
    assert stats["processed_books"] == 1
    assert stats["published_books"] == 1
    assert stats["chunks"] == 2
    assert stats["books_bytes"] == 300
    assert stats["space_occupied_bytes"] >= 300
    assert set(stats["derived_bytes"]) == {"state", "qdrant", "artifacts"}
    # The source root does not exist on disk: nothing to count there.
    assert stats["disk_books"] == 0
    assert stats["disk_by_ext"] == {}
    assert stats["disk_unprocessed"] == 0
    # The state DB exists and is non-empty after the seeds above.
    assert stats["db_size_bytes"] > 0
    assert stats["db_drive_total_bytes"] > stats["db_drive_free_bytes"] >= 0
    assert stats["db_path"].endswith("library.db")


def test_catalog_stats_empty_db(state_db: Database, base_config: Config) -> None:
    stats = si.catalog_stats(state_db, base_config)
    assert stats["books"] == 0
    assert stats["processed_books"] == 0
    assert stats["published_books"] == 0
    assert stats["chunks"] == 0
    assert stats["books_bytes"] == 0
    assert stats["space_occupied_bytes"] >= 0
    assert stats["disk_books"] == 0
    assert stats["disk_by_ext"] == {}
    assert stats["disk_unprocessed"] == 0


def test_catalog_stats_disk_walk(state_db: Database, base_config: Config) -> None:
    # "Books on disk" must be exactly what a scan would discover: the global
    # file_types, the scanner's ignore sets, symlinks never followed.
    src = base_config.paths.source_roots[0]
    (src / ".git").mkdir(parents=True)
    (src / "a.pdf").write_bytes(b"%PDF")
    (src / "A.PDF").write_bytes(b"%PDF")  # case-folds into the same bucket
    (src / "b.epub").write_bytes(b"EPUB")
    (src / "notes.txt").write_bytes(b"nope")  # not a configured type
    (src / ".git" / "hidden.pdf").write_bytes(b"%PDF")  # ignored dir
    (src / "link.pdf").symlink_to(src / "a.pdf")  # symlinks are not counted

    stats = si.catalog_stats(state_db, base_config)
    assert stats["disk_books"] == 3
    assert stats["disk_by_ext"] == {".pdf": 2, ".epub": 1}
    assert stats["disk_unprocessed"] == 3  # nothing in the catalog yet

    # Narrowing the global file_types narrows the count the same way a
    # scan would.
    cfg = base_config.model_copy(deep=True)
    cfg.file_types = [".pdf"]
    stats = si.catalog_stats(state_db, cfg)
    assert stats["disk_books"] == 2
    assert stats["disk_by_ext"] == {".pdf": 2}


# --- serve log -------------------------------------------------------------------


def test_serve_log_tail_missing(tmp_path: Path) -> None:
    out = si.serve_log_tail(tmp_path / "logs" / "serve.log", 100)
    assert out == {"path": str(tmp_path / "logs" / "serve.log"), "exists": False,
                   "line_count": 0, "lines": []}


def test_serve_log_tail_returns_newest_last(tmp_path: Path) -> None:
    p = tmp_path / "serve.log"
    p.write_text("\n".join(f"line {i}" for i in range(1, 2501)) + "\n", encoding="utf-8")
    out = si.serve_log_tail(p, 1000)
    assert out["exists"] is True
    assert out["line_count"] == 1000
    assert out["lines"][0] == "line 1501"
    assert out["lines"][-1] == "line 2500"


def test_serve_log_tail_fewer_lines_than_requested(tmp_path: Path) -> None:
    p = tmp_path / "serve.log"
    p.write_text("a\nb\nc", encoding="utf-8")  # no trailing newline
    out = si.serve_log_tail(p, 100)
    assert out["lines"] == ["a", "b", "c"]


def test_serve_log_tail_multibyte(tmp_path: Path) -> None:
    # Multi-byte UTF-8 must survive block-boundary reads untouched.
    p = tmp_path / "serve.log"
    line = "héllo wörld 日本語 — " * 20
    p.write_text("\n".join(line for _ in range(500)) + "\n", encoding="utf-8")
    out = si.serve_log_tail(p, 10)
    assert out["lines"] == [line] * 10


def test_serve_log_tail_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "serve.log"
    p.write_bytes(b"")
    out = si.serve_log_tail(p, 100)
    assert out["exists"] is True
    assert out["lines"] == []


# --- job logs --------------------------------------------------------------------


def _make_job_logs(dirpath: Path) -> None:
    dirpath.mkdir(parents=True)
    (dirpath / "1.attempt1.log").write_text("job 1 boom\n", encoding="utf-8")
    (dirpath / "2.attempt3.log").write_text("job 2 kaboom\n" * 10, encoding="utf-8")
    (dirpath / "12.attempt1.log").write_text("orphan\n", encoding="utf-8")
    (dirpath / "notes.txt").write_text("not a job log\n", encoding="utf-8")
    (dirpath / "2.attempt3.log.tmp").write_text("not ours\n", encoding="utf-8")
    # Deterministic mtime ordering: 2.attempt3 newest, 1.attempt1 oldest.
    os.utime(dirpath / "1.attempt1.log", (1000, 1000))
    os.utime(dirpath / "12.attempt1.log", (2000, 2000))
    os.utime(dirpath / "2.attempt3.log", (3000, 3000))


def _seed_jobs(db: Database) -> None:
    db.execute(
        """
        INSERT INTO jobs (job_id, task_key, stage, state, attempts, error_category,
                          created_at, updated_at)
        VALUES (1, "k1", "extract", "permanent_failed", 1, "ocr_failed", 0, 0),
               (2, "k2", "embed", "retryable_failed", 3, "embed_timeout", 0, 0)
        """
    )


def test_job_logs_list_annotates_and_orders(state_db: Database, tmp_path: Path) -> None:
    logs_dir = tmp_path / "job_logs"
    _make_job_logs(logs_dir)
    _seed_jobs(state_db)

    out = si.job_logs_list(state_db, logs_dir)
    assert out["count"] == 3
    names = [e["name"] for e in out["logs"]]
    assert names == ["2.attempt3.log", "12.attempt1.log", "1.attempt1.log"]

    by_name = {e["name"]: e for e in out["logs"]}
    j1 = by_name["1.attempt1.log"]
    assert j1["job_id"] == 1
    assert j1["state"] == "permanent_failed"
    assert j1["error_category"] == "ocr_failed"
    assert j1["stage"] == "extract"
    j2 = by_name["2.attempt3.log"]
    assert j2["attempt"] == 3
    assert j2["state"] == "retryable_failed"
    # The orphan file's job is not in the queue anymore.
    assert by_name["12.attempt1.log"]["state"] is None


def test_job_logs_list_empty_dir(state_db: Database, tmp_path: Path) -> None:
    out = si.job_logs_list(state_db, tmp_path / "job_logs")  # does not exist
    assert out["count"] == 0
    assert out["logs"] == []


def test_job_log_read_valid(state_db: Database, tmp_path: Path) -> None:
    logs_dir = tmp_path / "job_logs"
    _make_job_logs(logs_dir)
    out = si.job_log_read(logs_dir, "2.attempt3.log", lines=4000)
    assert out is not None
    assert out["line_count"] == 10
    assert out["lines"][0] == "job 2 kaboom"
    assert out["lines"][-1] == "job 2 kaboom"


def test_job_log_read_rejects_bad_names(tmp_path: Path) -> None:
    logs_dir = tmp_path / "job_logs"
    _make_job_logs(logs_dir)
    for name in ("notes.txt", "2.attempt3.log.tmp", "7.log", "1.attemptX.log",
                 "..%2Fevil.log", "1/attempt1.log", "1.attempt1.log/x"):
        assert si.job_log_read(logs_dir, name) is None, name


def test_job_log_read_missing_file(tmp_path: Path) -> None:
    logs_dir = tmp_path / "job_logs"
    _make_job_logs(logs_dir)
    assert si.job_log_read(logs_dir, "99.attempt1.log") is None


# --- service status ------------------------------------------------------------------


class _OKHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: object) -> None:
        pass  # keep the test output quiet


@pytest.fixture()
def live_port() -> Iterator[int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OKHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()
    server.server_close()


def _dead_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def test_service_status_up_and_down(base_config: Config, live_port: int) -> None:
    dead = _dead_port()
    cfg = base_config.model_copy(deep=True)
    cfg.services.answer_host = "127.0.0.1"
    cfg.services.answer_port = live_port
    cfg.answer.extra_endpoints = [AnswerEndpoint(host="127.0.0.1", port=dead)]
    cfg.services.embed_host = "127.0.0.1"
    cfg.services.embed_ports = [live_port, dead]

    out = si.service_status(cfg, timeout=2.0)
    assert [e["label"] for e in out["llm"]] == ["primary", "extra-1"]
    assert out["llm"][0]["up"] is True
    assert out["llm"][1]["up"] is False
    assert [e["label"] for e in out["embedders"]] == [f"embed-{live_port}", f"embed-{dead}"]
    assert out["embedders"][0]["up"] is True
    assert out["embedders"][1]["up"] is False
    for e in [*out["llm"], *out["embedders"]]:
        assert set(e) == {"label", "host", "port", "up", "detail"}


def test_service_status_reports_configured_single_embedder(base_config: Config) -> None:
    # Default pool is the single historical port 8081.
    out = si.service_status(base_config.model_copy(deep=True), timeout=1.0)
    assert [e["label"] for e in out["embedders"]] == [f"embed-{base_config.services.embed_ports[0]}"]
    assert len(out["llm"]) == 1
    assert out["llm"][0]["label"] == "primary"


# --- hardware --------------------------------------------------------------------


def test_gpu_status_shape() -> None:
    out = si.gpu_status()
    assert set(out) == {"available", "note", "gpus"}
    if out["available"]:
        for g in out["gpus"]:
            assert set(g) == {
                "index", "name", "mem_total_mib", "mem_used_mib",
                "mem_free_mib", "util_pct", "temp_c",
            }
            assert g["mem_used_mib"] <= g["mem_total_mib"]
    else:
        assert out["gpus"] == []


def test_cpu_status_shape() -> None:
    out = si.cpu_status()
    assert out["logical_cores"] >= 1
    for key in ("load1", "load5", "load15", "load_pct"):
        assert isinstance(out[key], float)
    assert set(out["ram"]) == {"total_bytes", "available_bytes", "used_pct"}
    assert 0.0 <= out["ram"]["used_pct"] <= 100.0


# --- /system/* routes -----------------------------------------------------------------

_TEXTS = ["The lighthouse keeper logged the fog.", "Fog horns sounded all night."]


@pytest.fixture()
def system_client(state_db: Database, base_config: Config) -> TestClient:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    model = FakeAnswerModel()
    publish_handbuilt(
        state_db, base_config, q,
        doc_id="docA", rev_id="revA", run_id="runA",
        texts=_TEXTS, title="Book A",
    )
    app = create_app(base_config, state_db, qdrant=q, embedder=emb, model=model)
    return TestClient(app)


def test_route_catalog(system_client: TestClient) -> None:
    r = system_client.get("/system/catalog")
    assert r.status_code == 200
    body: dict[str, Any] = r.json()
    assert body["books"] == 1
    assert body["processed_books"] == 1  # the hand-built rev has chunks
    assert body["published_books"] == 1  # publish_handbuilt activates it
    assert body["chunks"] == len(_TEXTS)
    assert body["disk_books"] == 0  # the fixture source root is not on disk
    assert body["db_size_bytes"] > 0


def test_route_serve_log_missing_then_present(system_client: TestClient, base_config: Config) -> None:
    r = system_client.get("/system/log")
    assert r.status_code == 200
    assert r.json()["exists"] is False

    log_path = base_config.paths.state_root / "logs" / "serve.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("first\nsecond\nthird\n", encoding="utf-8")
    r = system_client.get("/system/log", params={"lines": 2})
    assert r.json()["lines"] == ["second", "third"]


def test_route_job_logs(system_client: TestClient, state_db: Database, base_config: Config) -> None:
    logs_dir = base_config.paths.state_root / "job_logs"
    _make_job_logs(logs_dir)
    _seed_jobs(state_db)

    r = system_client.get("/system/job-logs")
    assert r.status_code == 200
    assert r.json()["count"] == 3

    r = system_client.get("/system/job-logs/1.attempt1.log")
    assert r.status_code == 200
    assert r.json()["lines"] == ["job 1 boom"]

    # Invalid names 404 rather than reading something.
    for name in ("notes.txt", "..%2Fevil.log"):
        assert system_client.get(f"/system/job-logs/{name}").status_code == 404


def test_route_status_shape(system_client: TestClient) -> None:
    r = system_client.get("/system/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body["services"]) == {"llm", "embedders"}
    assert isinstance(body["gpu"]["available"], bool)
    assert isinstance(body["cpu"]["load1"], float)
    assert "ram" in body["cpu"]


def test_route_shutdown_acks_then_signals(
    system_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("library_rag.api._self_terminate", lambda: calls.append("term"))
    r = system_client.post("/system/shutdown")
    assert r.status_code == 200
    assert r.json() == {"shutting_down": True}
    # TestClient runs BackgroundTasks before returning the response.
    assert calls == ["term"]


def test_system_routes_require_token_when_enforced(
    state_db: Database, base_config: Config
) -> None:
    # /system/* is not in _OPEN_PATHS: the bearer middleware covers it.
    base_config.services.api_token = "sekret"
    base_config.services.require_api_token = True
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    app = create_app(base_config, state_db, qdrant=q)
    client = TestClient(app)
    assert client.get("/system/catalog").status_code == 401
    assert client.post("/system/shutdown").status_code == 401
    assert client.get("/system/catalog",
                      headers={"Authorization": "Bearer sekret"}).status_code == 200
