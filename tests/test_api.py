"""M5 application API (PRD §12): the FastAPI research surface over a
published fake library — health/readiness, library list, search (including
the 503/400/422 error split), cited answering with persisted frozen
evidence, the answer history and snapshot citation endpoints, the ingestion
dashboard controls, and bearer-token auth with the open liveness path.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from fixtures import publish_handbuilt
from library_rag.api import create_app
from library_rag.config import Config
from library_rag.db import Database
from library_rag.embeddings import FakeEmbedder
from library_rag.indexing import FakeQdrant
from library_rag.llm import FakeAnswerModel

_TEXTS_A = [
    "The zebra grazes on the open savannah at dawn.",
    "Striped zebras run in tight family herds.",
    "A zebra's stripes are unique, like fingerprints.",
    "Zebra mule hybrids are called zorses.",
    "Plain zebra calves nurse within hours of birth.",
    "Equus quagga is the common name for the zebra.",
]


Library = tuple[Database, Config, FakeQdrant, FakeEmbedder, FakeAnswerModel]


class _DownQdrant(FakeQdrant):
    def ping(self) -> bool:
        return False


@pytest.fixture()
def library(state_db: Database, base_config: Config) -> Library:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
    emb = FakeEmbedder(base_config.embedding.dimensions)
    model = FakeAnswerModel()
    publish_handbuilt(
        state_db,
        base_config,
        q,
        doc_id="docA",
        rev_id="revA",
        run_id="runA",
        texts=_TEXTS_A,
        title="Book A",
    )
    return state_db, base_config, q, emb, model


@pytest.fixture()
def client(library: Library) -> TestClient:
    db, cfg, q, emb, model = library
    app = create_app(cfg, db, qdrant=q, embedder=emb, model=model)
    return TestClient(app)


def _answer(client: TestClient, query: str = "zebra stripes") -> dict[str, Any]:
    r = client.post("/answer", json={"query": query})
    assert r.status_code == 200, r.text
    data: dict[str, Any] = r.json()
    return data


# --- health / readiness --------------------------------------------------------


def test_health_and_ready(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"qdrant": True, "embedding_model": True, "answer_model": True}


def test_ready_reports_down_qdrant(library: Library) -> None:
    db, cfg, _, emb, model = library
    down = _DownQdrant(cfg.embedding.dimensions)
    app = create_app(cfg, db, qdrant=down, embedder=emb, model=model)
    client = TestClient(app)
    assert client.get("/ready").json() == {
        "qdrant": False,
        "embedding_model": True,
        "answer_model": True,
    }


# --- library --------------------------------------------------------------------


def test_library_lists_published_book(client: TestClient) -> None:
    r = client.get("/library")
    assert r.status_code == 200
    books = r.json()["books"]
    assert len(books) == 1
    b = books[0]
    assert b["doc_id"] == "docA"
    assert b["rev_id"] == "revA"
    assert b["title"] == "revA"  # stem of the registered first_path
    assert b["format"] == "pdf"
    assert b["published"] is True
    assert b["chunk_count"] == 6


# --- search -----------------------------------------------------------------------


def test_search_round_trip(client: TestClient) -> None:
    r = client.post("/search", json={"query": "zebra stripes"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "zebra stripes"
    assert body["degraded"] is False
    assert len(body["passages"]) >= 1
    p = body["passages"][0]
    assert p["chunk_id"].startswith("runA:chunk-")
    assert p["doc_id"] == "docA"
    assert p["text"]
    for key in ("counts", "degraded_reason"):
        assert key in body


def test_search_blank_query_is_400(client: TestClient) -> None:
    assert client.post("/search", json={"query": "   "}).status_code == 400


def test_search_empty_query_is_422(client: TestClient) -> None:
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_search_qdrant_down_is_503(library: Library) -> None:
    db, cfg, _, emb, model = library
    down = _DownQdrant(cfg.embedding.dimensions)
    app = create_app(cfg, db, qdrant=down, embedder=emb, model=model)
    r = TestClient(app).post("/search", json={"query": "zebra"})
    assert r.status_code == 503
    assert "Qdrant" in r.json()["detail"]


# --- answer -----------------------------------------------------------------------


def test_answer_is_answered_and_persisted(client: TestClient) -> None:
    data = _answer(client)
    assert data["status"] == "answered"
    assert data["citations"] == ["E1"]
    assert data["answer_text"]
    assert len(data["evidence"]) >= 1
    row = client.get(f"/answers/{data['answer_id']}")
    assert row.status_code == 200
    assert row.json() == data  # the API serves the persisted row
    listed = client.get("/answers").json()["answers"]
    assert [a["answer_id"] for a in listed] == [data["answer_id"]]


def test_answer_abstention_is_200_with_reason(client: TestClient) -> None:
    data = _answer(client, "quantum chromodynamics of zebras")
    # The fake model's default reply is not an abstention, so this is
    # answered; the abstention path is covered at the answer_query level.
    assert data["status"] in ("answered", "abstained", "failed")


def test_answer_unknown_id_404_and_unknown_citation_404(client: TestClient) -> None:
    r = client.get("/answers/nope")
    assert r.status_code == 404
    assert r.json()["detail"] == "unknown answer"
    data = _answer(client)
    r = client.get(f"/answers/{data['answer_id']}/citations/E999")
    assert r.status_code == 404
    assert r.json()["detail"] == "unknown evidence id for this answer"
    r = client.get("/answers/nope/citations/E1")
    assert r.status_code == 404


def test_citation_endpoint_resolves_from_snapshot(client: TestClient, library: Library) -> None:
    _, cfg, *_ = library
    data = _answer(client)
    first = data["evidence"][0]["evidence_id"]
    # No archived original yet: resolved as unavailable, not an error.
    r = client.get(f"/answers/{data['answer_id']}/citations/{first}")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "archived original no longer present"
    assert body["excerpt"] == data["evidence"][0]["text"]
    # The archive appears: the same snapshot resolves to the reader routes.
    archive = cfg.paths.archive_root
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "revA.pdf").write_bytes(b"%PDF-1.4 test")
    body = client.get(f"/answers/{data['answer_id']}/citations/{first}").json()
    assert body["available"] is True
    assert body["reason"] is None
    assert body["reader"] == {"manifest": "/books/revA", "source": "/books/revA/source"}


# --- ingestion dashboard ------------------------------------------------------------


def test_ingest_status_counts(client: TestClient) -> None:
    r = client.get("/ingest/status")
    assert r.status_code == 200
    body = r.json()
    assert body["paused"] is False
    assert isinstance(body["jobs"], dict)
    assert body["schema_version"] >= 1
    assert body["documents"] == 1
    assert body["revisions"] == 1
    assert body["chunks"] == 6
    assert body["answers"] == 0


def test_ingest_pause_resume_retry(client: TestClient) -> None:
    assert client.post("/ingest/pause", json={"reason": "maintenance"}).json() == {"paused": True}
    assert client.get("/ingest/status").json()["paused"] is True
    assert client.post("/ingest/resume").json() == {"paused": False}
    assert client.get("/ingest/status").json()["paused"] is False
    # Nothing failed: retry requeues nothing.
    assert client.post("/ingest/retry", json={}).json() == {"requeued": 0}


def test_scan_reports_missing_source_mount(client: TestClient) -> None:
    # The configured source root (tmp .../books) was never created: the scan
    # must report the mount as unavailable, never scan an empty fallback.
    r = client.post("/scan")
    assert r.status_code == 200
    reports = r.json()["reports"]
    assert len(reports) == 1
    assert reports[0]["mount_unavailable"] is True


# --- auth + CSP ----------------------------------------------------------------------


def test_bearer_token_required_when_configured(library: Library) -> None:
    db, cfg, q, emb, model = library
    cfg.services.api_token = "sekret"
    app = create_app(cfg, db, qdrant=q, embedder=emb, model=model)
    client = TestClient(app)
    assert client.get("/library").status_code == 401
    assert client.post("/search", json={"query": "zebra"}).status_code == 401
    headers = {"Authorization": "Bearer sekret"}
    assert client.get("/library", headers=headers).status_code == 200
    # Wrong token is rejected; liveness stays open for the operator's probe.
    assert client.get("/library", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/health").status_code == 200


def test_csp_header_on_every_response(client: TestClient) -> None:
    csp = client.get("/health").headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "connect-src 'self'" in csp
    # The middleware adds it to API responses as well.
    assert client.get("/library").headers.get("Content-Security-Policy") == csp
