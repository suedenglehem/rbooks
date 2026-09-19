"""M5 CLI answer (PRD §12): cited answering from the command line over a
published fake library — the JSON contract, the human text output, and the
two clean failure modes (unconfigured model, unreachable index) with the
exit-code split of answered/abstained = 0, failed/unavailable = 1.
"""

from __future__ import annotations

import json

import pytest

from fixtures import publish_handbuilt
from library_rag.cli import EXIT_ERROR, EXIT_OK, main
from library_rag.config import Config
from library_rag.db import Database
from library_rag.indexing import FakeQdrant

_TEXTS_A = [
    "The zebra grazes on the open savannah at dawn.",
    "Striped zebras run in tight family herds.",
    "A zebra's stripes are unique, like fingerprints.",
    "Zebra mule hybrids are called zorses.",
    "Plain zebra calves nurse within hours of birth.",
    "Equus quagga is the common name for the zebra.",
]


Library = tuple[Database, Config, FakeQdrant]


class _DownQdrant(FakeQdrant):
    def ping(self) -> bool:
        return False


@pytest.fixture()
def library(state_db: Database, base_config: Config) -> Library:
    base_config.embedding.fake = True
    q = FakeQdrant(base_config.embedding.dimensions)
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
    return state_db, base_config, q


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    cfg: Config,
    q: FakeQdrant,
) -> None:
    monkeypatch.setattr("library_rag.cli.load_config", lambda _path=None: cfg)

    def _fake(cfg_: Config) -> FakeQdrant:
        return q

    monkeypatch.setattr("library_rag.cli.RealQdrantOps", _fake)


def test_cli_answer_json_contract(
    library: Library,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, cfg, q = library
    cfg.answer.fake = True
    _wire(monkeypatch, cfg, q)
    rc = main(["answer", "zebra stripes", "--json"])
    assert rc == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "answered"
    assert data["citations"] == ["E1"]
    assert data["answer_text"]
    assert len(data["evidence"]) >= 1


def test_cli_answer_text_output(
    library: Library,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, cfg, q = library
    cfg.answer.fake = True
    _wire(monkeypatch, cfg, q)
    rc = main(["answer", "zebra stripes"])
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("status: answered  (answer ")
    assert "citations: E1" in out
    # The evidence lines carry the frozen snapshot source.
    assert "E1:" in out


def test_cli_answer_unconfigured_model_fails(
    library: Library,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, cfg, q = library
    # answer.fake unset and no model_revision: make_answer_model yields None.
    _wire(monkeypatch, cfg, q)
    rc = main(["answer", "zebra stripes"])
    assert rc == EXIT_ERROR
    out = capsys.readouterr().out
    assert "status: failed" in out
    assert "failed: answer model not configured (set answer.model_revision)" in out


def test_cli_answer_qdrant_down(
    library: Library,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, cfg, _ = library
    cfg.answer.fake = True
    dims = cfg.embedding.dimensions

    def _down(cfg_: Config) -> FakeQdrant:
        return _DownQdrant(dims)

    monkeypatch.setattr("library_rag.cli.load_config", lambda _path=None: cfg)
    monkeypatch.setattr("library_rag.cli.RealQdrantOps", _down)
    rc = main(["answer", "zebra stripes"])
    assert rc == EXIT_ERROR
    captured = capsys.readouterr()
    assert "Qdrant is unreachable" in captured.err
    assert "status:" not in captured.out
