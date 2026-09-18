"""M4 CLI search: an unreachable index is a clean non-zero exit with a
diagnostic — the CLI never fabricates results (PRD §8F/§9, M4 gate)."""

from __future__ import annotations

import pytest

from library_rag.cli import EXIT_ERROR, main
from library_rag.config import Config
from library_rag.indexing import FakeQdrant


class _DownQdrant(FakeQdrant):
    def ping(self) -> bool:
        return False


def test_cli_search_fails_cleanly_when_qdrant_down(
    base_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_config.embedding.fake = True
    dims = base_config.embedding.dimensions
    monkeypatch.setattr("library_rag.cli.load_config", lambda _path=None: base_config)

    def _down(cfg: Config) -> FakeQdrant:
        return _DownQdrant(dims)

    monkeypatch.setattr("library_rag.cli.RealQdrantOps", _down)

    rc = main(["search", "zebra"])
    assert rc == EXIT_ERROR
    captured = capsys.readouterr()
    assert "Qdrant is unreachable" in captured.err
    # No fabricated results on stdout.
    assert "zebra" not in captured.out
