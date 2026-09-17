"""Shared test fixtures.

Everything here is local and side-effect free: temp directories stand in for the
operator's storage roots, so the path-overlap rules and doctor probes can be
exercised without touching a real book library.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from library_rag.config import Config, Paths, Services


@pytest.fixture
def roots(tmp_path: Path) -> dict[str, Path]:
    """A set of disjoint temp directories usable as the six managed roots."""
    return {
        "archive_root": tmp_path / "archive",
        "artifact_root": tmp_path / "artifacts",
        "state_root": tmp_path / "state",
        "qdrant_root": tmp_path / "qdrant",
        "model_root": tmp_path / "models",
        "scratch_root": tmp_path / "scratch",
    }


@pytest.fixture
def base_paths(roots: dict[str, Path]) -> Paths:
    """A valid Paths instance over the temp roots (disjoint, empty source_roots)."""
    return Paths(source_roots=[], **roots)  # type: ignore[arg-type]


@pytest.fixture
def base_config(roots: dict[str, Path]) -> Config:
    """A valid top-level Config over the temp roots."""
    return Config(
        paths=Paths(
            source_roots=[tmp_path_src(roots)],
            archive_root=roots["archive_root"],
            artifact_root=roots["artifact_root"],
            state_root=roots["state_root"],
            qdrant_root=roots["qdrant_root"],
            model_root=roots["model_root"],
            scratch_root=roots["scratch_root"],
        ),
        services=Services(),
    )


def tmp_path_src(roots: dict[str, Path]) -> Path:
    # A source root that is a sibling of the managed roots, never nested in them.
    return roots["state_root"].parent / "books"
