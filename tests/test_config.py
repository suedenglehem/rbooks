"""Config loading and the path-overlap safety rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from library_rag.config import Config, ConfigError, Paths, load_config


def test_valid_config_is_accepted(base_config: Config) -> None:
    # A disjoint layout must validate cleanly.
    assert base_config.paths.state_root.name == "state"
    assert base_config.services.app_host == "127.0.0.1"


def test_managed_root_nested_in_source_is_refused(roots: dict[str, Path]) -> None:
    src = roots["state_root"].parent
    with pytest.raises(ConfigError, match="nested inside source"):
        Paths(
            source_roots=[src],
            archive_root=roots["archive_root"],
            artifact_root=roots["artifact_root"],
            state_root=roots["state_root"],  # state_root sits under src
            qdrant_root=roots["qdrant_root"],
            model_root=roots["model_root"],
            scratch_root=roots["scratch_root"],
        )


def test_source_root_nested_in_managed_is_refused(roots: dict[str, Path]) -> None:
    with pytest.raises(ConfigError, match=r"source root .* nested inside managed"):
        Paths(
            source_roots=[roots["state_root"] / "books"],  # books under state_root
            archive_root=roots["archive_root"],
            artifact_root=roots["artifact_root"],
            state_root=roots["state_root"],
            qdrant_root=roots["qdrant_root"],
            model_root=roots["model_root"],
            scratch_root=roots["scratch_root"],
        )


def test_one_managed_root_inside_another_is_refused(roots: dict[str, Path]) -> None:
    with pytest.raises(ConfigError, match="managed roots must be disjoint"):
        Paths(
            source_roots=[],
            archive_root=roots["archive_root"],
            artifact_root=roots["artifact_root"],
            state_root=roots["state_root"],
            qdrant_root=roots["qdrant_root"],
            model_root=roots["model_root"],
            scratch_root=roots["state_root"] / "scratch",  # scratch under state
        )


def test_duplicate_source_roots_are_refused(roots: dict[str, Path]) -> None:
    src = roots["state_root"].parent / "books"
    with pytest.raises(ConfigError, match="duplicate source root"):
        Paths(
            source_roots=[src, src],
            archive_root=roots["archive_root"],
            artifact_root=roots["artifact_root"],
            state_root=roots["state_root"],
            qdrant_root=roots["qdrant_root"],
            model_root=roots["model_root"],
            scratch_root=roots["scratch_root"],
        )


def test_relative_paths_are_absolutized(tmp_path: Path) -> None:
    p = Paths(
        source_roots=[],
        archive_root=Path("archive"),
        artifact_root=Path("artifacts"),
        state_root=Path("state"),
        qdrant_root=Path("qdrant"),
        model_root=Path("models"),
        scratch_root=Path("scratch"),
    )
    assert p.archive_root.is_absolute()


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "does-not-exist.yaml")


def test_load_config_from_file(roots: dict[str, Path], tmp_path: Path) -> None:
    src = roots["state_root"].parent / "books"
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "paths:",
                "  source_roots:",
                f"    - {src}",
                f"  archive_root: {roots['archive_root']}",
                f"  artifact_root: {roots['artifact_root']}",
                f"  state_root: {roots['state_root']}",
                f"  qdrant_root: {roots['qdrant_root']}",
                f"  model_root: {roots['model_root']}",
                f"  scratch_root: {roots['scratch_root']}",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.paths.source_roots == [src]
    assert cfg.services.qdrant_port == 6333


def test_api_token_env_override(
    roots: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIBRARY_RAG_API_TOKEN", "secret-token")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "paths:",
                "  source_roots: []",
                f"  archive_root: {roots['archive_root']}",
                f"  artifact_root: {roots['artifact_root']}",
                f"  state_root: {roots['state_root']}",
                f"  qdrant_root: {roots['qdrant_root']}",
                f"  model_root: {roots['model_root']}",
                f"  scratch_root: {roots['scratch_root']}",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.services.api_token == "secret-token"


def test_invalid_root_type_raises_config_error(tmp_path: Path) -> None:
    # A YAML mapping that is not a mapping at the root is a ConfigError, not a
    # raw pydantic ValidationError leaking to the caller.
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="config root must be a mapping"):
        load_config(bad)
