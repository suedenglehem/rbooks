"""Config loading and the path-overlap safety rules."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from library_rag.config import (
    AnswerEndpoint,
    AnswerSettings,
    Config,
    ConfigError,
    LoggingSettings,
    Paths,
    PilotSettings,
    Services,
    WorkerSettings,
    load_config,
)


def test_valid_config_is_accepted(base_config: Config) -> None:
    # A disjoint layout must validate cleanly.
    assert base_config.paths.state_root.name == "state"
    assert base_config.services.app_host == "127.0.0.1"
    # The source-path leak is opt-in, off by default.
    assert base_config.services.show_path_to_original is False
    # The bearer-token requirement is a master switch, off by default.
    assert base_config.services.require_api_token is False


def test_require_api_token_without_value_raises(
    roots: dict[str, Path], tmp_path: Path
) -> None:
    # Switch on with no value (neither in the file nor in the env) is
    # rejected at load time — it would enforce auth with no usable token.
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
                "services:",
                "  require_api_token: true",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="require_api_token"):
        load_config(cfg_file)


def test_require_api_token_with_value_is_accepted(
    roots: dict[str, Path], tmp_path: Path
) -> None:
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
                "services:",
                "  api_token: fixed-local-value",
                "  require_api_token: true",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.services.require_api_token is True
    assert cfg.services.api_token == "fixed-local-value"


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


def test_pilot_page_cap_default_and_validation() -> None:
    # None means "no cap" — the production default, which keeps extraction
    # keys byte-identical to pre-M6 ones.
    assert PilotSettings().page_cap is None
    assert PilotSettings(page_cap=32).page_cap == 32
    with pytest.raises(ConfigError, match=r"pilot\.page_cap must be >= 1"):
        PilotSettings(page_cap=0)
    with pytest.raises(ConfigError, match=r"pilot\.page_cap must be >= 1"):
        PilotSettings(page_cap=-1)


def test_services_qdrant_path_defaults_none() -> None:
    # No qdrant_path by default: operators on a Qdrant server set host/port,
    # and only the pilot sandbox switches on embedded local mode.
    assert Services().qdrant_path is None


def test_services_embed_ports_default_and_validation() -> None:
    # Empty pool by default: single-endpoint behavior via embed_port.
    assert Services().embed_ports == []
    assert Services(embed_ports=[8081, 8082]).embed_ports == [8081, 8082]
    with pytest.raises(ConfigError, match=r"port must be 1-65535"):
        Services(embed_port=0)
    with pytest.raises(ConfigError, match=r"port must be 1-65535"):
        Services(embed_ports=[8081, 70000])


def test_logging_settings_defaults() -> None:
    ls = LoggingSettings()
    assert ls.level == "INFO"
    assert ls.format == "human"
    assert ls.job_log_retention == 500


def test_config_default_logging(base_config: Config) -> None:
    # A config with no logging: section gets the built-in defaults.
    assert base_config.logging.level == "INFO"
    assert base_config.logging.format == "human"
    assert base_config.logging.job_log_retention == 500


def test_logging_level_is_normalized_to_uppercase() -> None:
    # Operators write "debug" / "info" in YAML; the stored value is canonical.
    assert LoggingSettings(level="debug").level == "DEBUG"
    assert LoggingSettings(level="warning").level == "WARNING"
    assert LoggingSettings(level="error").level == "ERROR"


def test_logging_invalid_values_raise() -> None:
    with pytest.raises(ConfigError, match=r"logging\.level"):
        LoggingSettings(level="LOUD")
    with pytest.raises(ConfigError, match=r"logging\.format"):
        LoggingSettings(format="xml")
    with pytest.raises(ConfigError, match=r"logging\.job_log_retention"):
        LoggingSettings(job_log_retention=0)
    with pytest.raises(ConfigError, match=r"logging\.job_log_retention"):
        LoggingSettings(job_log_retention=-3)


def test_load_config_logging_from_file(roots: dict[str, Path], tmp_path: Path) -> None:
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
                "logging:",
                "  level: debug",
                "  format: json",
                "  job_log_retention: 42",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.logging.level == "DEBUG"
    assert cfg.logging.format == "json"
    assert cfg.logging.job_log_retention == 42


def test_load_config_invalid_logging_raises(roots: dict[str, Path], tmp_path: Path) -> None:
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
                "logging:",
                "  level: LOUD",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"logging\.level"):
        load_config(cfg_file)


# --- M9: answer.extra_endpoints + worker ----------------------------------------


def test_answer_extra_endpoints_default_and_parse() -> None:
    assert AnswerSettings().extra_endpoints == []
    # The dict form is the YAML shape; pydantic validates it into models.
    s = AnswerSettings(
        model_revision="qwen3.8-27b-fp8@vllm-dual-max",
        extra_endpoints=cast("list[AnswerEndpoint]", [{"host": "ak", "port": 8080}]),
    )
    assert s.extra_endpoints == [AnswerEndpoint(host="ak", port=8080)]
    # model_name inherits the primary's when omitted.
    assert s.extra_endpoints[0].model_name is None


def test_answer_extra_endpoints_invalid_raise() -> None:
    with pytest.raises(ConfigError, match=r"extra_endpoints"):
        AnswerEndpoint(host="", port=80)
    with pytest.raises(ConfigError, match=r"extra_endpoints"):
        AnswerEndpoint(host="ak", port=0)


def test_worker_settings_default_and_validation() -> None:
    assert WorkerSettings().max_concurrent_jobs == 1
    with pytest.raises(ConfigError, match=r"max_concurrent_jobs"):
        WorkerSettings(max_concurrent_jobs=0)


def test_load_config_answer_pool_and_worker(
    roots: dict[str, Path], tmp_path: Path
) -> None:
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
                "answer:",
                "  model_revision: qwen3.8-27b-fp8@vllm-dual-max",
                "  model_name: qwen3.8-27b",
                "  extra_endpoints:",
                "    - host: ak",
                "      port: 8080",
                "worker:",
                "  max_concurrent_jobs: 2",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.answer.extra_endpoints == [AnswerEndpoint(host="ak", port=8080)]
    assert cfg.worker.max_concurrent_jobs == 2
