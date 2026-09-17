"""Configuration loading and validation.

The configuration is a plain data model (Pydantic) loaded from a YAML file with
environment-variable overrides for secrets (e.g. an API token). Paths are the
operator-supplied roots described in the PRD; this module refuses unsafe
overlapping path layouts so we never write into a read-only source tree or nest
one managed root inside another.

This module is deliberately side-effect free: it only reads the config file and
validates it. It never touches the filesystem roots, so unit tests can exercise
the overlap rules with temp directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Sentinel marker used to detect whether a mount is actually present. See the
# mount-sentinel check in doctor and the startup guard (M1). A missing mount must
# never cause scans or writes to an empty fallback directory.
MOUNT_SENTINEL_ENV = "LIBRARY_RAG_CONFIG"


class ConfigError(Exception):
    """Raised when configuration is invalid.

    Deliberately not a ``ValueError``: pydantic converts exceptions raised
    inside validators that subclass ``ValueError`` (or ``AssertionError``) into
    a ``ValidationError``. Because the path-overlap rules raise ``ConfigError``
    from a model validator and we want that specific, understandable type to
    reach the caller (the tests and CLI both catch ``ConfigError``), it must be
    a plain ``Exception`` so pydantic lets it propagate untouched.
    """


class Paths(BaseModel):
    """Operator-supplied storage roots.

    - source_roots: read-only directories that are scanned for PDF/EPUB files.
    - archive_root: content-addressed store of authoritative originals (HDD).
    - artifact_root: page/section/chunk embedding artifacts (HDD).
    - state_root: SQLite state DB (SSD).
    - qdrant_root: Qdrant storage (SSD).
    - model_root: local model weights (GGUF, embeddings, reranker).
    - scratch_root: temporary/intermediate files; hard-capped (SSD).
    """

    source_roots: list[Path] = Field(default_factory=list)
    archive_root: Path
    artifact_root: Path
    state_root: Path
    qdrant_root: Path
    model_root: Path
    scratch_root: Path

    @field_validator(
        "archive_root",
        "artifact_root",
        "state_root",
        "qdrant_root",
        "model_root",
        "scratch_root",
        "source_roots",
    )
    @classmethod
    def _make_absolute(cls, v: Any) -> Any:
        # Normalize to absolute paths so overlap checks are stable. Relative paths
        # are resolved against the current working directory at load time.
        if isinstance(v, list):
            return [Path(item).expanduser().absolute() for item in v]
        return Path(v).expanduser().absolute()

    @model_validator(mode="after")
    def _check_overlap(self) -> Paths:
        # Managed roots are writable; source roots are read-only. The dangerous
        # layouts are: (a) a managed root nested inside a source root (we would
        # write into the read-only library), (b) a source root nested inside a
        # managed root (we could clobber the source), or (c) one managed root
        # nested inside another (scratch could swallow the DB, etc.).
        managed = {
            "archive_root": self.archive_root,
            "artifact_root": self.artifact_root,
            "state_root": self.state_root,
            "qdrant_root": self.qdrant_root,
            "model_root": self.model_root,
            "scratch_root": self.scratch_root,
        }
        # (c) managed roots must not nest inside each other.
        for name_a, path_a in managed.items():
            for name_b, path_b in managed.items():
                if name_a == name_b:
                    continue
                if _is_within(path_a, path_b):
                    raise ConfigError(
                        f"managed root {name_a}={path_a} is nested inside "
                        f"{name_b}={path_b}; managed roots must be disjoint"
                    )
        # (a) and (b) source vs managed.
        for src in self.source_roots:
            for name, mpath in managed.items():
                if _is_within(src, mpath):
                    raise ConfigError(
                        f"source root {src} is nested inside managed root "
                        f"{name}={mpath}; the source must not sit in a writable area"
                    )
                if _is_within(mpath, src):
                    raise ConfigError(
                        f"managed root {name}={mpath} is nested inside source "
                        f"root {src}; we must never write into the read-only source"
                    )
        # Duplicate source roots.
        seen: set[Path] = set()
        for src in self.source_roots:
            if src in seen:
                raise ConfigError(f"duplicate source root: {src}")
            seen.add(src)
        return self


def _is_within(candidate: Path, ancestor: Path) -> bool:
    """Return True if *candidate* is equal to or nested inside *ancestor*.

    Both paths are assumed to be absolute. Uses os.path.commonpath so symbolic
    link resolution is left to the caller (callers pass .absolute(), not
    .resolve(), to avoid surprises when the target does not exist yet).
    """
    try:
        common = os.path.commonpath([str(candidate), str(ancestor)])
    except ValueError:
        # Different drives / roots on Windows; cannot be nested.
        return False
    return common == str(ancestor)


class Services(BaseModel):
    """Network endpoints for locally-run services. All 127.0.0.1 by default."""

    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    # llama.cpp answer-model server (OpenAI-compatible).
    answer_host: str = "127.0.0.1"
    answer_port: int = 8080
    # FastAPI app bind address. Must stay loopback unless the operator opts in.
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    # Optional bearer token required before any non-loopback exposure.
    api_token: str | None = None

    @field_validator("app_host", "qdrant_host", "answer_host")
    @classmethod
    def _warn_public(cls, v: str) -> str:
        # We do not refuse a public bind (an operator may proxy), but this is the
        # only place a non-loopback host is surfaced, so record it explicitly.
        return v


class Config(BaseModel):
    """Top-level application configuration."""

    paths: Paths
    services: Services = Field(default_factory=Services)

    # Optional sentinel files that must exist to prove each mount is present.
    # Mapping of a human label to a file path that must exist. An empty value
    # means "no sentinel configured for this mount."
    mount_sentinels: dict[str, str] = Field(default_factory=dict)

    # Path to the operator's GPU-assignment map (UUID -> role). Populated by the
    # operator; see the hardware operating profiles (PRD §11).
    gpu_profiles: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sentinels(self) -> Config:
        for label, p in self.mount_sentinels.items():
            if p and not Path(p).is_absolute():
                raise ConfigError(f"mount sentinel for {label!r} must be absolute: {p!r}")
        return self


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate configuration from YAML (optional) and environment.

    Precedence: explicit *path* argument > $LIBRARY_RAG_CONFIG > defaults.
    Secrets (only api_token today) may also come from $LIBRARY_RAG_API_TOKEN.
    """
    cfg_path: Path | None = None
    if path is not None:
        cfg_path = Path(path)
    elif os.environ.get(MOUNT_SENTINEL_ENV):
        cfg_path = Path(os.environ[MOUNT_SENTINEL_ENV])

    raw: dict[str, Any] = {}
    if cfg_path is not None:
        if not cfg_path.is_file():
            raise ConfigError(f"config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if loaded is None:
                raw = {}
            elif not isinstance(loaded, dict):
                raise ConfigError(f"config root must be a mapping: {cfg_path}")
            else:
                raw = loaded

    # Environment overrides for secrets.
    services = raw.setdefault("services", {})
    if os.environ.get("LIBRARY_RAG_API_TOKEN") and not services.get("api_token"):
        services["api_token"] = os.environ["LIBRARY_RAG_API_TOKEN"]

    # Coerce to Config; pydantic raises ValidationError which we wrap so callers
    # see a single, understandable exception type.
    try:
        return Config.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError, ConfigError, etc.
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"invalid configuration: {exc}") from exc
