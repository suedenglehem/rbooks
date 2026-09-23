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

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

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
    - backup_root: optional default destination for `library-rag backup` (M7).
      Backups on internal storage recover accidental loss; they are not
      independent disaster backups (PRD §14).
    """

    source_roots: list[Path] = Field(default_factory=list)
    archive_root: Path
    artifact_root: Path
    state_root: Path
    qdrant_root: Path
    model_root: Path
    scratch_root: Path
    backup_root: Path | None = None

    @field_validator(
        "archive_root",
        "artifact_root",
        "state_root",
        "qdrant_root",
        "model_root",
        "scratch_root",
        "backup_root",
        "source_roots",
    )
    @classmethod
    def _make_absolute(cls, v: Any) -> Any:
        # Normalize to absolute paths so overlap checks are stable. Relative paths
        # are resolved against the current working directory at load time.
        if v is None:
            return None
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
        managed: dict[str, Path] = {}
        for name in (
            "archive_root",
            "artifact_root",
            "state_root",
            "qdrant_root",
            "model_root",
            "scratch_root",
            "backup_root",
        ):
            path = getattr(self, name)
            if path is not None:
                managed[name] = path
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
    # M6: embedded Qdrant storage directory. When set, the index runs in
    # qdrant-client local mode under this directory and qdrant_host/port are
    # ignored — no Qdrant server process is needed (pilot sandboxes use this).
    # The directory is a *managed* storage area owned by the client process.
    qdrant_path: str | None = None
    # llama.cpp answer-model server (OpenAI-compatible /v1/chat/completions).
    # One model per server, so the embedding model has its own endpoint.
    answer_host: str = "127.0.0.1"
    answer_port: int = 8080
    # llama.cpp embedding-model server (OpenAI-compatible /v1/embeddings).
    embed_host: str = "127.0.0.1"
    embed_port: int = 8081
    # Optional pool of identical embedding-model replicas (one llama-server
    # per port). When non-empty, embedding requests round-robin across these
    # ports and fail over to the next on transient unavailability; embed_port
    # is then ignored. Empty keeps the single-embed_port behavior.
    embed_ports: list[int] = Field(default_factory=list)
    # FastAPI app bind address. Must stay loopback unless the operator opts in.
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    # Bearer token VALUE, used only while require_api_token is true. May also
    # come from $LIBRARY_RAG_API_TOKEN (env wins over this field).
    api_token: str | None = None
    # Master switch for the bearer-token requirement. Off by default: no
    # request needs a token and the web UI never shows its token field —
    # the default for local-machine runs. When true, every non-static
    # request needs Authorization: Bearer <api_token>, and the web UI shows
    # the token field (on the first 401).
    require_api_token: bool = False
    # Local-machine convenience (operator opt-in): when True, the reader
    # manifest (GET /books/{rev_id}) carries the book's full source path,
    # which the web UI shows in the reader pane next to the content. Off by
    # default — a deployed app must not leak server filesystem layout.
    show_path_to_original: bool = False
    # Connect-phase timeout (seconds) for the LLM answer endpoints (the
    # answer model and the resume pool). Deliberately separate from the
    # generation timeouts (``answer.timeout_seconds`` /
    # ``resume.timeout_seconds``), which bound the whole request: a LAN
    # connect is sub-millisecond, so this only caps how long a dead or
    # packet-dropping endpoint (e.g. a stopped llama-server behind a
    # stateful firewall) can pin a pool thread before failover kicks in.
    connect_timeout_seconds: float = 2.0

    @field_validator("app_host", "qdrant_host", "answer_host", "embed_host")
    @classmethod
    def _warn_public(cls, v: str) -> str:
        # We do not refuse a public bind (an operator may proxy), but this is the
        # only place a non-loopback host is surfaced, so record it explicitly.
        return v

    @field_validator("embed_port", "embed_ports")
    @classmethod
    def _valid_ports(cls, v: int | list[int]) -> int | list[int]:
        ports = v if isinstance(v, list) else [v]
        for port in ports:
            if not 1 <= port <= 65535:
                raise ConfigError(f"port must be 1-65535 (got {port})")
        return v

    @field_validator("connect_timeout_seconds")
    @classmethod
    def _valid_connect_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ConfigError("services.connect_timeout_seconds must be positive")
        return v


class ScanSettings(BaseModel):
    """Streaming-discovery limits (PRD §8A): ignored names and symlink policy."""

    ignore_dirs: set[str] = Field(
        default_factory=lambda: {
            "__pycache__", ".git", ".hg", ".svn", ".cache", ".idea", "node_modules",
        }
    )
    ignore_files: set[str] = Field(default_factory=lambda: {".DS_Store", "Thumbs.db", "desktop.ini"})
    # Symlinked files/directories are never followed: a link out of the source
    # tree would escape the read-only boundary (PRD §8A "constrain symlink
    # traversal").
    follow_symlinks: bool = False


class PdfLimits(BaseModel):
    """PDF extraction quality-assessment thresholds (PRD §8B).

    M2 records these decisions per page; M3 routes on them (OCR selection).
    """

    # Fewer non-whitespace characters than this => "sparse" (cover/illustration).
    sparse_chars: int = 40
    # Share of replacement characters (U+FFFD) above this => "malformed" layer.
    max_replacement_ratio: float = 0.05


class EpubLimits(BaseModel):
    """EPUB archive safety limits (PRD §8D): ZIP-bomb / traversal defenses."""

    max_entries: int = 10_000
    max_uncompressed_bytes: int = 2 * (1024 ** 3)
    # Uncompressed/Compressed ratio above this is treated as a compression bomb.
    max_compression_ratio: float = 100.0


class OcrSettings(BaseModel):
    """Selective-OCR engine settings (PRD §8C).

    Only pages routed to OCR by :mod:`library_rag.extraction.routing` are
    rasterized; the raster is a bounded scratch file, never archived.
    """

    # Tesseract binary; ``tesseract`` is resolved on PATH.
    bin: str = "tesseract"
    # Explicit language pack(s) passed as ``-l``; never rely on the host default.
    languages: list[str] = Field(default_factory=lambda: ["eng"])
    # Page segmentation mode (3 = fully automatic, no OSD on plain pages).
    psm: int = 3
    # Rasterization DPI; the pixmap is capped by max_side_px regardless.
    dpi: int = 300
    # Hard cap on the longest pixmap side in pixels (memory bound, PRD §8C).
    max_side_px: int = 16_000
    # Subprocess timeout for one page.
    timeout_seconds: float = 600.0
    # A no-text/sparse page is OCR'd only when >= this share of the page area
    # is covered by embedded images (below it the page is a legit blank or
    # illustration and OCR would only add noise).
    image_area_threshold: float = 0.5

    def settings_sha(self) -> str:
        """Canonical hash of the engine settings, recorded in each OCR artifact
        so a replay can detect an already-OCR'd unit without re-running the
        (expensive) engine. Part of the parent :class:`ExtractionSettings` hash
        too, via the usual field serialization."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class NormalizationSettings(BaseModel):
    """Searchable-text normalization rules (PRD §8E).

    Normalization is recorded per unit *in the unit artifact* together with
    character-level source-span mappings, so every searchable character is
    quotable back to the source text (quotations always come from the source).
    """

    collapse_whitespace: bool = True
    # Join words split by a line-end hyphen ("de-\nline") into "de line"→"deline".
    dehyphenate: bool = True
    # Remove a repeated first/last line (running header/footer). Only applied
    # when the line recurs across units of the same book (pages, not sections).
    remove_repeated_headers: bool = True
    # A repeated line counts as header/footer only when it occurs on at least
    # this many units...
    header_min_pages: int = 2
    # ...and its stripped length is within this range (filters page-number-only
    # noise at the low end, full paragraphs at the high end).
    header_min_chars: int = 3
    header_max_chars: int = 80

    def settings_sha(self) -> str:
        """Canonical hash of the normalization rules; feeds the chunk pipeline
        fingerprint so a rule change re-chunks (and re-normalizes) without
        re-extracting or re-OCR-ing."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ChunkingSettings(BaseModel):
    """Chunking settings (PRD §8E).

    Kept out of :class:`ExtractionSettings` on purpose: chunking re-runs are
    cheap (no re-parse, no re-OCR) and must not invalidate the whole
    extraction run. ``settings_sha`` feeds the pipeline fingerprint so a
    change here re-chunks without re-extracting.
    """

    target_tokens: int = 600
    overlap_tokens: int = 100
    # Absolute hard cap; never exceeded, even for a single enormous token.
    max_tokens: int = 2_048
    # "words" is the deterministic development tokenizer; a BPE name (e.g.
    # "bge-m3") is resolved lazily at chunk time (M4) and a missing model is a
    # permanent ``tokenizer_missing`` failure, never silent word fallback.
    tokenizer: str = "words"
    tokenizer_path: str | None = None
    # Prefix EPUB section titles onto their first chunk (counted against the
    # token budget, excluded from the chunk text/spans).
    title_prefix: bool = True

    def settings_sha(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _check_ranges(self) -> ChunkingSettings:
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.target_tokens:
            raise ValueError(
                f"overlap_tokens must satisfy 0 <= overlap_tokens < target_tokens "
                f"({self.overlap_tokens} !< {self.target_tokens})"
            )
        if self.target_tokens > self.max_tokens:
            raise ValueError(
                f"target_tokens must be <= max_tokens ({self.target_tokens} > {self.max_tokens})"
            )
        return self


class ExtractionSettings(BaseModel):
    """All stage settings that define an extraction run (PRD §6).

    ``settings_sha`` is the canonical hash of this model: any change produces a
    new extraction key and therefore fresh, non-clobbering outputs.
    """

    pdf: PdfLimits = Field(default_factory=PdfLimits)
    epub: EpubLimits = Field(default_factory=EpubLimits)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)

    def settings_sha(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EmbeddingSettings(BaseModel):
    """Embedding model + encoding settings (PRD §6/§8F).

    ``settings_sha()`` is the "encoding configuration" of the embedding key:
    any change here (model revision, dimensions, dtype, normalization, batch
    policy) produces a new embedding key and therefore fresh checkpoints —
    while extraction and chunking stay valid.

    The deterministic fake embedder is for tests only: ``fake`` defaults to
    False and production configs must never set it; when it is set, the
    resolved model revision is the explicit label ``fake-v1`` so any artifact
    it touches is identifiable as non-real.
    """

    fake: bool = False
    # Required unless ``fake``: the exact model + revision, e.g.
    # "bge-m3@561cab6a4299" — never a bare family name.
    model_revision: str | None = None
    # The server-side "model" field for /v1/embeddings (llama.cpp usually accepts
    # the loaded model's name/alias). Defaults to model_revision. Distinct
    # servers/models produce distinct values, so it is part of the embedding key.
    model_name: str | None = None
    dimensions: int = 1024
    dtype: str = "float32"
    # L2-normalize vectors (BGE-M3 expects normalized cosine inputs).
    normalize: bool = True
    # Bounded batch for encoding; OOM policy halves it (PRD §8F).
    batch_size: int = 8
    oom_max_halvings: int = 4

    def settings_sha(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _check_model(self) -> EmbeddingSettings:
        if self.dimensions <= 0 or self.batch_size <= 0:
            raise ConfigError("embedding dimensions and batch_size must be positive")
        if self.oom_max_halvings < 0:
            raise ConfigError("embedding.oom_max_halvings must be >= 0")
        return self

    @property
    def is_configured(self) -> bool:
        """True when an embedder can be constructed.

        Staged rollouts are a first-class case: an operator runs scan/extract/
        chunk before the embedding model is installed, so the *config* stays
        valid with no model. What is forbidden by default is the fake embedder
        (``fake`` defaults to False); commands that need a model (search, the
        embed stage) raise an explicit :class:`ConfigError` when this is False.
        """
        return self.fake or self.model_revision is not None

    @property
    def effective_revision(self) -> str:
        """The revision persisted on checkpoints/generations (labeled for fake).

        Raises :class:`ConfigError` when no real model is configured — callers
        must check :attr:`is_configured` before reaching for this.
        """
        if self.fake:
            return "fake-v1"
        if self.model_revision is None:
            raise ConfigError(
                "embedding.model_revision is not set; configure the embedding "
                "model (or embedding.fake: true in tests only) before embedding"
            )
        return self.model_revision


class Bm25Settings(BaseModel):
    """Client-side BM25 parameters for the sparse vectors (PRD §8F/§9)."""

    k1: float = 1.5
    b: float = 0.75

    def settings_sha(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RetrievalSettings(BaseModel):
    """Search-shape settings (PRD §9). Every count is configurable there."""

    dense_top: int = 60
    sparse_top: int = 60
    rrf_k: int = 60
    # Rerank up to this many fused candidates.
    rerank_max: int = 80
    # Select 8-12 passages under the token budget.
    min_passages: int = 8
    max_passages: int = 12
    passage_token_budget: int = 6_000
    # Diversify across books by default (ignored when a book filter is given).
    max_per_book: int = 4
    # Postvalidation top-up: when validation drops hits, refetch with a larger
    # limit, at most this many rounds and never above the cap (PRD §9).
    topup_max_rounds: int = 3
    topup_limit_cap: int = 240
    bm25: Bm25Settings = Field(default_factory=Bm25Settings)
    # Corpus-epoch policy for BM25 statistics (M6 pilot finding — the epoch
    # includes the document count, so in "live" mode it moves on *every*
    # publish, and every stale-epoch republish paid a full O(corpus)
    # recompute: 12.8k jobs x ~4.2 s ~= the measured 15 h pilot tail).
    #   "live"   — each new revision recomputes; stale republishes converge
    #              through the epoch fan-out. Exact, but O(N²) at library
    #              scale (~700 h for 35k books measured-and-projected).
    #   "frozen" — the epoch is pinned to the last committed record for the
    #              whole campaign: no per-book recompute, and the fan-out
    #              finds no mismatches, so it enqueues nothing. A new book's
    #              novel terms get zero sparse weight until the campaign-end
    #              re-freeze: flip back to "live", delete the
    #              corpus_stats_epoch meta row, and the next publish
    #              recomputes once over the full corpus, converging
    #              everything in one bounded round.
    # Deliberately not part of bm25.settings_sha(): it changes how the epoch
    # is resolved, never the weights for a given epoch.
    stats_epoch: Literal["live", "frozen"] = "live"

    @model_validator(mode="after")
    def _check_counts(self) -> RetrievalSettings:
        for name in ("dense_top", "sparse_top", "rerank_max", "min_passages", "max_passages"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"retrieval.{name} must be positive")
        if self.min_passages > self.max_passages:
            raise ConfigError("retrieval.min_passages must be <= max_passages")
        if self.topup_limit_cap < max(self.dense_top, self.sparse_top):
            raise ConfigError("retrieval.topup_limit_cap must cover the base limits")
        return self


class AnswerEndpoint(BaseModel):
    """An extra answer-model endpoint serving the same model (M9).

    Load-balanced alongside the primary ``services.answer_host:answer_port``;
    the pool fails over per call, so an endpoint that is down costs its
    (fast) connect timeout and the pool degrades to the live ones.
    """

    host: str
    port: int
    # The server-side "model" field for /v1/chat/completions when it differs
    # from the primary's; None inherits ``answer.model_name``.
    model_name: str | None = None
    # Pool weight (default 1): a relative-capacity multiplier on top of the
    # endpoint's observed speed. With every weight at 1 the pool dispatches
    # proportionally to speed; raise it to route this endpoint a share
    # beyond its natural one.
    weight: int = 1

    @model_validator(mode="after")
    def _check_endpoint(self) -> AnswerEndpoint:
        if not self.host:
            raise ConfigError("answer.extra_endpoints[].host must be non-empty")
        if not 1 <= self.port <= 65535:
            raise ConfigError(
                f"answer.extra_endpoints[].port must be 1-65535 (got {self.port})"
            )
        if self.weight < 1:
            raise ConfigError(
                f"answer.extra_endpoints[].weight must be >= 1 (got {self.weight})"
            )
        return self


class AnswerSettings(BaseModel):
    """Answer-model (LLM) settings for cited answering (PRD §12).

    Deliberately separate from :class:`EmbeddingSettings`: the answer model
    never participates in the embedding key, so swapping it must not
    invalidate checkpoints or the index. ``fake`` is for tests only, mirroring
    the embedding convention (default False; a production config must name a
    real model revision).
    """

    fake: bool = False
    # The exact model + revision, e.g. "qwen2.5-7b-instruct@<sha>" — required
    # unless ``fake``.
    model_revision: str | None = None
    # The server-side "model" field for /v1/chat/completions (llama.cpp usually
    # accepts the loaded model's name/alias). Defaults to model_revision.
    model_name: str | None = None
    # Bounded generation: the answer must finish within these limits or the
    # request fails explicitly (PRD §12: bounded timeouts, no open-ended runs).
    max_tokens: int = 1024
    temperature: float = 0.1
    timeout_seconds: float = 60.0
    # The prompt contract version; persisted with every answer so a saved
    # answer can be explained (and re-repaired) under the same rules.
    prompt_version: str = "m5-v1"
    # M9: extra endpoints serving the same model, load-balanced with per-call
    # failover alongside the primary (services.answer_host:answer_port).
    extra_endpoints: list[AnswerEndpoint] = Field(default_factory=list)
    # Pool weight of the primary endpoint (default 1), see
    # ``AnswerEndpoint.weight``. The usual knob is raising the PRIMARY's
    # weight to shed batch load off a shared secondary endpoint.
    weight: int = 1

    @model_validator(mode="after")
    def _check_answer(self) -> AnswerSettings:
        if self.max_tokens <= 0:
            raise ConfigError("answer.max_tokens must be positive")
        if not 0.0 <= self.temperature < 2.0:
            raise ConfigError("answer.temperature must be in [0, 2)")
        if self.timeout_seconds <= 0:
            raise ConfigError("answer.timeout_seconds must be positive")
        if self.weight < 1:
            raise ConfigError(f"answer.weight must be >= 1 (got {self.weight})")
        return self

    @property
    def is_configured(self) -> bool:
        """True when an answer model can be constructed (mirrors embedding)."""
        return self.fake or self.model_revision is not None


class ResumeSettings(BaseModel):
    """Extended per-book summary ("resume") generation settings.

    Resumes are ~700-1000 word English summaries of each published book,
    generated by the *answer* model (``answer`` + ``services.answer_host`` /
    ``answer_port``) but with generation-specific parameters: longer output
    and a higher temperature than cited answering. ``prompt_version`` is
    stamped into the stored resume and into the job idempotency key, so a
    prompt change mints fresh work without touching the old rows. An
    unconfigured answer model is not a config error — resume jobs fail
    per-job with ``answer_model_not_configured`` (mirrors
    ``embedding_not_configured``), so a config without an LLM still serves
    search and the reader.
    """

    enabled: bool = True
    max_tokens: int = 2400
    temperature: float = 0.3
    timeout_seconds: float = 300.0
    prompt_version: str = "resume-v1"
    # Total characters of sampled book text fed to the model as input.
    input_char_budget: int = 16000

    @model_validator(mode="after")
    def _check_resume(self) -> ResumeSettings:
        if self.max_tokens <= 0:
            raise ConfigError("resume.max_tokens must be positive")
        if not 0.0 <= self.temperature < 2.0:
            raise ConfigError("resume.temperature must be in [0, 2)")
        if self.timeout_seconds <= 0:
            raise ConfigError("resume.timeout_seconds must be positive")
        if self.input_char_budget < 2000:
            raise ConfigError("resume.input_char_budget must be >= 2000")
        return self


class PilotSettings(BaseModel):
    """M6 pilot controls (PRD §12).

    ``page_cap`` limits how many pages (PDF) or spine sections (EPUB) of each
    source the extractor processes, so a stratified sample runs with a
    comparable workload. ``None`` (the default) means "no cap" — production
    configs leave it unset and their extraction keys are byte-identical to
    pre-M6 ones. A capped run is a *different* extraction key (the cap is
    hashed into it), so capped and full runs of the same bytes never share
    artifacts or claim each other's state.
    """

    page_cap: int | None = None

    @field_validator("page_cap")
    @classmethod
    def _check_page_cap(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ConfigError(f"pilot.page_cap must be >= 1 (got {v})")
        return v


class WorkerSettings(BaseModel):
    """Ingest-worker execution settings (M9).

    ``max_concurrent_jobs`` is the size of the worker's job thread pool.
    ``1`` (default) is the original strictly-sequential loop, byte-identical
    in behavior; ``> 1`` claims ``max_concurrent_jobs`` jobs and runs their
    handlers concurrently, so multiple LLM endpoints (see
    ``answer.extra_endpoints``) can drain a backfill at once. The state
    database connection is shared-thread-safe and ``jobs.claim`` is
    transactional, so concurrent claims are safe; ``RealQdrantOps`` (the
    embedded/local Qdrant client, which has no internal locking) is guarded
    by its own lock to match.
    """

    max_concurrent_jobs: int = 1

    @model_validator(mode="after")
    def _check_worker(self) -> WorkerSettings:
        if self.max_concurrent_jobs < 1:
            raise ConfigError(
                f"worker.max_concurrent_jobs must be >= 1 (got {self.max_concurrent_jobs})"
            )
        return self


class LoggingSettings(BaseModel):
    """Short-log + per-job verbose-log configuration (PRD §14).

    The worker's short log goes to stderr (a one-line-per-job summary); a
    DEBUG-level per-job capture is written to
    ``<state_root>/job_logs/<job_id>.attempt<N>.log`` and kept only for failed
    jobs. This section is the source of truth for detached/worker runs:
    ``level`` and ``format`` drive the short log, and ``job_log_retention``
    caps how many failure logs accumulate (oldest pruned first). The CLI's
    ``--log-level`` / ``--log-format`` flags override ``level`` / ``format``
    for a single interactive invocation.
    """

    level: str = "INFO"
    format: str = "human"
    job_log_retention: int = 500

    @model_validator(mode="after")
    def _check_logging(self) -> LoggingSettings:
        level = str(self.level).upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ConfigError(
                f"logging.level must be DEBUG/INFO/WARNING/ERROR (got {self.level!r})"
            )
        self.level = level
        if self.format not in ("human", "json"):
            raise ConfigError(f"logging.format must be 'human' or 'json' (got {self.format!r})")
        if self.job_log_retention < 1:
            raise ConfigError(
                f"logging.job_log_retention must be >= 1 (got {self.job_log_retention})"
            )
        return self


class Config(BaseModel):
    """Top-level application configuration."""

    paths: Paths
    services: Services = Field(default_factory=Services)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    scan: ScanSettings = Field(default_factory=ScanSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    resume: ResumeSettings = Field(default_factory=ResumeSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    pilot: PilotSettings = Field(default_factory=PilotSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Optional sentinel files that must exist to prove each mount is present.
    # Mapping of a SOURCE ROOT PATH (the same string used as the key in
    # paths.source_roots) to a file path that must exist. scan_root looks the
    # sentinel up by the root's path string, so key by path, not by a label;
    # a root with no entry is scanned with no sentinel check.
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

    # require_api_token without a value would enforce auth with no usable
    # token; fail at load instead of at the first request.
    if services.get("require_api_token") and not services.get("api_token"):
        raise ConfigError(
            "services.require_api_token is true but no token value is set "
            "(services.api_token or $LIBRARY_RAG_API_TOKEN)"
        )

    # Coerce to Config; pydantic raises ValidationError which we wrap so callers
    # see a single, understandable exception type.
    try:
        return Config.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError, ConfigError, etc.
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"invalid configuration: {exc}") from exc
