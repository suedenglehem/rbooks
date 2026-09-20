"""Deterministic identities.

The identity contract (PRD §6) is content-anchored so that the same bytes always
resolve to the same catalog entry and, later, the same Qdrant point ID:

* **Document** — a catalog entry. Its identity is derived from the SHA-256 of the
  *first* content that established it (its "anchor"). Because the anchor is the
  content, a rename (same bytes, new path) never changes the document, and a
  rebuild from a rescan of unchanged content reproduces the same document ID.
* **Source revision** — a specific content of a document. Identified by
  (document, content hash); a rename reuses the existing revision, a content
  change at a known path creates a new one.
* **Task key** — the idempotency key for a durable job, a pure function of the
  stage, input identity, and the versions/configs that define its work.

IDs are UUID-version-5 (name-based, SHA-1) so they are stable across processes
and across a database rebuild, while remaining "UUID-format" for use as Qdrant
point IDs. The full content hashes are stored separately for audit; the UUIDs
are *derived from* them, not substitutes.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import PurePosixPath

__all__ = [
    "NAMESPACE_CHUNK",
    "NAMESPACE_DOC",
    "NAMESPACE_EXTRACT",
    "NAMESPACE_GEN",
    "NAMESPACE_POINT",
    "NAMESPACE_REV",
    "NAMESPACE_TASK",
    "NAMESPACE_UNIT",
    "chunk_key",
    "document_id",
    "embedding_key",
    "extraction_key",
    "generation_id",
    "make_task_key",
    "normalize_path",
    "point_id",
    "publication_id",
    "revision_id",
    "unit_id_for",
    "units_fingerprint",
]

# Fixed name-registry UUIDs. Chosen once and never changed; changing them would
# re-identify the whole library, so they are treated like a schema constant.
NAMESPACE_DOC = uuid.UUID("8f3e2a10-0000-4000-8000-0000000000d1")
NAMESPACE_REV = uuid.UUID("8f3e2a10-0000-4000-8000-0000000000d2")
NAMESPACE_TASK = uuid.UUID("8f3e2a10-0000-4000-8000-0000000000d3")
NAMESPACE_EXTRACT = uuid.UUID("8f3e2a10-0000-4000-8000-0000000000d4")
NAMESPACE_UNIT = uuid.UUID("8f3e2a10-0000-4000-8000-0000000000d5")
NAMESPACE_CHUNK = uuid.UUID("8f3e2a10-0000-4000-8000-0000000000d6")
NAMESPACE_GEN = uuid.UUID("8f3e2a10-0000-4000-8000-0000000000d7")
NAMESPACE_POINT = uuid.UUID("8f3e2a10-0000-4000-8000-0000000000d8")


def normalize_path(path: str | os.PathLike[str]) -> str:
    """Return the canonical path string used as a catalog key.

    Absolute and lexically normalized (``os.path.normpath``), stored in POSIX
    form. We deliberately use ``abspath`` (not ``resolve``) so identity does not
    depend on symlink resolution — consistent with the path-overlap rules in
    :mod:`library_rag.config`.
    """
    ap = os.path.normpath(os.path.abspath(os.fspath(path)))
    return PurePosixPath(ap).as_posix()


def document_id(anchor_sha256: str) -> str:
    """Deterministic document UUID from the first (anchor) content hash."""
    return str(uuid.uuid5(NAMESPACE_DOC, anchor_sha256))


def revision_id(doc_id: str, sha256: str) -> str:
    """Deterministic revision UUID for a given document and content hash."""
    return str(uuid.uuid5(NAMESPACE_REV, f"{doc_id}:{sha256}"))


def extraction_key(
    rev_sha256: str, parser_version: str, settings_sha: str, page_cap: int | None = None
) -> str:
    """Deterministic extraction-run UUID.

    Per PRD §6 the extraction key hashes the source bytes hash, the parser
    version, and the (canonicalized) stage settings. Changing parser or
    normalization settings yields a *different* key, so a re-extract never
    clobbers the outputs of the old settings; changing the answer model or
    embedding model does not appear here, so it never invalidates extraction.

    *page_cap* (M6 pilot, PRD §12) is appended when set, so a capped partial
    run of the same bytes gets its own key and can never share state or
    artifacts with the full run. ``page_cap=None`` leaves the key
    byte-identical to the pre-M6 form — production extraction keys are
    unchanged by the existence of this parameter.
    """
    cap = "" if page_cap is None else f":cap{page_cap}"
    return str(
        uuid.uuid5(
            NAMESPACE_EXTRACT, f"extract:{rev_sha256}:{parser_version}:{settings_sha}{cap}"
        )
    )


def unit_id_for(run_id: str, kind: str, position: int) -> str:
    """Deterministic source-unit UUID within an extraction run.

    *kind* is ``page`` (PDF, zero-based position) or ``section`` (EPUB, zero-based
    spine order position). The same unit rebuilt from the same run always has
    the same ID, which is what makes per-unit re-extraction idempotent.
    """
    return str(uuid.uuid5(NAMESPACE_UNIT, f"unit:{run_id}:{kind}:{position}"))


def units_fingerprint(rows: list[tuple[int, str]]) -> str:
    """SHA-256 over the (position, artifact hash) pairs of a run's units.

    *rows* is a list of ``(position, artifact_sha256)`` tuples, any order. The
    result is the "content of the extraction" as far as downstream stages are
    concerned: any OCR or normalization rewrite that changes a unit artifact
    changes the fingerprint, so chunk jobs keyed on it re-run exactly when the
    inputs to chunking actually changed (PRD §6/§8E).
    """
    canonical = json.dumps(sorted(rows), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def chunk_key(
    run_id: str,
    chunker_sha: str,
    span_signature: str,
    text_sha: str,
) -> str:
    """Deterministic chunk UUID.

    Derived from the run, the chunker settings hash, a hash of the chunk's
    source-span list, and a hash of its text. The same chunk rebuilt from the
    same run and settings always has the same ID, which is what makes
    chunk-job replays idempotent (DELETE-then-INSERT keyed on this identity).
    """
    return str(
        uuid.uuid5(
            NAMESPACE_CHUNK,
            f"chunk:{run_id}:{chunker_sha}:{span_signature}:{text_sha}",
        )
    )


def make_task_key(
    stage: str,
    input_id: str,
    input_version: str,
    range_spec: str = "",
) -> str:
    """Deterministic idempotency key for a durable job.

    Two jobs with the same stage, input identity, and defining versions produce
    the same key, so re-enqueueing after a crash is a no-op (at-least-once with
    idempotent effects, PRD §7).
    """
    return str(uuid.uuid5(NAMESPACE_TASK, f"{stage}:{input_id}:{input_version}:{range_spec}"))


def embedding_key(run_id: str, model_revision: str, settings_sha: str, chunk_id: str) -> str:
    """Deterministic embedding identity for one chunk (PRD §6).

    Hashes the chunk key, the model revision, and the canonical encoding
    configuration. Changing the embedding model never invalidates extraction
    or chunking; changing the chunk (new run/chunker) or the encoding config
    yields new keys, so checkpoints never clobber each other.
    """
    return str(
        uuid.uuid5(NAMESPACE_EXTRACT, f"embed:{run_id}:{model_revision}:{settings_sha}:{chunk_id}")
    )


def generation_id(run_id: str, embedding_sha: str, sparse_stats_sha: str) -> str:
    """Deterministic index-generation UUID.

    A generation is the set of Qdrant points published for one extraction run
    under one encoding and one corpus-wide sparse-statistics epoch. The dense
    side depends only on *embedding_sha* (model revision + encoding config);
    the sparse side on *sparse_stats_sha* (BM25 corpus statistics, PRD §8F).
    Adding or re-chunking a book changes the corpus statistics and therefore
    the generation of *every* run, which is what forces the cheap re-upsert
    (dense vectors are read back from checkpoints, never re-embedded) that
    keeps BM25 IDF correct as the corpus grows.
    """
    return str(uuid.uuid5(NAMESPACE_GEN, f"gen:{run_id}:{embedding_sha}:{sparse_stats_sha}"))


def point_id(chunk_id: str, gen_id: str) -> str:
    """Deterministic Qdrant point ID (UUID format, as the server requires).

    Includes the generation on purpose: staging a new generation mints fresh
    point IDs, so activating replacement points can never overwrite (and thus
    destroy) the still-active points of the generation being replaced.
    Re-upserting the *same* generation hits the same IDs and is a pure
    idempotent overwrite (M4 gate).
    """
    return str(uuid.uuid5(NAMESPACE_POINT, f"point:{chunk_id}:{gen_id}"))


def publication_id(rev_id: str, gen_id: str) -> str:
    """Deterministic publication UUID for a generation of a revision.

    A publication is the act of making one generation's points the visible
    evidence for its revision. Deterministic, so replaying a publish job (or
    reconciling after a crash) resolves to the same row, never a duplicate.
    """
    return str(uuid.uuid5(NAMESPACE_GEN, f"pub:{rev_id}:{gen_id}"))
