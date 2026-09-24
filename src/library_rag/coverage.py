"""Library coverage report (M7 slice 4, PRD §2 / §14).

Answers two questions about the library, read-only and honestly:

* **Pipeline funnel** (PRD §14: "distinguish discovered, archived, extracted,
  OCR-complete, indexed, published, and failed counts") — global counts for
  every stage: documents, revisions (active and total), revisions whose
  archive object is on disk, revisions with a succeeded extraction run, OCR
  unit states, revisions with chunks / embedding batches / ready generations
  / active publications, and the index point counts (catalog expectation
  versus live Qdrant counts).
* **Corpus versus source roots** (PRD §2: "report partial coverage and
  processing failures honestly") — per configured source root: how many
  candidate files were discovered, how many are registered, which valid
  books are *unindexed* (on disk, never scanned), which registered books are
  *orphaned* (alias row whose file is gone), which registered books are
  *stale* (size or mtime differ from the scan record), and which candidates
  are *invalid* (magic bytes match no supported format).

Plus *stalled documents*: documents without an active publication, labeled
with the furthest pipeline stage they reached.

The report never mutates anything: no DB writes, no hashing (staleness is a
stat-only check, like the scan's fast check), no file deletions. Qdrant is
optional — when the storage cannot be opened (a running worker holds the
local lock) the point counts come back as ``None`` and the report still
covers everything else.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .archive import archive_path_for
from .config import Config
from .db import Database
from .identity import normalize_path
from .indexing import FieldCond, IndexFilter, QdrantOps
from .scan import detect_format, iter_candidate_paths

__all__ = [
    "CoverageReport",
    "FailureCounts",
    "RootCoverage",
    "StageCounts",
    "StalledDoc",
    "coverage_report",
]

# Stalled-doc ladder, best first: the stage a document without an active
# publication has reached. "registered" is the floor (a revision row exists,
# nothing else).
_STALEDDOC_STAGES = ("indexed", "embedded", "chunked", "extracted", "registered")


@dataclass
class RootCoverage:
    """Coverage of one source root against the catalog (scan semantics)."""

    root: str
    mount_unavailable: bool = False
    discovered: int = 0
    registered: int = 0
    unindexed: list[str] = field(default_factory=list)
    orphaned: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)


@dataclass
class StageCounts:
    """Pipeline funnel over the whole library (PRD §14 stage vocabulary)."""

    documents: int = 0
    revisions: int = 0
    active_revisions: int = 0
    archived: int = 0
    extracted: int = 0
    ocr_routed: int = 0
    ocr_done: int = 0
    ocr_pending: int = 0
    ocr_failed: int = 0
    chunked: int = 0
    embedded: int = 0
    indexed: int = 0
    indexed_points: int = 0
    published: int = 0
    staged: int = 0
    superseded: int = 0
    published_docs: int = 0
    archive_missing: list[str] = field(default_factory=list)


@dataclass
class FailureCounts:
    """Processing failures, broken out by where they happened (PRD §2)."""

    jobs_by_state: dict[str, int] = field(default_factory=dict)
    failed_jobs_by_stage: dict[str, int] = field(default_factory=dict)
    failed_extractions: int = 0
    ocr_failed_units: int = 0


@dataclass
class StalledDoc:
    """A document without an active publication, and how far it got."""

    doc_id: str
    stage: str


@dataclass
class CoverageReport:
    generated_at: float
    stages: StageCounts = field(default_factory=StageCounts)
    failures: FailureCounts = field(default_factory=FailureCounts)
    roots: list[RootCoverage] = field(default_factory=list)
    stalled: list[StalledDoc] = field(default_factory=list)
    points_expected: int = 0
    points_active: int | None = None
    points_total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count(db: Database, sql: str, params: Sequence[Any] = ()) -> int:
    row = db.query_one(sql, params)
    return int(row["n"]) if row is not None else 0


def _stage(db: Database, sql: str, params: Sequence[Any] = ()) -> set[str]:
    return {str(r["doc_id"]) for r in db.query(sql, params)}


def _stalled_docs(db: Database) -> list[StalledDoc]:
    """Documents without an active publication, labeled by furthest stage."""
    published = _stage(db, "SELECT DISTINCT doc_id FROM publications WHERE state = 'active'")
    # index_generations carries rev_id, not doc_id: the revision holds it.
    indexed = _stage(
        db,
        "SELECT DISTINCT r.doc_id FROM index_generations g"
        " JOIN source_revisions r ON r.rev_id = g.rev_id WHERE g.state = 'ready'",
    )
    embedded = _stage(
        db,
        "SELECT DISTINCT e.doc_id FROM embedding_batches b"
        " JOIN extraction_runs e ON e.run_id = b.run_id",
    )
    chunked = _stage(
        db,
        "SELECT DISTINCT r.doc_id FROM chunks c JOIN source_revisions r ON r.rev_id = c.rev_id",
    )
    extracted = _stage(db, "SELECT DISTINCT doc_id FROM extraction_runs WHERE state = 'succeeded'")
    stalled: list[StalledDoc] = []
    for row in db.query("SELECT doc_id FROM documents"):
        doc_id = str(row["doc_id"])
        if doc_id in published:
            continue
        if doc_id in indexed:
            stage = "indexed"
        elif doc_id in embedded:
            stage = "embedded"
        elif doc_id in chunked:
            stage = "chunked"
        elif doc_id in extracted:
            stage = "extracted"
        else:
            stage = "registered"
        stalled.append(StalledDoc(doc_id=doc_id, stage=stage))
    stalled.sort(key=lambda s: (_STALEDDOC_STAGES.index(s.stage), s.doc_id))
    return stalled


def _root_coverage(
    db: Database, cfg: Config, root: Path, aliases: set[str]
) -> RootCoverage:
    rc = RootCoverage(root=normalize_path(root))
    # Same mount check as scan_root: a missing sentinel (or a missing
    # directory) means the source is not reachable, and nothing below it is
    # meaningful.
    sentinel = cfg.mount_sentinels.get(str(root))
    if sentinel is not None and not Path(sentinel).exists():
        rc.mount_unavailable = True
        return rc
    if not root.is_dir():
        rc.mount_unavailable = True
        return rc

    prefix = normalize_path(root) + "/"
    rc.registered = sum(1 for a in aliases if a.startswith(prefix))

    for p in iter_candidate_paths(
        root, cfg.scan.ignore_dirs, cfg.scan.ignore_files, frozenset(cfg.file_types)
    ):
        norm = normalize_path(p)
        rc.discovered += 1
        if detect_format(p) is None:
            rc.invalid.append(norm)
        elif norm not in aliases:
            rc.unindexed.append(norm)
    rc.invalid.sort()
    rc.unindexed.sort()

    # Orphaned: an alias under this root whose source file no longer exists
    # (the scan's "missing" set; report-only, the row is left in place).
    for a in sorted(a for a in aliases if a.startswith(prefix)):
        if not Path(a).is_file():
            rc.orphaned.append(a)

    # Stale: a scan_state row whose size or mtime no longer matches the file
    # on disk (stat-only — the same cheap check the scan's fast check uses;
    # a gone file is orphaned, not stale).
    for row in db.query("SELECT path, size_bytes, mtime FROM scan_state"):
        path = str(row["path"])
        if not path.startswith(prefix):
            continue
        try:
            st = Path(path).stat()
        except OSError:
            continue
        if st.st_size != int(row["size_bytes"]) or st.st_mtime != float(row["mtime"]):
            rc.stale.append(path)
    rc.stale.sort()
    return rc


def coverage_report(
    db: Database, cfg: Config, *, qdrant: QdrantOps | None = None
) -> CoverageReport:
    """Build a coverage report over the whole library (report-only).

    See the module docstring for the contract. *qdrant* may be ``None`` when
    the index storage cannot be opened; the point counts then stay ``None``.
    """
    report = CoverageReport(generated_at=time.time())
    s = report.stages
    s.documents = _count(db, "SELECT COUNT(*) AS n FROM documents")
    s.revisions = _count(db, "SELECT COUNT(*) AS n FROM source_revisions")
    s.active_revisions = _count(db, "SELECT COUNT(*) AS n FROM source_revisions WHERE is_active = 1")

    for rev in db.query("SELECT sha256, archive_relpath FROM source_revisions"):
        if archive_path_for(cfg.paths.archive_root, str(rev["sha256"])).is_file():
            s.archived += 1
        else:
            s.archive_missing.append(str(rev["archive_relpath"]))
    s.archive_missing.sort()

    s.extracted = _count(
        db, "SELECT COUNT(DISTINCT rev_id) AS n FROM extraction_runs WHERE state = 'succeeded'"
    )
    s.ocr_routed = _count(db, "SELECT COUNT(*) AS n FROM source_units WHERE route = 'ocr'")
    s.ocr_done = _count(db, "SELECT COUNT(*) AS n FROM source_units WHERE ocr_state = 'done'")
    s.ocr_pending = _count(db, "SELECT COUNT(*) AS n FROM source_units WHERE ocr_state = 'pending'")
    s.ocr_failed = _count(db, "SELECT COUNT(*) AS n FROM source_units WHERE ocr_state = 'failed'")
    s.chunked = _count(db, "SELECT COUNT(DISTINCT rev_id) AS n FROM chunks")
    # embedding_batches has no rev_id of its own: the run carries it.
    s.embedded = _count(
        db,
        "SELECT COUNT(DISTINCT e.rev_id) AS n FROM embedding_batches b"
        " JOIN extraction_runs e ON e.run_id = b.run_id",
    )
    s.indexed = _count(
        db, "SELECT COUNT(DISTINCT rev_id) AS n FROM index_generations WHERE state = 'ready'"
    )
    s.indexed_points = _count(
        db, "SELECT COALESCE(SUM(point_count), 0) AS n FROM index_generations WHERE state = 'ready'"
    )
    s.published = _count(db, "SELECT COUNT(*) AS n FROM publications WHERE state = 'active'")
    s.staged = _count(db, "SELECT COUNT(*) AS n FROM publications WHERE state = 'staged'")
    s.superseded = _count(db, "SELECT COUNT(*) AS n FROM publications WHERE state = 'superseded'")
    s.published_docs = _count(
        db, "SELECT COUNT(DISTINCT doc_id) AS n FROM publications WHERE state = 'active'"
    )

    report.points_expected = _count(
        db, "SELECT COALESCE(SUM(expected_points), 0) AS n FROM publications WHERE state = 'active'"
    )
    if qdrant is not None and qdrant.collection_exists():
        report.points_active = qdrant.count(IndexFilter.all(FieldCond("active", "eq", True)))
        report.points_total = qdrant.count(IndexFilter())

    f = report.failures
    for row in db.query("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state"):
        f.jobs_by_state[str(row["state"])] = int(row["n"])
    for row in db.query(
        "SELECT stage, COUNT(*) AS n FROM jobs"
        " WHERE state IN ('retryable_failed','permanent_failed') GROUP BY stage"
    ):
        f.failed_jobs_by_stage[str(row["stage"])] = int(row["n"])
    f.failed_extractions = _count(db, "SELECT COUNT(*) AS n FROM extraction_runs WHERE state = 'failed'")
    f.ocr_failed_units = s.ocr_failed

    aliases = {str(r["path"]) for r in db.query("SELECT path FROM path_aliases")}
    report.roots = [_root_coverage(db, cfg, root, aliases) for root in cfg.paths.source_roots]
    report.stalled = _stalled_docs(db)
    return report
