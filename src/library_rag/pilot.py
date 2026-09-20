"""M6 pilot tooling: corpus survey, stratified sampling, manifest (PRD §12).

The pilot must answer, from *measured* numbers, what a full ingestion will
cost and whether the hardware can hold it. This module supplies the two
model-free building blocks:

* **survey** — a read-only walk of the configured source roots that profiles
  every candidate (format, page/section count, OCR class, language) without
  hashing, archiving, or writing anything into the catalog. The survey is a
  JSONL file, one record per candidate path, resumable and re-runnable;
* **sample** — a deterministic, seeded stratified selection over the survey
  records (stratum = format|ocr_class|language), with a configurable
  page cap, written as a versioned manifest. Same survey bytes + same
  parameters always yields the same manifest entries;
* **run** — ingests exactly the manifest's books into an isolated sandbox
  (its own archive/state/qdrant/scratch roots, embedded local qdrant, the
  real index never touched), drains the full pipeline, and returns
  :class:`PilotRunMetrics`: measured counts, tokens, stage seconds, peak
  RAM/VRAM, and per-root disk growth.

The survey is deliberately coarse: the language heuristic is a stratification
key (script + a small stopword table), not a claim about any book, and the
OCR class is based on a small *sampled* set of pages, not a full pass.
Anything that needs the true source bytes (hashing, archiving, extraction)
happens only on the sampled books, in the isolated pilot sandbox.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .db import Database, db_path_for
from .embeddings import Embedder, make_embedder
from .identity import normalize_path
from .indexing import QdrantOps, RealQdrantOps
from .jobs import Jobs
from .migrations import migrate
from .scan import detect_format, iter_candidate_paths, process_paths
from .worker import run_worker

__all__ = [
    "MANIFEST_SCHEMA",
    "PILOT_REPORT_SCHEMA",
    "PilotRunMetrics",
    "SurveyRecord",
    "SurveySummary",
    "detect_language",
    "format_manifest_summary",
    "load_survey",
    "profile_candidate",
    "run_pilot",
    "sample_manifest",
    "sandbox_config",
    "stratum_key",
    "survey_sources",
    "write_manifest",
    "write_pilot_report",
    "write_survey",
]

MANIFEST_SCHEMA = 1

# How many evenly-spaced PDF pages the survey inspects per book. 16 spreads
# the risk that a scanned book's text-bearing front matter skews the class,
# at a few milliseconds per page of get_text() cost.
_SURVEY_SAMPLE_PAGES = 16
# A sampled page counts as "has a text layer" at or above this char count;
# below it we assume OCR-grade noise/absent layer.
_TEXT_PAGE_CHARS = 20
# How many EPUB spine sections the survey reads for language detection.
_SURVEY_EPUX_SECTIONS = 3
# Survey text kept per record for provenance (and for the language heuristic).
_SAMPLE_TEXT_CHARS = 200


# --- Language heuristic ------------------------------------------------------

# Small, high-precision stopword sets (top-frequency function words). These
# exist only to break ties between Latin-script languages for the stratum
# key; an unknown Latin language falls through to "other", which is an honest
# stratum of its own.
_LATIN_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        ["the", "of", "and", "to", "in", "a", "that", "is", "was", "for", "on", "with",
         "he", "she", "it", "they", "had", "not", "but", "as", "are", "his", "her"]
    ),
    "fr": frozenset(
        ["le", "la", "les", "de", "des", "et", "à", "a", "un", "une", "est", "que",
         "qui", "dans", "pour", "avec", "pas", "sur", "au", "aux", "par", "ne", "ce"]
    ),
    "de": frozenset(
        ["der", "die", "das", "und", "in", "ein", "eine", "ist", "nicht", "mit", "sie",
         "er", "es", "auf", "an", "von", "zu", "dem", "den", "des", "so"]
    ),
    "es": frozenset(
        ["el", "la", "los", "las", "de", "del", "en", "un", "una", "que", "por", "con",
         "para", "se", "no", "es", "y", "o", "su"]
    ),
    "it": frozenset(
        ["il", "lo", "la", "le", "di", "e", "in", "un", "una", "che", "per", "con",
         "non", "su", "si", "ma", "è", "da", "del"]
    ),
    "pt": frozenset(
        ["o", "a", "os", "as", "de", "da", "e", "em", "um", "uma", "que", "por", "com",
         "para", "não", "são", "ser", "se"]
    ),
    "nl": frozenset(
        ["de", "het", "en", "een", "is", "in", "dat", "te", "voor", "met", "op", "aan",
         "der", "van"]
    ),
    "pl": frozenset(
        ["i", "w", "nie", "jest", "to", "co", "jak", "się", "po", "na", "ze", "od",
         "dla", "tym"]
    ),
}

_SCRIPT_RANGES = (
    ("lat", (0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)),
    ("cyr", (0x0400, 0x04FF)),
    ("cjk", (0x4E00, 0x9FFF), (0x3040, 0x30FF)),
    ("ar", (0x0600, 0x06FF)),
    ("gr", (0x0370, 0x03FF)),
    ("he", (0x0590, 0x05FF)),
)


def _dominant_script(text: str) -> str | None:
    counts: dict[str, int] = {}
    for ch in text[:4000]:
        o = ord(ch)
        for name, *ranges in _SCRIPT_RANGES:
            if any(lo <= o <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return None
    return max(sorted(counts), key=lambda k: counts[k])


def detect_language(text: str) -> str:
    """Coarse language label for strata: a script tag (``cyr``/``cjk``/...) or
    a Latin language code (``en``/``fr``/...). ``other`` = Latin but no
    stopword match; ``unknown`` = no letters at all. Deterministic."""
    if not text:
        return "unknown"
    script = _dominant_script(text)
    if script is None:
        return "unknown"
    if script != "lat":
        return script
    words = set(re.findall(r"[a-zà-ÿä-öø-ÿ]+", text[:4000].lower()))
    best: str | None = None
    best_score = 0
    for lang in sorted(_LATIN_STOPWORDS):
        score = len(words & _LATIN_STOPWORDS[lang])
        if score > best_score:
            best, best_score = lang, score
    return best if best is not None else "other"


# --- Per-candidate profiling (read-only) -------------------------------------


@dataclass
class SurveyRecord:
    """One survey row. ``path`` is normalized (see identity.normalize_path)."""

    path: str
    size: int
    mtime: float
    format: str  # "pdf" | "epub" | "invalid" | "unreadable"
    pages: int | None
    ocr_class: str  # "native" | "scanned" | "mixed" | "unknown" (epub: "native")
    language: str
    sampled: int
    text_chars: int
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "path": self.path,
                "size": self.size,
                "mtime": self.mtime,
                "format": self.format,
                "pages": self.pages,
                "ocr_class": self.ocr_class,
                "language": self.language,
                "sampled": self.sampled,
                "text_chars": self.text_chars,
                "error": self.error,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> SurveyRecord:
        d = json.loads(line)
        return cls(
            path=d["path"],
            size=d["size"],
            mtime=d["mtime"],
            format=d["format"],
            pages=d.get("pages"),
            ocr_class=d.get("ocr_class", "unknown"),
            language=d.get("language", "unknown"),
            sampled=d.get("sampled", 0),
            text_chars=d.get("text_chars", 0),
            error=d.get("error"),
        )


def _classify(text_fractions: list[float]) -> str:
    """Map the fraction of sampled pages with a text layer to an OCR class."""
    if not text_fractions:
        return "unknown"
    frac = sum(text_fractions) / len(text_fractions)
    if frac >= 0.75:
        return "native"
    if frac <= 0.25:
        return "scanned"
    return "mixed"


def _profile_pdf(path: Path) -> dict[str, object]:
    """Open the PDF read-only; sample pages; return profile fields."""
    import pymupdf  # local import: keeps module import cheap for non-survey paths

    with pymupdf.open(str(path)) as doc:  # type: ignore[no-untyped-call]
        n = doc.page_count
        if n == 0:
            return {"pages": 0, "ocr_class": "unknown", "sampled": 0, "text_chars": 0,
                    "text_sample": "", "error": "empty_pdf"}
        k = min(n, _SURVEY_SAMPLE_PAGES)
        indices = [0] if k == 1 else [round(i * (n - 1) / (k - 1)) for i in range(k)]
        fractions: list[float] = []
        chars = 0
        text_sample = ""
        for i in indices:
            try:
                text = doc[i].get_text()
            except Exception:
                text = ""
            stripped = text.strip()
            fractions.append(1.0 if len(stripped) >= _TEXT_PAGE_CHARS else 0.0)
            chars += len(stripped)
            if not text_sample and stripped:
                text_sample = stripped[:_SAMPLE_TEXT_CHARS]
        return {
            "pages": n,
            "ocr_class": _classify(fractions),
            "sampled": len(fractions),
            "text_chars": chars,
            "text_sample": text_sample,
            "error": None,
        }


def _opf_language(opf_xml: bytes) -> str | None:
    """Declared ``dc:language`` in an OPF document, or None."""
    try:
        root = ET.fromstring(opf_xml)
    except ET.ParseError:
        return None
    for el in root.iter():
        if el.tag.split("}")[-1] == "language":
            code = (el.text or "").strip().split("-")[0].lower()
            return code or None
    return None


def _profile_epub(path: Path) -> dict[str, object]:
    """Read the OPF spine (no ebooklib, no full parse); sample section text."""
    pages: int | None = None
    error: str | None = None
    declared: str | None = None
    texts: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            try:
                container = zf.read("META-INF/container.xml")
            except KeyError:
                raise ValueError("no META-INF/container.xml") from None
            croot = ET.fromstring(container)
            full_path = None
            for el in croot.iter():
                if el.tag.split("}")[-1] == "rootfile":
                    full_path = el.get("full-path")
                    break
            if not full_path:
                raise ValueError("no rootfile in container.xml")
            opf_dir = os.path.dirname(full_path)
            opf_xml = zf.read(full_path)
            declared = _opf_language(opf_xml)
            oroot = ET.fromstring(opf_xml)
            manifest: dict[str, str] = {}
            for el in oroot.iter():
                if el.tag.split("}")[-1] == "item":
                    iid, href = el.get("id"), el.get("href")
                    if iid and href:
                        manifest[iid] = href
            spine_ids: list[str] = []
            for el in oroot.iter():
                if el.tag.split("}")[-1] == "itemref":
                    idref = el.get("idref")
                    if idref:
                        spine_ids.append(idref)
            pages = len(spine_ids)
            # Sample the first few XHTML spine items for language detection.
            for idref in spine_ids[:_SURVEY_EPUX_SECTIONS]:
                href = manifest.get(idref)
                if not href:
                    continue
                entry = (opf_dir + "/" + href) if opf_dir else href
                try:
                    data = zf.read(entry)
                except (KeyError, OSError):
                    continue
                text = re.sub(r"<[^>]+>", " ", data.decode("utf-8", "replace"))
                texts.append(" ".join(text.split())[:1500])
    except (OSError, ET.ParseError, ValueError, zipfile.BadZipFile) as exc:
        error = f"epub: {exc}"

    sample_text = " ".join(texts)
    if declared:
        # The OPF-declared BCP-47 code is authoritative for strata — normalize
        # its primary subtag ("en-US" -> "en", "eng" -> "en"). Re-running it
        # through detect_language() would stopword-match the code itself.
        language = declared.strip().lower().split("-")[0][:2] or detect_language(sample_text)
    else:
        language = detect_language(sample_text)
    return {
        "pages": pages,
        "ocr_class": "native" if pages else "unknown",  # EPUBs are digital text
        "sampled": len(texts),
        "text_chars": sum(len(t) for t in texts),
        "text_sample": sample_text[:_SAMPLE_TEXT_CHARS],
        "language": language,
        "error": error,
    }


def profile_candidate(path: Path) -> SurveyRecord:
    """Profile one candidate file. Read-only; never raises on bad content —
    unprofileable files are records with an ``error``, never exceptions."""
    norm = normalize_path(path)
    try:
        st = path.stat()
    except OSError as exc:
        return SurveyRecord(norm, 0, 0.0, "unreadable", None, "unknown", "unknown", 0, 0,
                            error=f"stat: {exc}")
    fmt = detect_format(path)
    if fmt is None:
        return SurveyRecord(norm, st.st_size, st.st_mtime, "invalid", None, "unknown", "unknown",
                            0, 0, error="magic bytes do not match extension")
    try:
        prof = _profile_pdf(path) if fmt.value == "pdf" else _profile_epub(path)
    except Exception as exc:  # unreadable structure of any kind
        return SurveyRecord(norm, st.st_size, st.st_mtime, fmt.value, None, "unknown", "unknown",
                            0, 0, error=f"open: {exc}")
    text_sample = str(prof.get("text_sample", ""))
    language = str(prof["language"]) if fmt.value == "epub" else detect_language(text_sample)
    pages = prof.get("pages")
    sampled = prof.get("sampled", 0)
    text_chars = prof.get("text_chars", 0)
    err = prof.get("error")
    return SurveyRecord(
        norm,
        st.st_size,
        st.st_mtime,
        fmt.value,
        pages if isinstance(pages, int) else None,
        str(prof["ocr_class"]),
        language,
        sampled if isinstance(sampled, int) else 0,
        text_chars if isinstance(text_chars, int) else 0,
        None if err is None else str(err),
    )


# --- Survey driver -------------------------------------------------------------


@dataclass
class SurveySummary:
    roots: list[str] = field(default_factory=list)
    files_scanned: int = 0
    by_format: dict[str, int] = field(default_factory=dict)
    by_stratum: dict[str, int] = field(default_factory=dict)
    errors: int = 0
    seconds: float = 0.0


def _load_existing_survey(out_path: Path) -> list[SurveyRecord]:
    if not out_path.is_file():
        return []
    records: list[SurveyRecord] = []
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(SurveyRecord.from_json(line))
    return records


def write_survey(out_path: Path, records: Iterable[SurveyRecord]) -> None:
    """Write the survey JSONL atomically, sorted by normalized path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in sorted(records, key=lambda r: r.path):
            fh.write(rec.to_json() + "\n")
    os.replace(tmp, out_path)


def load_survey(out_path: Path) -> list[SurveyRecord]:
    records: list[SurveyRecord] = []
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(SurveyRecord.from_json(line))
    return records


def survey_sources(
    cfg: Config,
    out_path: Path,
    *,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> SurveySummary:
    """Profile every candidate under ``cfg.paths.source_roots`` into *out_path*.

    Read-only with respect to the sources (stat + open-for-read). Resumable:
    records already in *out_path* are kept and only the missing paths are
    profiled; the file is rewritten atomically with the merged, sorted set.

    Profiling is deliberately *serial*: MuPDF's C layer is not thread-safe for
    concurrent ``get_textpage`` (observed segfault at 16 threads, clean at 1),
    and on a CPU-quota-capped host threads only add crash risk — 8 workers
    profiled slower per file than 1. A process pool would be safe but buys
    nothing here, so we profile one file at a time.
    """
    started = time.monotonic()
    existing = _load_existing_survey(out_path)
    have = {r.path for r in existing}

    candidates: list[Path] = []
    for root in cfg.paths.source_roots:
        candidates.extend(iter_candidate_paths(root, cfg.scan.ignore_dirs, cfg.scan.ignore_files))
    candidates = [p for p in candidates if normalize_path(p) not in have]
    if limit is not None:
        candidates = candidates[:limit]

    fresh: list[SurveyRecord] = []
    for done, path in enumerate(candidates, start=1):
        fresh.append(profile_candidate(path))
        if on_progress is not None and done % 500 == 0:
            on_progress(done, len(candidates))

    write_survey(out_path, existing + fresh)

    all_records = existing + fresh
    summary = SurveySummary(
        roots=[normalize_path(r) for r in cfg.paths.source_roots],
        files_scanned=len(all_records),
        seconds=round(time.monotonic() - started, 1),
    )
    for rec in all_records:
        summary.by_format[rec.format] = summary.by_format.get(rec.format, 0) + 1
        if rec.format in ("pdf", "epub") and rec.error is None:
            summary.by_stratum[stratum_key(rec)] = summary.by_stratum.get(stratum_key(rec), 0) + 1
        if rec.error is not None:
            summary.errors += 1
    return summary


# --- Stratified sampling -------------------------------------------------------


def stratum_key(rec: SurveyRecord) -> str:
    """The sampling stratum: ``format|ocr_class|language``."""
    return f"{rec.format}|{rec.ocr_class}|{rec.language}"


def _rank(seed: int, key: str) -> int:
    """Stable per-seed pseudo-rank in [0, 4096) for a path or stratum key."""
    return int(hashlib.sha256(f"m6pilot:{seed}:{key}".encode()).hexdigest()[:4], 16)


def _select_within_stratum(seed: int, records: list[SurveyRecord], k: int) -> list[SurveyRecord]:
    ordered = sorted(records, key=lambda r: (_rank(seed, r.path), r.path))
    return ordered[: min(k, len(ordered))]


def sample_manifest(
    records: list[SurveyRecord],
    *,
    seed: int,
    target: int,
    page_cap: int | None = None,
    survey_sha: str | None = None,
    created_at: float | None = None,
) -> dict[str, object]:
    """Deterministic stratified manifest over survey records (PRD §12).

    Candidates are survey-successful PDF/EPUB records (format known, page
    count known, no error). Every non-empty stratum gets at least one book
    (minimum coverage); the remainder is allocated proportionally to stratum
    size by largest remainder. Within a stratum, selection is a seeded
    pseudo-shuffle (ranked by a SHA-256 of ``seed:path``), so the same survey
    bytes + parameters always give the same manifest.
    """
    if target < 1:
        raise ValueError("target must be >= 1")
    if page_cap is not None and page_cap < 1:
        raise ValueError("page_cap must be >= 1")

    candidates = [
        r
        for r in records
        if r.format in ("pdf", "epub") and r.pages is not None and r.error is None
    ]
    strata: dict[str, list[SurveyRecord]] = {}
    for rec in candidates:
        strata.setdefault(stratum_key(rec), []).append(rec)

    order = sorted(strata, key=lambda k: (_rank(seed, k), k))
    chosen: dict[str, list[SurveyRecord]] = {}
    if target < len(order):
        # Fewer books than strata: one per stratum, in rank order, up to target.
        for k in order[:target]:
            chosen[k] = _select_within_stratum(seed, strata[k], 1)
    else:
        base = 1
        remainder = target - len(order)
        n_total = len(candidates)
        ideal = {k: remainder * len(strata[k]) / n_total for k in order}
        floors = {k: int(ideal[k]) for k in order}
        leftover = remainder - sum(floors.values())
        # Largest-remainder: distribute by fractional part, ties by stratum rank.
        for k in sorted(order, key=lambda k: (-(ideal[k] - floors[k]), _rank(seed, k), k)):
            if leftover <= 0:
                break
            floors[k] += 1
            leftover -= 1
        for k in order:
            chosen[k] = _select_within_stratum(seed, strata[k], base + floors[k])

    entries = sorted(
        (
            {
                "path": r.path,
                "format": r.format,
                "pages": r.pages,
                "ocr_class": r.ocr_class,
                "language": r.language,
                "size": r.size,
            }
            for k in order
            for r in chosen.get(k, ())
        ),
        key=lambda e: str(e["path"]),
    )
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "created_at": created_at if created_at is not None else time.time(),
        "seed": seed,
        "target": target,
        "page_cap": page_cap,
        "survey_sha256": survey_sha,
        "totals": {
            "candidates": len(candidates),
            "strata": len(order),
            "sampled": len(entries),
        },
        "strata": {
            k: {"population": len(strata[k]), "sampled": len(chosen.get(k, ()))} for k in order
        },
        "entries": entries,
    }
    return manifest


def write_manifest(manifest: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)


def format_manifest_summary(manifest: dict[str, object]) -> list[str]:
    """Human-readable strata table for a manifest (CLI output)."""
    totals: dict[str, int] = manifest["totals"]  # type: ignore[assignment]
    strata: dict[str, dict[str, int]] = manifest["strata"]  # type: ignore[assignment]
    lines = [
        f"  candidates: {totals['candidates']}  strata: {totals['strata']}  "
        f"sampled: {totals['sampled']}  (seed {manifest['seed']}, page cap {manifest['page_cap']})",
        "  strata (population/sampled):",
    ]
    for key in sorted(strata, key=lambda k: (-strata[k]["sampled"], k)):
        s = strata[key]
        lines.append(f"    {key}: {s['population']}/{s['sampled']}")
    return lines


# --- Pilot run (phase 2) ------------------------------------------------------

PILOT_REPORT_SCHEMA = "pilot-run/1"

# Sandbox sub-roots whose byte growth the run report attributes to the run.
_DISK_ROOT_ATTRS: tuple[tuple[str, str], ...] = (
    ("archive", "archive_root"),
    ("artifacts", "artifact_root"),
    ("state", "state_root"),
    ("qdrant", "qdrant_root"),
)


def _dir_bytes(root: Path) -> int:
    """Total file bytes under *root* (0 when the directory is absent)."""
    total = 0
    if not root.is_dir():
        return 0
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            with contextlib.suppress(OSError):
                total += p.stat().st_size
    return total


def _vram_snapshot() -> dict[str, int] | None:
    """Per-GPU ``memory.used`` (MiB) from nvidia-smi, or None if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    snap: dict[str, int] = {}
    for line in out.stdout.splitlines():
        index, _, used = line.partition(",")
        if index.strip().isdigit() and used.strip().isdigit():
            snap[f"gpu{index.strip()}"] = int(used.strip())
    return snap or None


class _VramSampler:
    """Background nvidia-smi sampler: baseline before, peaks during, per GPU.

    The operator's services (bge-m3 on GPU 2, vLLM on GPU 0) are already
    resident, so the *delta* (peak minus baseline) is the marginal GPU cost
    of the pilot's embedding load — that is what the report quotes.
    """

    def __init__(self, interval: float = 5.0) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline: dict[str, int] = {}
        self.peaks: dict[str, int] = {}

    def start(self) -> None:
        self.baseline = _vram_snapshot() or {}
        self.peaks = dict(self.baseline)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            snap = _vram_snapshot()
            if snap is None:
                continue
            for key, used in snap.items():
                if used > self.peaks.get(key, 0):
                    self.peaks[key] = used

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def report(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for key in sorted(set(self.baseline) | set(self.peaks)):
            out[key] = {"baseline_mb": self.baseline.get(key, 0), "peak_mb": self.peaks.get(key, 0)}
        return out


def sandbox_config(cfg: Config, sandbox_root: Path, *, page_cap: int | None) -> Config:
    """Derive the isolated sandbox config for a pilot run (PRD §12).

    Every writable root (archive, artifacts, state, qdrant, scratch) moves
    under *sandbox_root*; source roots, mount sentinels, and all
    measurement-relevant settings (extraction, OCR, chunking, embedding,
    BM25, retrieval) are inherited unchanged, so the run measures exactly
    the operator's intended frozen config. Qdrant switches to embedded local
    mode under the sandbox: no server, and the real index is never touched.
    """
    root = Path(sandbox_root).expanduser().resolve()
    out = cfg.model_copy(deep=True)
    out.paths.archive_root = root / "archive"
    out.paths.artifact_root = root / "artifacts"
    out.paths.state_root = root / "state"
    out.paths.qdrant_root = root / "qdrant"
    out.paths.scratch_root = root / "scratch"
    out.services.qdrant_path = str(root / "qdrant")
    out.pilot.page_cap = page_cap
    return out


@dataclass
class PilotRunMetrics:
    """Measured outcome of one pilot sandbox run (PRD §12).

    Every number here is a measurement, not an estimate: counts and tokens
    come from the sandbox state database, stage seconds from the jobs table
    (job lifecycles, so the earliest jobs include their short queue wait),
    memory from ``getrusage`` plus periodic nvidia-smi sampling, and disk
    from byte walks of the sandbox roots taken before and after the run.
    """

    manifest_sha256: str
    manifest_target: int
    page_cap: int | None
    sandbox_root: str
    started_at: float
    finished_at: float
    registration_seconds: float
    worker_seconds: float
    seconds: float
    completed_jobs: int
    documents: int
    revisions: int
    registered_new: int
    registered_aliases: int
    unchanged: int
    invalid: int
    missing: int
    units: int
    unit_chars: int
    chunks: int
    chunk_tokens: int
    chunk_chars: int
    publications_active: int
    extraction_failures: int
    stages: dict[str, dict[str, float]]
    failed_jobs: list[dict[str, str]]
    pending_jobs: int
    per_book: list[dict[str, object]]
    peak_rss_mb: float
    peak_children_rss_mb: float
    vram: dict[str, dict[str, int]]
    disk: dict[str, dict[str, int]]
    embed_model_revision: str
    embed_dimensions: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_pilot(
    cfg: Config,
    manifest_path: Path,
    sandbox_root: Path,
    *,
    qdrant: QdrantOps | None = None,
    embedder: Embedder | None = None,
    vram_interval: float = 5.0,
) -> PilotRunMetrics:
    """Ingest exactly the manifest's books into an isolated sandbox, and measure.

    Two phases, both measured: (1) *registration* — hash, archive, and
    register each manifest entry and enqueue its extract job (the same
    per-file rules as a full scan); (2) *worker* — drain the whole pipeline
    (extract → ocr → chunk → embed → publish) with ``once=True``.

    Pass explicit *qdrant* and *embedder* in tests (``FakeQdrant`` plus
    ``embedding.fake = True`` keeps the run Docker/GPU-free).
    """
    manifest_path = Path(manifest_path)
    manifest_raw: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_raw.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema {manifest_raw.get('schema')!r} != {MANIFEST_SCHEMA}")
    entries: list[dict[str, Any]] = manifest_raw.get("entries", [])
    page_cap_raw = manifest_raw.get("page_cap")
    page_cap: int | None = int(page_cap_raw) if page_cap_raw is not None else None

    sandbox_cfg = sandbox_config(cfg, Path(sandbox_root), page_cap=page_cap)
    for root in (
        sandbox_cfg.paths.archive_root,
        sandbox_cfg.paths.artifact_root,
        sandbox_cfg.paths.state_root,
        sandbox_cfg.paths.qdrant_root,
        sandbox_cfg.paths.scratch_root,
    ):
        root.mkdir(parents=True, exist_ok=True)

    disk_roots = [(name, getattr(sandbox_cfg.paths, attr)) for name, attr in _DISK_ROOT_ATTRS]
    disk_baseline = {name: _dir_bytes(root) for name, root in disk_roots}
    vram_sampler = _VramSampler(vram_interval)
    vram_sampler.start()
    started_at = time.time()
    started = time.monotonic()
    db = Database.connect(db_path_for(sandbox_cfg.paths.state_root))
    try:
        migrate(db)
        report = process_paths(
            db, sandbox_cfg, Jobs(db), (Path(str(e["path"])) for e in entries)
        )
        registration_seconds = time.monotonic() - started
        q = qdrant if qdrant is not None else RealQdrantOps(sandbox_cfg)
        emb = embedder if embedder is not None else make_embedder(sandbox_cfg)
        worker_started = time.monotonic()
        completed_jobs = run_worker(db, sandbox_cfg, once=True, qdrant=q, embedder=emb)
        worker_seconds = time.monotonic() - worker_started

        def _count(sql: str) -> int:
            row = db.query_one(sql)
            return int(row["n"]) if row is not None else 0

        stages: dict[str, dict[str, float]] = {}
        for row in db.query(
            """
            SELECT stage,
                   COUNT(*) AS n,
                   COALESCE(SUM(state = 'succeeded'), 0) AS ok,
                   COALESCE(SUM(state IN ('retryable_failed', 'permanent_failed')), 0) AS failed,
                   COALESCE(SUM(updated_at - created_at), 0.0) AS wall,
                   COALESCE(MAX(updated_at - created_at), 0.0) AS wall_max
            FROM jobs
            GROUP BY stage
            """
        ):
            stages[str(row["stage"])] = {
                "jobs": float(row["n"]),
                "succeeded": float(row["ok"]),
                "failed": float(row["failed"]),
                "seconds": round(float(row["wall"]), 1),
                "seconds_max": round(float(row["wall_max"]), 1),
            }
        failed_jobs = [
            {
                "stage": str(row["stage"]),
                "error_category": str(row["error_category"] or ""),
                "error_detail": str(row["error_detail"] or "")[:300],
            }
            for row in db.query(
                "SELECT stage, error_category, error_detail FROM jobs "
                "WHERE state = 'permanent_failed' ORDER BY job_id"
            )
        ]
        per_book: list[dict[str, object]] = []
        for row in db.query(
            """
            SELECT r.rev_id, r.format,
                   (SELECT COUNT(*) FROM source_units u WHERE u.rev_id = r.rev_id) AS units,
                   (SELECT COALESCE(SUM(u.char_count), 0)
                      FROM source_units u WHERE u.rev_id = r.rev_id) AS chars,
                   (SELECT COUNT(*) FROM chunks c WHERE c.rev_id = r.rev_id) AS chunks,
                   (SELECT COALESCE(SUM(c.token_count), 0)
                      FROM chunks c WHERE c.rev_id = r.rev_id) AS tokens
            FROM source_revisions r
            ORDER BY r.rev_id
            """
        ):
            per_book.append(
                {
                    "rev_id": str(row["rev_id"]),
                    "format": str(row["format"]),
                    "units": int(row["units"]),
                    "chars": int(row["chars"]),
                    "chunks": int(row["chunks"]),
                    "tokens": int(row["tokens"]),
                }
            )
        units_row = db.query_one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(char_count), 0) AS chars FROM source_units"
        )
        chunks_row = db.query_one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(token_count), 0) AS tokens, "
            "COALESCE(SUM(LENGTH(text)), 0) AS chars FROM chunks"
        )
        disk: dict[str, dict[str, int]] = {}
        for name, root in disk_roots:
            after = _dir_bytes(root)
            disk[name] = {"bytes": after, "delta_bytes": after - disk_baseline[name]}
        metrics = PilotRunMetrics(
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            manifest_target=int(manifest_raw.get("target", 0)),
            page_cap=page_cap,
            sandbox_root=sandbox_cfg.paths.state_root.parent.as_posix(),
            started_at=started_at,
            finished_at=time.time(),
            registration_seconds=round(registration_seconds, 1),
            worker_seconds=round(worker_seconds, 1),
            seconds=round(time.monotonic() - started, 1),
            completed_jobs=completed_jobs,
            documents=_count("SELECT COUNT(*) AS n FROM documents"),
            revisions=_count("SELECT COUNT(*) AS n FROM source_revisions"),
            registered_new=int(report.new_documents) + int(report.new_revisions),
            registered_aliases=int(report.aliases),
            unchanged=int(report.unchanged),
            invalid=len(report.invalid),
            missing=len(report.missing),
            units=int(units_row["n"]) if units_row is not None else 0,
            unit_chars=int(units_row["chars"]) if units_row is not None else 0,
            chunks=int(chunks_row["n"]) if chunks_row is not None else 0,
            chunk_tokens=int(chunks_row["tokens"]) if chunks_row is not None else 0,
            chunk_chars=int(chunks_row["chars"]) if chunks_row is not None else 0,
            publications_active=_count(
                "SELECT COUNT(*) AS n FROM publications WHERE state = 'active'"
            ),
            extraction_failures=_count(
                "SELECT COUNT(*) AS n FROM extraction_runs WHERE state = 'failed'"
            ),
            stages=stages,
            failed_jobs=failed_jobs,
            pending_jobs=_count(
                "SELECT COUNT(*) AS n FROM jobs WHERE state IN ('pending', 'running', 'retryable_failed')"
            ),
            per_book=per_book,
            peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
            peak_children_rss_mb=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0, 1),
            vram=vram_sampler.report(),
            disk=disk,
            embed_model_revision=emb.model_revision,
            embed_dimensions=emb.dimensions,
        )
    finally:
        vram_sampler.stop()
        db.close()
    return metrics


def write_pilot_report(metrics: PilotRunMetrics, out_path: Path) -> None:
    """Write the run report atomically as schema-tagged JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"schema": PILOT_REPORT_SCHEMA, **metrics.to_dict()}
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
