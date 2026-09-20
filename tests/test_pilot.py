"""M6 pilot survey + stratified sampling (PRD §12).

Covers: the coarse language heuristic, per-candidate profiling (never
raises on bad content), survey resumability, and the sampler's
determinism/coverage guarantees — min-1-per-stratum, largest-remainder
totals, path-sorted entries, page-cap propagation, and exclusion of
unprofileable records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from fixtures import make_epub, make_mixed_pdf, make_pdf
from library_rag.config import Config
from library_rag.indexing import FakeQdrant
from library_rag.pilot import (
    PILOT_REPORT_SCHEMA,
    SurveyRecord,
    detect_language,
    format_manifest_summary,
    load_survey,
    profile_candidate,
    run_pilot,
    sample_manifest,
    sandbox_config,
    stratum_key,
    survey_sources,
    write_manifest,
    write_pilot_report,
    write_survey,
)

_EN_PAGE = "The cat sat on the mat and the dog was there."
_EN_PAGE_2 = "More words on this second page of the book."


def _rec(
    path: str,
    fmt: str = "pdf",
    ocr: str = "native",
    lang: str = "en",
    pages: int | None = 10,
    error: str | None = None,
) -> SurveyRecord:
    return SurveyRecord(path, 1000, 0.0, fmt, pages, ocr, lang, 16, 500, error)


# --- language heuristic ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The cat sat on the mat and the dog was not there.", "en"),
        ("Le chat est dans la maison et la femme est sur le balcon.", "fr"),
        ("Der Hund ist auf dem Balkon und die Katze ist nicht im Haus.", "de"),
        ("Привет мир это простой тестовый текст.", "cyr"),
        ("这是一个中文的测试文本样本。", "cjk"),
        ("مرحبا بالعالم هذا نص تجريبي.", "ar"),
        ("Γειά σου Κόσμε αυτό είναι δοκιμή.", "gr"),  # noqa: RUF001
        ("שלום עולם זהו נסיון.", "he"),
        ("", "unknown"),
        ("12345 67890 !!", "unknown"),
        ("qwzx bcvv plmn rtuy", "other"),
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


def test_detect_language_is_deterministic() -> None:
    text = "The cat sat on the mat and the dog was not there."
    assert detect_language(text) == detect_language(text)


def test_stratum_key_shape() -> None:
    rec = _rec("/b/x.pdf", fmt="epub", ocr="native", lang="de")
    assert stratum_key(rec) == "epub|native|de"


# --- per-candidate profiling -------------------------------------------------


def test_profile_candidate_never_raises(base_config: Config) -> None:
    src_root = base_config.paths.source_roots[0]
    src_root.mkdir(parents=True)

    missing = profile_candidate(src_root / "nope.pdf")
    assert missing.format == "unreadable"
    assert missing.error is not None

    bad = src_root / "bad.pdf"
    bad.write_text("not a pdf", encoding="utf-8")
    rec = profile_candidate(bad)
    assert rec.format == "invalid"
    assert rec.error is not None

    good = src_root / "good.pdf"
    make_pdf(good, [_EN_PAGE])
    ok = profile_candidate(good)
    assert ok.format == "pdf"
    assert ok.pages == 1
    assert ok.error is None
    assert ok.ocr_class == "native"
    assert ok.language == "en"


def test_survey_write_load_roundtrip_is_sorted(tmp_path: Path) -> None:
    recs = [_rec(f"/b/z{i}.pdf") for i in range(3)] + [_rec("/a/first.epub", fmt="epub")]
    out = tmp_path / "survey.jsonl"
    write_survey(out, recs)
    loaded = load_survey(out)
    assert [r.path for r in loaded] == ["/a/first.epub", "/b/z0.pdf", "/b/z1.pdf", "/b/z2.pdf"]
    assert loaded[0] == recs[3]  # dataclass round-trips field-for-field


# --- survey driver -----------------------------------------------------------


def test_survey_sources_profiles_and_resumes(base_config: Config) -> None:
    src_root = base_config.paths.source_roots[0]
    (src_root / "a").mkdir(parents=True)
    (src_root / "b").mkdir()
    (src_root / "c").mkdir()
    make_pdf(src_root / "a" / "native.pdf", [_EN_PAGE, _EN_PAGE_2])
    make_mixed_pdf(src_root / "a" / "scanned.pdf", ["scanned", "scanned"])
    make_epub(
        src_root / "b" / "book.epub",
        [("Ch1", ["The cat sat on the mat."]), ("Ch2", ["The dog was in the yard."])],
    )
    (src_root / "c" / "bad.pdf").write_text("not a pdf", encoding="utf-8")
    (src_root / "c" / "notes.txt").write_text("ignored", encoding="utf-8")
    # A symlinked directory pointing outside the source tree is never followed.
    outside = src_root.parent / "outside"
    outside.mkdir(exist_ok=True)
    make_pdf(outside / "hidden.pdf", [_EN_PAGE])
    (src_root / "link").symlink_to(outside)

    out = base_config.paths.scratch_root / "survey.jsonl"
    summary = survey_sources(base_config, out)

    assert summary.files_scanned == 4  # hidden.pdf and notes.txt never seen
    assert summary.by_format == {"pdf": 2, "epub": 1, "invalid": 1}
    assert summary.errors == 1
    assert summary.by_stratum == {
        "pdf|native|en": 1,
        "pdf|scanned|unknown": 1,
        "epub|native|en": 1,
    }

    records = load_survey(out)
    assert [r.path for r in records] == sorted(r.path for r in records)
    by_path = {r.path: r for r in records}
    assert len(by_path) == 4

    # Resume: a new file appears; existing records are kept, not re-profiled.
    make_pdf(src_root / "a" / "later.pdf", [_EN_PAGE])
    summary2 = survey_sources(base_config, out)
    assert summary2.files_scanned == 5
    records2 = load_survey(out)
    assert len(records2) == 5
    assert [r.path for r in records2] == sorted(r.path for r in records2)
    native = next(r for r in records2 if r.path.endswith("native.pdf"))
    assert (native.pages, native.ocr_class, native.language) == (2, "native", "en")


def test_survey_sources_limit(base_config: Config) -> None:
    src_root = base_config.paths.source_roots[0]
    src_root.mkdir(parents=True)
    for i in range(4):
        make_pdf(src_root / f"b{i}.pdf", [_EN_PAGE])
    out = base_config.paths.scratch_root / "survey.jsonl"
    summary = survey_sources(base_config, out, limit=2)
    assert summary.files_scanned == 2
    assert len(load_survey(out)) == 2


# --- stratified sampling -------------------------------------------------------


def _records(n_per_stratum: int = 10) -> list[SurveyRecord]:
    combos = [
        ("pdf", "native", "en"),
        ("pdf", "scanned", "en"),
        ("pdf", "native", "fr"),
        ("epub", "native", "en"),
        ("epub", "native", "de"),
    ]
    recs: list[SurveyRecord] = []
    for s, (fmt, ocr, lang) in enumerate(combos):
        for i in range(n_per_stratum):
            recs.append(_rec(f"/books/{fmt}/{s:02d}-{i:03d}.pdf", fmt, ocr, lang, pages=20 + i))
    return recs


def _kwargs(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"seed": 42, "target": 25, "created_at": 1.0, "survey_sha": "s" * 64}
    base.update(over)
    return base


def _sample(records: list[SurveyRecord], **over: Any) -> dict[str, Any]:
    """sample_manifest narrowed to dict[str, Any] for structural asserts."""
    return sample_manifest(records, **_kwargs(**over))


def test_sample_manifest_is_deterministic() -> None:
    m1 = _sample(_records())
    m2 = _sample(_records())
    assert m1 == m2

    other_seed = _sample(_records(), seed=7)
    assert other_seed["entries"] != m1["entries"]


def test_sample_manifest_totals_and_min_coverage() -> None:
    m = _sample(_records(), target=13)
    entries = m["entries"]
    assert len(entries) == 13
    assert m["totals"]["sampled"] == 13
    assert m["totals"]["strata"] == 5
    assert m["totals"]["candidates"] == 50
    strata: dict[str, dict[str, int]] = m["strata"]
    assert sum(s["sampled"] for s in strata.values()) == 13
    assert all(s["sampled"] >= 1 for s in strata.values())
    for _key, s in strata.items():
        assert s["sampled"] <= s["population"]


def test_sample_manifest_target_below_strata_count() -> None:
    m = _sample(_records(), target=3)
    assert len(m["entries"]) == 3
    strata: dict[str, dict[str, int]] = m["strata"]
    selected = [k for k, s in strata.items() if s["sampled"] == 1]
    assert len(selected) == 3  # exactly one book from three strata


def test_sample_manifest_excludes_unprofileable_records() -> None:
    recs = [*_records(),
            _rec("/books/broken.pdf", error="open: broken"),
            _rec("/books/ghost.pdf", pages=None),
            _rec("/books/notes.txt", fmt="invalid", error="magic bytes do not match extension"),
            _rec("/books/gone.pdf", fmt="unreadable", error="stat: no"),
            ]
    m = _sample(recs)
    assert m["totals"]["candidates"] == 50
    assert all(e["path"].startswith("/books/pdf/") or e["path"].startswith("/books/epub/")
               for e in m["entries"])


def test_sample_manifest_entries_sorted_by_path() -> None:
    m = _sample(_records(), target=40)
    paths = [e["path"] for e in m["entries"]]
    assert paths == sorted(paths)


def test_sample_manifest_page_cap_propagates() -> None:
    m = _sample(_records(), page_cap=32)
    assert m["page_cap"] == 32
    assert _sample(_records())["page_cap"] is None
    with pytest.raises(ValueError, match="target"):
        _sample(_records(), target=0)
    with pytest.raises(ValueError, match="page_cap"):
        _sample(_records(), page_cap=0)


def test_format_manifest_summary_lists_strata() -> None:
    m = _sample(_records(), target=8)
    lines = format_manifest_summary(m)
    assert "candidates:" in lines[0] and "sampled:" in lines[0] and "seed 42" in lines[0]
    for key in m["strata"]:
        assert any(key in line for line in lines)


# --- pilot run (phase 2) -----------------------------------------------------


def _pilot_books(base_config: Config) -> list[Path]:
    """A native-text PDF (2 pages) and an EPUB (2 sections) in the source root."""
    src_root = base_config.paths.source_roots[0]
    src_root.mkdir(parents=True)
    pdf = src_root / "pilot-a.pdf"
    make_pdf(pdf, [_EN_PAGE, _EN_PAGE_2])
    epub = src_root / "pilot-b.epub"
    make_epub(epub, [("Ch1", ["The cat sat on the mat."]), ("Ch2", ["The dog was in the yard."])])
    return [pdf, epub]


def _pilot_manifest(
    books: list[Path], tmp_path: Path, *, page_cap: int | None = None
) -> Path:
    """Profile *books*, sample all of them into a manifest, and write it."""
    records = [profile_candidate(p) for p in books]
    assert all(r.error is None for r in records)
    manifest = sample_manifest(
        records, seed=42, target=len(books), page_cap=page_cap, created_at=1.0, survey_sha="x" * 64
    )
    out = tmp_path / "manifest.json"
    write_manifest(manifest, out)
    return out


def test_sandbox_config_isolates_writable_roots(base_config: Config, tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    out = sandbox_config(base_config, sandbox, page_cap=3)
    root = sandbox.resolve()
    assert out.paths.archive_root == root / "archive"
    assert out.paths.artifact_root == root / "artifacts"
    assert out.paths.state_root == root / "state"
    assert out.paths.qdrant_root == root / "qdrant"
    assert out.paths.scratch_root == root / "scratch"
    assert out.services.qdrant_path == str(root / "qdrant")
    assert out.pilot.page_cap == 3
    # Source roots and measurement settings are inherited, and the input is
    # not mutated (the real config keeps its own roots and cap).
    assert out.paths.source_roots == base_config.paths.source_roots
    assert out.extraction == base_config.extraction
    assert base_config.paths.state_root != out.paths.state_root
    assert base_config.services.qdrant_path is None
    assert base_config.pilot.page_cap is None


def test_run_pilot_end_to_end_fake(base_config: Config, tmp_path: Path) -> None:
    base_config.embedding.fake = True
    books = _pilot_books(base_config)
    manifest = _pilot_manifest(books, tmp_path)
    sandbox = tmp_path / "sandbox"

    m = run_pilot(
        base_config,
        manifest,
        sandbox,
        qdrant=FakeQdrant(base_config.embedding.dimensions),
        vram_interval=0.1,
    )

    # Catalog + pipeline completeness.
    assert m.documents == 2
    assert m.revisions == 2
    assert m.registered_new == 2
    assert m.registered_aliases == 0
    assert m.unchanged == 0
    assert m.invalid == 0 and m.missing == 0
    assert m.units == 4  # 2 PDF pages + 2 EPUB sections
    assert m.unit_chars > 0
    assert m.chunks > 0
    assert m.chunk_tokens > 0
    assert m.chunk_chars > 0
    assert m.publications_active == 2
    assert m.extraction_failures == 0
    assert m.failed_jobs == []
    assert m.pending_jobs == 0

    # Stage accounting is consistent and every stage fully succeeded.
    assert set(m.stages) >= {"extract", "chunk", "embed", "publish"}
    assert m.completed_jobs == sum(int(s["jobs"]) for s in m.stages.values())
    for s in m.stages.values():
        assert s["failed"] == 0.0
        assert s["succeeded"] == s["jobs"]
        # Sum of job lifetimes >= the longest single job (rounding to 0.1s).
        assert s["seconds"] >= s["seconds_max"] - 0.1 >= 0.0

    # Per-book breakdown adds up to the totals.
    assert len(m.per_book) == 2
    assert {str(b["format"]) for b in m.per_book} == {"pdf", "epub"}
    assert sum(cast("int", b["units"]) for b in m.per_book) == m.units
    assert sum(cast("int", b["chunks"]) for b in m.per_book) == m.chunks

    # Measurements are populated (unit-safe: vram is {} without nvidia-smi).
    assert m.peak_rss_mb > 0.0
    assert m.finished_at >= m.started_at
    assert m.registration_seconds >= 0.0 and m.worker_seconds >= 0.0
    assert m.embed_model_revision == "fake-v1"
    assert m.embed_dimensions == base_config.embedding.dimensions
    for entry in m.vram.values():
        assert set(entry) == {"baseline_mb", "peak_mb"}
        assert entry["peak_mb"] >= entry["baseline_mb"]

    # Isolation: everything landed under the sandbox; the real roots are clean.
    assert m.sandbox_root == sandbox.resolve().as_posix()
    assert (sandbox / "state" / "library.db").is_file()
    assert not (base_config.paths.state_root / "library.db").exists()
    assert m.disk["archive"]["delta_bytes"] > 0
    assert m.disk["state"]["delta_bytes"] > 0
    assert m.disk["artifacts"]["bytes"] >= 0 and m.disk["qdrant"]["bytes"] >= 0


def test_run_pilot_page_cap_limits_units(base_config: Config, tmp_path: Path) -> None:
    base_config.embedding.fake = True
    books = _pilot_books(base_config)
    manifest = _pilot_manifest(books, tmp_path, page_cap=1)
    m = run_pilot(
        base_config,
        manifest,
        tmp_path / "sandbox",
        qdrant=FakeQdrant(base_config.embedding.dimensions),
        vram_interval=0.1,
    )
    assert m.page_cap == 1
    assert m.units == 2  # one page of the PDF, one section of the EPUB
    for b in m.per_book:
        assert b["units"] == 1


def test_run_pilot_rerun_is_idempotent(base_config: Config, tmp_path: Path) -> None:
    base_config.embedding.fake = True
    books = _pilot_books(base_config)
    manifest = _pilot_manifest(books, tmp_path)
    sandbox = tmp_path / "sandbox"
    dims = base_config.embedding.dimensions
    m1 = run_pilot(base_config, manifest, sandbox, qdrant=FakeQdrant(dims), vram_interval=0.1)
    m2 = run_pilot(base_config, manifest, sandbox, qdrant=FakeQdrant(dims), vram_interval=0.1)

    # Nothing re-registered, nothing re-enqueued, state unchanged.
    assert m2.registered_new == 0
    assert m2.registered_aliases == 0
    assert m2.unchanged == 2
    assert m2.completed_jobs == 0
    assert m2.pending_jobs == 0
    assert m2.documents == m1.documents == 2
    assert m2.chunks == m1.chunks
    assert m2.publications_active == 2


def test_run_pilot_rejects_wrong_schema(base_config: Config, tmp_path: Path) -> None:
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"schema": "nope", "entries": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        run_pilot(base_config, bad, tmp_path / "sandbox")


def test_write_pilot_report_roundtrip(base_config: Config, tmp_path: Path) -> None:
    base_config.embedding.fake = True
    books = _pilot_books(base_config)
    manifest = _pilot_manifest(books, tmp_path)
    m = run_pilot(
        base_config,
        manifest,
        tmp_path / "sandbox",
        qdrant=FakeQdrant(base_config.embedding.dimensions),
        vram_interval=0.1,
    )
    out = tmp_path / "pilot_run.json"
    write_pilot_report(m, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == PILOT_REPORT_SCHEMA
    assert payload["documents"] == 2
    assert payload["chunks"] == m.chunks
    assert payload["stages"] == m.stages
    assert payload["per_book"] == m.per_book
    assert payload["manifest_sha256"] == m.manifest_sha256
