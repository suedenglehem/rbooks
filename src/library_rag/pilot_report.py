"""M6 pilot capacity report (PRD §12): the operator-approval document.

Aggregates the four pilot inputs into one report:

* the corpus **survey** (what the library actually contains),
* the stratified **manifest** (what the pilot measured),
* the **pilot run** measurements (what the pipeline costs per unit), and
* the **question bank** (evaluation coverage status).

Everything in the ``measured`` section is a measurement taken by ``pilot run``;
the ``projections`` section scales the pilot to the full surveyed corpus with
the pilot's measured per-unit rates and a bootstrap interval over the
per-book outcomes. Projections are estimates and say so. This module never
runs the pipeline — it loads inputs and computes.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Config
from .pilot import SurveyRecord
from .questions import QUESTION_CATEGORIES, QuestionEntry

__all__ = [
    "REPORT_SCHEMA",
    "build_report",
    "render_markdown",
    "write_report",
]

REPORT_SCHEMA = "pilot-report/1"

_BOOTSTRAP_REPS = 200
_BOOTSTRAP_SEED = 42


# --- small helpers -----------------------------------------------------------


def _cpu_quota() -> int | None:
    """The cgroup v2 CPU quota in whole cores, or None when not readable."""
    try:
        max_str, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except (OSError, ValueError):
        return None
    if max_str == "max":
        return None
    try:
        return max(1, round(int(max_str) / int(period)))
    except (ValueError, ZeroDivisionError):
        return None


def _free_bytes(path: Path) -> int | None:
    """Free bytes on the volume holding *path*, or None when not checkable."""
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    return int(st.f_bavail) * int(st.f_frsize)


def _bootstrap_ci(
    ratios: Sequence[float], scale: float, *, reps: int = _BOOTSTRAP_REPS, seed: int = _BOOTSTRAP_SEED
) -> list[float] | None:
    """95% CI of ``scale * mean(ratio)`` by nonparametric bootstrap.

    Returns None when the sample is too small to mean anything (< 5 books).
    """
    n = len(ratios)
    if n < 5:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(reps):
        means.append(sum(ratios[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[min(n - 1, int(0.025 * reps))]
    hi = means[min(n - 1, int(0.975 * reps))]
    return [round(lo * scale), round(hi * scale)]


def _gb(value: float | int) -> str:
    return f"{value / 1e9:.1f} GB"


def _dur(seconds: float) -> str:
    if seconds >= 48 * 3600:
        return f"{seconds / 86400:.1f} days"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


# --- sections ----------------------------------------------------------------


def _corpus_section(records: Sequence[SurveyRecord]) -> dict[str, Any]:
    by_format: dict[str, int] = {}
    by_language: dict[str, int] = {}
    by_ocr: dict[str, int] = {}
    size = 0
    est_units = 0
    units_books = 0
    profiled = 0
    for r in records:
        by_format[r.format] = by_format.get(r.format, 0) + 1
        by_language[r.language] = by_language.get(r.language, 0) + 1
        by_ocr[r.ocr_class] = by_ocr.get(r.ocr_class, 0) + 1
        size += r.size
        if r.error is None and r.pages is not None:
            profiled += 1
            est_units += r.pages
            units_books += 1
    return {
        "candidates": len(records),
        "profiled": profiled,
        "unprofiled": len(records) - profiled,
        "by_format": dict(sorted(by_format.items())),
        "by_language": dict(sorted(by_language.items())),
        "by_ocr_class": dict(sorted(by_ocr.items())),
        "size_bytes": size,
        "est_units": est_units,
        "est_units_books": units_books,
    }


def _sample_section(manifest: dict[str, Any]) -> dict[str, Any]:
    totals = manifest.get("totals", {})
    return {
        "target": manifest.get("target"),
        "seed": manifest.get("seed"),
        "page_cap": manifest.get("page_cap"),
        "survey_sha256": manifest.get("survey_sha256"),
        "candidates": totals.get("candidates"),
        "strata": totals.get("strata"),
        "sampled": totals.get("sampled"),
        "stratum_detail": manifest.get("strata", {}),
    }


def _measured_section(run: dict[str, Any]) -> dict[str, Any]:
    """The run report's fields, verbatim (all already measurements)."""
    keys = (
        "sandbox_root",
        "manifest_sha256",
        "page_cap",
        "seconds",
        "registration_seconds",
        "worker_seconds",
        "completed_jobs",
        "documents",
        "units",
        "chunks",
        "chunk_tokens",
        "chunk_chars",
        "publications_active",
        "extraction_failures",
        "pending_jobs",
        "peak_rss_mb",
        "embed_model_revision",
        "embed_dimensions",
    )
    out: dict[str, Any] = {k: run.get(k) for k in keys}
    out["stages"] = run.get("stages", {})
    out["per_book"] = run.get("per_book", [])
    out["disk"] = run.get("disk", {})
    out["vram"] = run.get("vram", {})
    out["failed_jobs"] = run.get("failed_jobs", [])
    return out


def _projections_section(corpus: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any] | None:
    units = measured.get("units") or 0
    chunks = measured.get("chunks") or 0
    tokens = measured.get("chunk_tokens") or 0
    total_units = corpus.get("est_units") or 0
    if units <= 0 or total_units <= 0:
        return None

    est_chunks = total_units * (chunks / units)
    est_tokens = total_units * (tokens / units)
    per_stage: dict[str, Any] = {}
    eta = 0.0
    for stage, s in sorted(measured.get("stages", {}).items()):
        stage_seconds = float(s.get("seconds", 0.0))
        full = total_units * (stage_seconds / units)
        eta += full
        per_stage[stage] = {
            "pilot_seconds": stage_seconds,
            "full_seconds_est": round(full),
            "full_duration_est": _dur(full),
        }

    embed_seconds = float(measured.get("stages", {}).get("embed", {}).get("seconds", 0.0))
    tok_s = round(tokens / embed_seconds, 1) if embed_seconds > 0 else None

    # Bootstrap over per-book ratios (chunks per unit, tokens per unit).
    c_ratios: list[float] = []
    t_ratios: list[float] = []
    for b in measured.get("per_book", []):
        u = b.get("units") or 0
        if u <= 0:
            continue
        c_ratios.append((b.get("chunks") or 0) / u)
        t_ratios.append((b.get("tokens") or 0) / u)

    assumptions = [
        "Per-unit rates come from a single stratified pilot run; the bootstrap "
        "resamples the pilot's per-book outcomes, so the intervals reflect "
        "book-to-book variance, not machine or model variance.",
        "Stages run serially per job, so the full-run ETA is the sum of the "
        "per-stage wall estimates (queue waits included), i.e. a single-worker "
        "wall clock.",
        "Archive storage assumes one archived copy per book; the Qdrant "
        "estimate adds a two-generation rebuild headroom (2x current bytes).",
    ]
    page_cap = measured.get("page_cap")
    if page_cap:
        assumptions.append(
            f"The pilot applied a page cap of {page_cap}; books longer than the cap "
            "were only partially measured, so their full-length per-unit rates "
            "are extrapolated, not observed."
        )

    return {
        "pilot_units": units,
        "pilot_chunks": chunks,
        "pilot_tokens": tokens,
        "full_units_est": total_units,
        "full_chunks_est": round(est_chunks),
        "full_tokens_est": round(est_tokens),
        "bootstrap_ci_95": {
            "chunks": _bootstrap_ci(c_ratios, total_units),
            "tokens": _bootstrap_ci(t_ratios, total_units),
        },
        "per_stage": per_stage,
        "full_run_seconds_est": round(eta),
        "full_run_duration_est": _dur(eta),
        "embed_tokens_per_s": tok_s,
        "assumptions": assumptions,
    }


def _storage_section(
    corpus: dict[str, Any], measured: dict[str, Any], projections: dict[str, Any] | None
) -> dict[str, Any] | None:
    chunks = measured.get("chunks") or 0
    docs = measured.get("documents") or 0
    disk = measured.get("disk", {})
    if chunks <= 0 or docs <= 0:
        return None
    qdrant = disk.get("qdrant", {})
    state = disk.get("state", {})
    artifacts = disk.get("artifacts", {})
    est_chunks = (projections or {}).get("full_chunks_est", chunks)
    est_docs = corpus.get("profiled", docs)
    est_qdrant = (qdrant.get("bytes", 0) / chunks) * est_chunks * 2
    est_state = (state.get("delta_bytes", 0) / docs) * est_docs
    est_artifacts = (artifacts.get("bytes", 0) / chunks) * est_chunks
    components = {
        "archive (one copy per book)": corpus.get("size_bytes", 0),
        "qdrant index (2-generation headroom)": round(est_qdrant),
        "state database": round(est_state),
        "artifacts": round(est_artifacts),
    }
    free = None
    sandbox = measured.get("sandbox_root")
    if sandbox:
        free = _free_bytes(Path(str(sandbox)) / "state")
    return {
        "components": components,
        "total_bytes": round(sum(components.values())),
        "total_gb": round(sum(components.values()) / 1e9, 1),
        "free_bytes_on_sandbox_state_volume": free,
    }


def _questions_section(entries: Sequence[QuestionEntry] | None) -> dict[str, Any]:
    by_category = dict.fromkeys(QUESTION_CATEGORIES, 0)
    total = 0
    answerable = 0
    for e in entries or ():
        by_category[e.category] = by_category.get(e.category, 0) + 1
        total += 1
        answerable += 1 if e.answerable else 0
    if total == 0:
        status = "empty — labeling not started"
    elif total < 100:
        status = f"in progress ({total}/100, PRD target 100-200)"
    else:
        status = "PRD target met"
    return {
        "total": total,
        "answerable": answerable,
        "by_category": by_category,
        "status": status,
    }


def _frozen_section(
    measured: dict[str, Any] | None, config: Config | None
) -> dict[str, Any] | None:
    if measured is None:
        return None
    notes: list[str] = []
    quota = _cpu_quota()
    notes.append(
        f"Single worker process (1): the machine's cgroup CPU quota is "
        f"{quota if quota is not None else 'unknown'} core(s); more workers would "
        "not add throughput and would contend for the embedder."
    )
    notes.append(
        f"Embedder {measured.get('embed_model_revision')} "
        f"({measured.get('embed_dimensions')} dims) is already resident on a GPU; "
        "keep it running and never restart it for this work."
    )
    out: dict[str, Any] = {
        "embed_model_revision": measured.get("embed_model_revision"),
        "embed_dimensions": measured.get("embed_dimensions"),
        "page_cap": measured.get("page_cap"),
        "workers": 1,
        "cpu_quota": quota,
        "notes": notes,
    }
    if config is not None:
        c = config.chunking
        out["chunking"] = {
            "target_tokens": c.target_tokens,
            "overlap_tokens": c.overlap_tokens,
            "max_tokens": c.max_tokens,
            "tokenizer": c.tokenizer,
        }
    return out


# --- assembly ----------------------------------------------------------------


def build_report(
    *,
    survey: Sequence[SurveyRecord] | None = None,
    manifest: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
    questions: Sequence[QuestionEntry] | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """Assemble the report; each section is present only for inputs given.

    ``projections`` and ``storage`` need both survey and run; they are omitted
    (not zero-filled) when either is missing, so an incomplete report can
    never pretend to a number it cannot support.
    """
    corpus = _corpus_section(survey) if survey is not None else None
    measured = _measured_section(run) if run is not None else None
    projections = (
        _projections_section(corpus, measured)
        if corpus is not None and measured is not None
        else None
    )
    storage = (
        _storage_section(corpus, measured, projections)
        if corpus is not None and measured is not None
        else None
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_at": time.time(),
        "inputs": {
            "survey_records": len(survey) if survey is not None else None,
            "manifest": bool(manifest),
            "run": bool(run),
            "questions": len(questions) if questions is not None else None,
        },
        "corpus": corpus,
        "sample": _sample_section(manifest) if manifest is not None else None,
        "measured": measured,
        "projections": projections,
        "storage": storage,
        "questions": _questions_section(questions),
        "frozen_config": _frozen_section(measured, config),
    }
    return report


# --- rendering ----------------------------------------------------------------


def _kv_table(rows: Sequence[tuple[str, str]]) -> list[str]:
    return [f"| {k} | {v} |" for k, v in rows] + ["|---|---|"]


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as Markdown (the operator-approval document)."""
    lines: list[str] = ["# Pilot Capacity Report (PRD §12)", ""]
    gen = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(report["generated_at"]))
    lines.append(f"Generated: {gen}")
    inputs = report["inputs"]
    parts = []
    if inputs["survey_records"] is not None:
        parts.append(f"survey ({inputs['survey_records']} records)")
    if inputs["manifest"]:
        parts.append("manifest")
    if inputs["run"]:
        parts.append("pilot run")
    if inputs["questions"] is not None:
        parts.append(f"question bank ({inputs['questions']})")
    lines.append(f"Inputs: {', '.join(parts) if parts else 'none'}")
    lines.append("")

    corpus = report.get("corpus")
    if corpus:
        lines += ["## 1. Corpus", ""]
        lines += _kv_table(
            [
                ("candidates", str(corpus["candidates"])),
                ("profiled (units known)", str(corpus["profiled"])),
                ("unprofiled / invalid", str(corpus["unprofiled"])),
                ("total size", _gb(corpus["size_bytes"])),
                ("estimated full-run units", f"{corpus['est_units']:,} ({corpus['est_units_books']:,} books)"),
            ]
        )
        lines += [""]
        for name, key in (("format", "by_format"), ("language", "by_language"), ("ocr class", "by_ocr_class")):
            bits = ", ".join(f"{k}: {v}" for k, v in corpus[key].items())
            lines.append(f"- **{name}** — {bits}")
        lines.append("")

    sample = report.get("sample")
    if sample:
        lines += ["## 2. Sample", ""]
        lines += _kv_table(
            [
                ("target", str(sample["target"])),
                ("seed", str(sample["seed"])),
                ("page cap", str(sample["page_cap"])),
                ("survey sha256", str(sample["survey_sha256"])[:16] + "…"),
                ("candidates / strata / sampled",
                 f"{sample['candidates']} / {sample['strata']} / {sample['sampled']}"),
            ]
        )
        lines += [""]
        lines.append("| stratum | population | sampled |")
        lines.append("|---|---|---|")
        for k in sorted(sample["stratum_detail"]):
            s = sample["stratum_detail"][k]
            lines.append(f"| {k} | {s['population']} | {s['sampled']} |")
        lines.append("")

    measured = report.get("measured")
    if measured:
        lines += ["## 3. Measured pilot run", ""]
        lines += _kv_table(
            [
                ("sandbox", str(measured["sandbox_root"])),
                ("documents", str(measured["documents"])),
                ("units", str(measured["units"])),
                ("chunks", f"{measured['chunks']:,}"),
                ("chunk tokens", f"{measured['chunk_tokens']:,}"),
                ("wall", _dur(measured["seconds"])),
                ("peak RSS", f"{measured['peak_rss_mb']} MB"),
                ("embed model", f"{measured['embed_model_revision']} ({measured['embed_dimensions']} dims)"),
            ]
        )
        lines += [""]
        lines.append("| stage | jobs | ok | failed | seconds | max |")
        lines.append("|---|---|---|---|---|---|")
        for stage, s in sorted(measured["stages"].items()):
            lines.append(
                f"| {stage} | {s['jobs']:.0f} | {s['succeeded']:.0f} | {s['failed']:.0f} "
                f"| {s['seconds']} | {s['seconds_max']} |"
            )
        lines += [""]
        if measured["per_book"]:
            lines.append("| book | format | units | chunks | tokens |")
            lines.append("|---|---|---|---|---|")
            for i, b in enumerate(measured["per_book"][:50], 1):
                lines.append(
                    f"| {i} | {b['format']} | {b['units']} | {b['chunks']} | {b['tokens']:,} |"
                )
            if len(measured["per_book"]) > 50:
                lines.append(f"| … | ({len(measured['per_book']) - 50} more) | | | |")
            lines.append("")
        vram = measured.get("vram") or {}
        if vram:
            bits = ", ".join(
                f"{k}: baseline {v['baseline_mb']} MB, peak {v['peak_mb']} MB"
                for k, v in sorted(vram.items())
            )
            lines.append(f"- **VRAM** — {bits}")
        disk = measured.get("disk") or {}
        if disk:
            bits = ", ".join(f"{k}: +{d['delta_bytes'] / 1e9:.2f} GB" for k, d in sorted(disk.items()))
            lines.append(f"- **disk growth** — {bits}")
        lines.append("")

    proj = report.get("projections")
    if proj:
        lines += ["## 4. Full-corpus projections (estimates)", ""]
        lines += _kv_table(
            [
                ("full units (est.)", f"{proj['full_units_est']:,}"),
                ("full chunks (est.)", f"{proj['full_chunks_est']:,}"),
                ("full tokens (est.)", f"{proj['full_tokens_est']:,}"),
                ("full-run ETA (single worker)", proj["full_run_duration_est"]),
                ("embed throughput", f"{proj['embed_tokens_per_s']} tok/s" if proj["embed_tokens_per_s"] else "n/a"),
            ]
        )
        lines += [""]
        ci = proj["bootstrap_ci_95"]
        if ci["chunks"]:
            lines.append(
                f"95% bootstrap CI (per-book resample): chunks "
                f"{ci['chunks'][0]:,}-{ci['chunks'][1]:,}"
            )
        if ci["tokens"]:
            lines.append(
                f"          tokens "
                f"{ci['tokens'][0]:,}-{ci['tokens'][1]:,}"
            )
        lines += [""]
        lines.append("| stage | pilot s | full est. | duration |")
        lines.append("|---|---|---|---|")
        for stage, s in sorted(proj["per_stage"].items()):
            lines.append(
                f"| {stage} | {s['pilot_seconds']} | {s['full_seconds_est']:,} s | {s['full_duration_est']} |"
            )
        lines.append("")

    storage = report.get("storage")
    if storage:
        lines += ["## 5. Storage estimate (full corpus)", ""]
        lines.append("| component | size |")
        lines.append("|---|---|")
        for k, v in storage["components"].items():
            lines.append(f"| {k} | {_gb(v)} |")
        lines.append(f"| **total** | **{_gb(storage['total_bytes'])}** |")
        lines.append("")
        if storage["free_bytes_on_sandbox_state_volume"] is not None:
            lines.append(f"Free on the sandbox state volume: {_gb(storage['free_bytes_on_sandbox_state_volume'])}")
            lines.append("")

    q = report.get("questions") or {}
    if q:
        lines += ["## 6. Question bank (operator-labeled)", ""]
        lines.append(f"- **{q['total']}** questions ({q['answerable']} answerable) — {q['status']}")
        bits = ", ".join(f"{k}: {v}" for k, v in q["by_category"].items())
        lines.append(f"- {bits}")
        lines.append("")

    frozen = report.get("frozen_config")
    if frozen:
        lines += ["## 7. Recommended frozen config", ""]
        lines += _kv_table(
            [
                ("embed model", str(frozen["embed_model_revision"])),
                ("dimensions", str(frozen["embed_dimensions"])),
                ("page cap", str(frozen["page_cap"])),
                ("workers", str(frozen["workers"])),
                ("cpu quota (cores)", str(frozen["cpu_quota"])),
            ]
        )
        if "chunking" in frozen:
            c = frozen["chunking"]
            lines.append(
                f"- chunking: target {c['target_tokens']}, overlap {c['overlap_tokens']}, "
                f"max {c['max_tokens']}, tokenizer {c['tokenizer']}"
            )
        for n in frozen["notes"]:
            lines.append(f"- {n}")
        lines.append("")

    if proj:
        lines += ["## 8. Uncertainty and assumptions", ""]
        for a in proj["assumptions"]:
            lines.append(f"- {a}")
        lines.append("")

    return "\n".join(lines)


def write_report(report: dict[str, Any], out_path: Path) -> None:
    """Write the Markdown report atomically (JSON: use ``json.dumps`` on the dict)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(render_markdown(report), encoding="utf-8")
    os.replace(tmp, out_path)
