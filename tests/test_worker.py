"""Worker loop (PRD §7): claim durable jobs, run the extract stage, classify
failures (permanent vs. transient), and heartbeat the lease between units.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fixtures import make_pdf
from library_rag.config import Config
from library_rag.db import Database
from library_rag.identity import make_task_key
from library_rag.jobs import Claimed, Jobs
from library_rag.scan import scan_roots
from library_rag.worker import run_worker


@pytest.fixture
def src(base_config: Config) -> Path:
    root = base_config.paths.source_roots[0]
    root.mkdir(parents=True)
    return root


def _last_job(db: Database) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM jobs ORDER BY job_id DESC")
    assert row is not None
    return dict(row)


def test_worker_end_to_end(state_db: Database, base_config: Config, src: Path) -> None:
    make_pdf(src / "A.pdf", ["a page of text " * 5])
    jobs = Jobs(state_db)
    report = scan_roots(state_db, base_config, jobs)[0]
    assert report.new_documents == 1

    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1
    assert jobs.counts() == {"succeeded": 1}

    run = state_db.query_one("SELECT state, unit_count FROM extraction_runs")
    assert run is not None
    assert run["state"] == "succeeded"
    assert int(run["unit_count"]) == 1
    manifest = json.loads(_last_job(state_db)["output_manifest"])
    assert set(manifest) == {"run_id", "units"}
    assert manifest["units"] == 1


def test_worker_corrupt_permanent_failed(state_db: Database, base_config: Config, src: Path) -> None:
    # Magic bytes say PDF; PyMuPDF cannot open it => permanent "corrupt".
    (src / "bad.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)

    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1

    assert jobs.counts() == {"permanent_failed": 1}
    job = _last_job(state_db)
    assert job["error_category"] == "corrupt"


def test_worker_heartbeats_between_units(
    state_db: Database, base_config: Config, src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 25 pages => a heartbeat at pages 10 and 20 (indices 9 and 19).
    make_pdf(src / "big.pdf", [f"page {i} " + "word " * 10 for i in range(25)])
    jobs = Jobs(state_db)
    scan_roots(state_db, base_config, jobs)

    beats: list[int] = []

    def fake_heartbeat(self: Jobs, job: Claimed, ttl: float, now: float | None = None) -> None:
        beats.append(job.job_id)

    monkeypatch.setattr(Jobs, "heartbeat", fake_heartbeat)
    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1

    assert len(beats) == 2
    assert jobs.counts() == {"succeeded": 1}


def test_worker_unknown_stage_permanent(state_db: Database, base_config: Config) -> None:
    jobs = Jobs(state_db)
    jobs.enqueue(make_task_key("embed", "x", "v1"), "embed", input_id="x", input_version="v1")

    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1

    assert jobs.counts() == {"permanent_failed": 1}
    assert _last_job(state_db)["error_category"] == "unknown_stage"


def test_worker_missing_revision_permanent(state_db: Database, base_config: Config) -> None:
    jobs = Jobs(state_db)
    jobs.enqueue(
        make_task_key("extract", "no-such-rev", "0" * 64),
        "extract",
        input_id="no-such-rev",
        input_version="0" * 64,
    )

    assert run_worker(state_db, base_config, once=True, poll_delay=0) == 1

    assert jobs.counts() == {"permanent_failed": 1}
    job = _last_job(state_db)
    assert job["error_category"] == "missing_source"
