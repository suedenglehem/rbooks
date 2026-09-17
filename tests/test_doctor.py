"""Doctor probes must be defensive: no GPU, no tesseract, no services, no mounts.

The M0 gate requires ``doctor`` to handle a host with no GPU cleanly. These tests
assert that by stubbing the environment probes and checking the report renders
without raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from library_rag import __version__, doctor
from library_rag.config import Config
from library_rag.doctor import (
    DoctorReport,
    MountInfo,
    detect_disks,
    detect_ocr,
    render_json,
    render_text,
    run_doctor,
)


@pytest.fixture
def no_services(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep doctor hermetic: never hit the network for qdrant/answer probes.
    monkeypatch.setattr(doctor, "detect_services", lambda cfg: [])


def test_run_doctor_returns_full_report(
    base_config: Config, no_services: None
) -> None:
    report = run_doctor(base_config)
    assert isinstance(report, DoctorReport)
    assert report.cpu.physical > 0
    assert report.ram.total_bytes > 0
    assert report.config.ok is True
    # Every configured root must appear in the paths-OK map.
    assert "state" in str(list(report.config.paths_ok))


def test_no_gpu_is_handled_cleanly(
    base_config: Config,
    no_services: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor, "detect_gpus", lambda: (False, [], "no GPU on this host"))
    report = run_doctor(base_config)
    assert report.gpus_available is False
    assert report.gpus == []
    text = render_text(report)
    assert "unavailable" in text
    # JSON rendering must still succeed with the empty GPU list.
    data = json.loads(render_json(report))
    assert data["gpus_available"] is False
    assert data["gpus"] == []


def test_ocr_missing_tesseract_is_clean(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stub the subprocess runner: tesseract absent on PATH -> probe returns None.
    monkeypatch.setattr(doctor, "_run", lambda cmd, timeout=10.0: None)
    info = detect_ocr()
    assert info.available is False
    assert info.version is None
    assert info.languages == []


def test_detect_disks_matches_longest_prefix_mount(tmp_path: Path) -> None:
    mounts = [
        MountInfo(device="/dev/sda1", mountpoint="/", fstype="ext4"),
        MountInfo(device="/dev/sdb1", mountpoint="/data", fstype="xfs"),
    ]
    target = tmp_path / "data" / "deep" / "path"  # under /data if it were; here under tmp
    # Use a path we know is under an actual mount: the filesystem root.
    disks = detect_disks([Path("/data")], mounts)
    assert len(disks) == 1
    assert disks[0].mountpoint == "/data"
    assert disks[0].device == "/dev/sdb1"
    # The unused tmp path should not leak in.
    assert target not in {Path(d.path) for d in disks}


def test_render_text_includes_all_sections(
    base_config: Config, no_services: None
) -> None:
    report = run_doctor(base_config)
    text = render_text(report)
    for section in ("GPUs", "Disks", "OCR", "Services", "Config"):
        assert section in text


def test_render_json_round_trips(base_config: Config, no_services: None) -> None:
    report = run_doctor(base_config)
    data = json.loads(render_json(report))
    assert data["library_rag_version"] == __version__
    assert isinstance(data["mounts"], list)
