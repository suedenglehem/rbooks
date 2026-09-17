"""Read-only environment diagnostics (`library-rag doctor`).

Every probe is defensive: it never raises and never mutates state. A host with
no GPUs, no Tesseract, or no services running reports those as *unavailable*
rather than failing. This is what makes the M0 gate "doctor handles no GPU
cleanly" hold.

The report is a plain data structure (:class:`DoctorReport`) so it can be
rendered as text (for humans), JSON (for scripts), and asserted on in tests.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psutil

from . import __version__
from .config import Config, ConfigError

# --- Data records ----------------------------------------------------------


@dataclass
class DistroInfo:
    id: str
    name: str
    version_id: str


@dataclass
class CpuInfo:
    physical: int
    logical: int
    arch: str


@dataclass
class RamInfo:
    total_bytes: int
    available_bytes: int


@dataclass
class GpuInfo:
    index: int
    name: str
    uuid: str
    mem_total_mib: int
    mem_free_mib: int
    driver_version: str


@dataclass
class MountInfo:
    device: str
    mountpoint: str
    fstype: str


@dataclass
class DiskInfo:
    path: str
    device: str
    mountpoint: str
    size_bytes: int
    used_bytes: int
    free_bytes: int


@dataclass
class OcrInfo:
    available: bool
    version: str | None = None
    languages: list[str] = field(default_factory=list)


@dataclass
class ServiceStatus:
    name: str
    url: str
    reachable: bool
    detail: str


@dataclass
class ConfigStatus:
    ok: bool
    detail: str
    paths_ok: dict[str, bool] = field(default_factory=dict)


@dataclass
class DoctorReport:
    library_rag_version: str
    python_version: str
    distro: DistroInfo
    cpu: CpuInfo
    ram: RamInfo
    gpus_available: bool
    gpus: list[GpuInfo]
    gpu_note: str
    mounts: list[MountInfo]
    disks: list[DiskInfo]
    ocr: OcrInfo
    services: list[ServiceStatus]
    config: ConfigStatus


# --- Probes ----------------------------------------------------------------


def detect_distro() -> DistroInfo:
    info: dict[str, str] = {}
    osrelease = Path("/etc/os-release")
    if osrelease.is_file():
        for line in osrelease.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                info[k.strip()] = v.strip().strip('"')
    return DistroInfo(
        id=info.get("ID", "unknown"),
        name=info.get("PRETTY_NAME", info.get("NAME", "unknown")),
        version_id=info.get("VERSION_ID", "unknown"),
    )


def detect_cpu() -> CpuInfo:
    logical = os.cpu_count() or 1
    # Physical core count from /proc/cpuinfo when available (Linux).
    physical = logical
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        sids: set[str] = set()
        pkg: set[str] = set()
        current_sid: str | None = None
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "physical id":
                pkg.add(value)
            elif key == "cpu core id" and current_sid is not None:
                sids.add(f"{current_sid}:{value}")
            elif key == "processor":
                current_sid = value
        if pkg:
            physical = len(pkg)
        elif sids:
            physical = len(sids)
    return CpuInfo(physical=physical, logical=logical, arch=platform.machine())


def detect_ram() -> RamInfo:
    vm = psutil.virtual_memory()
    return RamInfo(total_bytes=vm.total, available_bytes=vm.available)


def _run(cmd: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None


def detect_gpus() -> tuple[bool, list[GpuInfo], str]:
    """Return (available, gpus, note). Never raises."""
    probe = _run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"])
    if probe is None:
        return False, [], "nvidia-smi not found (no NVIDIA driver/CLI on PATH)"
    if probe.returncode != 0:
        return False, [], f"nvidia-smi failed: {probe.stderr.strip()[:200]}"

    full = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if full is None or full.returncode != 0:
        return False, [], "nvidia-smi present but GPU query failed"

    gpus: list[GpuInfo] = []
    for line in full.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    uuid=parts[2],
                    mem_total_mib=int(parts[3]),
                    mem_free_mib=int(parts[4]),
                    driver_version=parts[5],
                )
            )
        except ValueError:
            continue
    if not gpus:
        return False, [], "nvidia-smi returned no GPUs"
    return True, gpus, f"{len(gpus)} GPU(s) detected"


def read_mounts() -> list[MountInfo]:
    """Parse /proc/mounts (Linux). Returns [] on other platforms."""
    out: list[MountInfo] = []
    mounts = Path("/proc/mounts")
    if not mounts.is_file():
        return out
    for line in mounts.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        # Un-escape \040 (space) etc. in mountpoints.
        mountpoint = parts[1].replace("\\040", " ")
        out.append(MountInfo(device=parts[0], mountpoint=mountpoint, fstype=parts[2]))
    return out


def _mount_for(path: Path, mounts: list[MountInfo]) -> tuple[str, str]:
    """Return (device, mountpoint) for the longest-prefix mount covering path."""
    best = ("", "")
    best_len = -1
    for m in mounts:
        mp = m.mountpoint
        try:
            is_rel = path.is_relative_to(Path(mp))
        except ValueError:
            continue
        if is_rel and len(mp) > best_len:
            best = (m.device, mp)
            best_len = len(mp)
    return best


def detect_disks(paths: list[Path], mounts: list[MountInfo]) -> list[DiskInfo]:
    out: list[DiskInfo] = []
    seen: set[str] = set()
    for p in paths:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        device, mountpoint = _mount_for(p, mounts)
        try:
            du = shutil.disk_usage(str(p) if p.exists() else os.sep)
        except OSError:
            out.append(DiskInfo(str(p), device, mountpoint, 0, 0, 0))
            continue
        out.append(
            DiskInfo(
                path=str(p),
                device=device,
                mountpoint=mountpoint,
                size_bytes=du.total,
                used_bytes=du.used,
                free_bytes=du.free,
            )
        )
    return out


def detect_ocr() -> OcrInfo:
    version_probe = _run(["tesseract", "--version"])
    if version_probe is None:
        return OcrInfo(available=False)
    # tesseract --version prints to stderr.
    version = version_probe.stderr.strip().splitlines()[0] if version_probe.stderr.strip() else None
    langs_probe = _run(["tesseract", "--list-langs"])
    languages: list[str] = []
    if langs_probe is not None:
        # Output is either "List of available languages in <path> (n)" or a
        # bare list; take lines after the header.
        for line in (langs_probe.stdout + langs_probe.stderr).splitlines():
            line = line.strip()
            if not line or line.startswith("List of available"):
                continue
            # Language entries look like "eng" or "eng (English)"; take first token.
            languages.append(line.split()[0])
    return OcrInfo(available=True, version=version, languages=sorted(set(languages)))


def detect_services(cfg: Config) -> list[ServiceStatus]:
    svc = cfg.services
    checks: list[tuple[str, str, list[str]]] = [
        ("qdrant", f"http://{svc.qdrant_host}:{svc.qdrant_port}/healthz", ["/healthz"]),
        ("answer-model", f"http://{svc.answer_host}:{svc.answer_port}", ["/health", "/v1/models"]),
    ]
    out: list[ServiceStatus] = []
    timeout = 2.0
    for name, base, candidates in checks:
        try:
            for suffix in candidates:
                url = f"{base}{suffix}"
                try:
                    r = httpx.get(url, timeout=timeout)
                    detail = f"HTTP {r.status_code}"
                    out.append(ServiceStatus(name, url, r.status_code < 500, detail))
                    break
                except httpx.HTTPError as exc:
                    last_err = str(exc)
            else:
                out.append(ServiceStatus(name, base, False, last_err[:160]))
        except Exception as exc:
            out.append(ServiceStatus(name, base, False, f"error: {exc}"[:160]))
    return out


def _config_status(cfg: Config) -> ConfigStatus:
    paths_ok: dict[str, bool] = {}
    roots = [
        *cfg.paths.source_roots,
        cfg.paths.archive_root,
        cfg.paths.artifact_root,
        cfg.paths.state_root,
        cfg.paths.qdrant_root,
        cfg.paths.model_root,
        cfg.paths.scratch_root,
    ]
    for p in roots:
        # Existence is advisory in doctor: a not-yet-created root is not an error,
        # but a configured source root that is missing is worth flagging.
        paths_ok[str(p)] = p.exists()
    return ConfigStatus(ok=True, detail="config parsed and validated", paths_ok=paths_ok)


def run_doctor(cfg: Config | None = None) -> DoctorReport:
    """Assemble a full diagnostic report. Never raises on probe failures."""
    if cfg is None:
        try:
            cfg = _load_best_effort()
        except ConfigError:
            cfg = _default_cfg()

    gpus_available, gpus, gpu_note = detect_gpus()
    mounts = read_mounts()
    roots = [
        *cfg.paths.source_roots,
        cfg.paths.archive_root,
        cfg.paths.artifact_root,
        cfg.paths.state_root,
        cfg.paths.qdrant_root,
        cfg.paths.model_root,
        cfg.paths.scratch_root,
    ]
    disks = detect_disks(roots, mounts)

    try:
        config_status = _config_status(cfg)
    except Exception as exc:
        config_status = ConfigStatus(ok=False, detail=f"config error: {exc}")

    return DoctorReport(
        library_rag_version=__version__,
        python_version=platform.python_version(),
        distro=detect_distro(),
        cpu=detect_cpu(),
        ram=detect_ram(),
        gpus_available=gpus_available,
        gpus=gpus,
        gpu_note=gpu_note,
        mounts=mounts,
        disks=disks,
        ocr=detect_ocr(),
        services=detect_services(cfg),
        config=config_status,
    )


def _load_best_effort() -> Config:
    from .config import load_config

    return load_config()


def _default_cfg() -> Config:
    # A minimal, self-consistent config so doctor runs with no config file.
    from .config import Paths, Services

    base = Path("/var/lib/library-rag")
    return Config(
        paths=Paths(
            source_roots=[],
            archive_root=base / "archive",
            artifact_root=base / "artifacts",
            state_root=base / "state",
            qdrant_root=base / "qdrant",
            model_root=base / "models",
            scratch_root=base / "scratch",
        ),
        services=Services(),
    )


def _gb(n: float) -> str:
    return f"{n / (1024**3):.1f} GB"


def render_text(report: DoctorReport) -> str:
    lines: list[str] = []
    add = lines.append
    add("library-rag doctor")
    add(f"  version : {report.library_rag_version}")
    add(f"  python  : {report.python_version}")
    add(f"  distro  : {report.distro.name} (id={report.distro.id}, {report.distro.version_id})")
    add(f"  cpu     : {report.cpu.physical} physical / {report.cpu.logical} logical cores ({report.cpu.arch})")
    add(
        f"  ram     : {_gb(report.ram.total_bytes)} total, {_gb(report.ram.available_bytes)} available"
    )
    add("")
    add("GPUs")
    if report.gpus_available:
        add(f"  {report.gpu_note}")
        for g in report.gpus:
            add(
                f"    [{g.index}] {g.name}  uuid={g.uuid}  "
                f"{_gb(g.mem_total_mib * 1024**2)} total, {_gb(g.mem_free_mib * 1024**2)} free  "
                f"driver={g.driver_version}"
            )
    else:
        add(f"  unavailable: {report.gpu_note}")
    add("")
    add("Disks (configured roots)")
    for d in report.disks:
        if d.size_bytes:
            add(
                f"  {d.path}\n"
                f"      dev={d.device} mount={d.mountpoint} "
                f"{_gb(d.size_bytes)} size, {_gb(d.free_bytes)} free"
            )
        else:
            add(f"  {d.path}\n      dev={d.device} mount={d.mountpoint} (stat failed)")
    add("")
    add("OCR (Tesseract)")
    if report.ocr.available:
        langs = ", ".join(report.ocr.languages) if report.ocr.languages else "(none listed)"
        add(f"  {report.ocr.version or 'present'}  languages: {langs}")
    else:
        add("  unavailable: tesseract not on PATH")
    add("")
    add("Services")
    for s in report.services:
        state = "up" if s.reachable else "down"
        add(f"  {s.name:<12} {s.url:<45} {state}  ({s.detail})")
    add("")
    add("Config")
    add(f"  {report.config.detail}")
    for p, ok in report.config.paths_ok.items():
        mark = "present" if ok else "missing"
        add(f"    [{mark}] {p}")
    return "\n".join(lines)


def render_json(report: DoctorReport) -> str:
    import dataclasses
    import json

    return json.dumps(dataclasses.asdict(report), indent=2, default=str)
