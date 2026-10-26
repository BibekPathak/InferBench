"""Environment inspection for InferBench.

This module answers two questions:

1. *Can this host run a given experiment?* (the ``doctor`` command)
2. *What exactly was the host when this run was recorded?* (run metadata)

Every hardware probe degrades gracefully. On a CPU-only machine, GPU fields are
reported as unavailable with a reason — never as a fabricated value, and never
as a crash. This is what allows unit tests and the mock backend to run anywhere.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inferbench.logging_utils import get_logger, log_event

__all__ = [
    "GpuInfo",
    "DiskInfo",
    "PackageInfo",
    "EnvironmentReport",
    "collect_environment",
    "detect_gpu",
    "detect_disk",
    "environment_to_json",
]

_log = get_logger("env")

# Probing a broken NVIDIA driver can hang; bound it.
_NVIDIA_SMI_TIMEOUT_S = 10.0

# Packages whose versions materially affect reproducibility, if installed.
_TRACKED_PACKAGES = (
    "torch",
    "vllm",
    "sglang",
    "transformers",
    "numpy",
    "httpx",
    "pydantic",
)


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """Detected accelerator information.

    ``available`` is the single source of truth: when it is ``False``, every
    other field is ``None``/empty and ``reason`` explains why.
    """

    available: bool
    reason: str
    driver_version: str | None = None
    cuda_version: str | None = None
    devices: tuple[dict[str, Any], ...] = ()

    @property
    def device_count(self) -> int:
        return len(self.devices)


@dataclass(frozen=True, slots=True)
class DiskInfo:
    """Filesystem capacity for a path, used to gate model downloads."""

    path: str
    total_gb: float
    free_gb: float
    used_percent: float


@dataclass(frozen=True, slots=True)
class PackageInfo:
    """Name/version of an optional dependency, or a reason it is absent."""

    name: str
    version: str | None
    available: bool


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """A complete, serializable snapshot of the execution environment."""

    python_version: str
    python_implementation: str
    platform: str
    machine: str
    processor: str
    cpu_count: int
    total_memory_gb: float | None
    gpu: GpuInfo
    disk: DiskInfo | None
    packages: tuple[PackageInfo, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict, flattening nested dataclasses."""
        data = asdict(self)
        # ``asdict`` turns tuples into lists, which is what JSON wants.
        return data


def _run_quiet(cmd: list[str], timeout: float) -> tuple[bool, str]:
    """Run a command, returning ``(succeeded, stdout)`` without ever raising."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log_event(_log, logging.DEBUG, "probe failed", cmd=cmd[0], error=str(exc))
        return False, ""
    if proc.returncode != 0:
        log_event(_log, logging.DEBUG, "probe nonzero exit", cmd=cmd[0], returncode=proc.returncode)
        return False, proc.stdout
    return True, proc.stdout


def _parse_cuda_version(nvidia_smi_output: str) -> str | None:
    """Extract the CUDA version from the ``nvidia-smi`` header, if present."""
    for line in nvidia_smi_output.splitlines():
        if "CUDA Version" in line:
            # Format: "| NVIDIA-SMI 550.54  Driver Version: 550.54  CUDA Version: 12.4 |"
            _, _, tail = line.partition("CUDA Version:")
            version = tail.strip().split()[0] if tail.strip() else None
            return version or None
    return None


def detect_gpu() -> GpuInfo:
    """Probe for an NVIDIA GPU via ``nvidia-smi``.

    Returns an explicitly *unavailable* :class:`GpuInfo` rather than raising when
    no GPU, no driver, or no tooling is present. AMD/Intel accelerators are not
    auto-detected; a future backend adapter can add its own probe.
    """
    if shutil.which("nvidia-smi") is None:
        return GpuInfo(available=False, reason="nvidia-smi not found on PATH")

    ok, output = _run_quiet(["nvidia-smi"], timeout=_NVIDIA_SMI_TIMEOUT_S)
    if not ok or not output.strip():
        # Most common case on CPU-only hosts: the binary exists but no driver.
        return GpuInfo(
            available=False,
            reason="nvidia-smi present but no working NVIDIA driver (or no GPU)",
        )

    driver_version = None
    for line in output.splitlines():
        if "Driver Version" in line:
            _, _, tail = line.partition("Driver Version:")
            driver_version = tail.strip().split()[0] if tail.strip() else None
            break

    devices = _query_gpu_devices()
    if not devices:
        return GpuInfo(
            available=False,
            reason="nvidia-smi reported no usable GPU devices",
            driver_version=driver_version,
            cuda_version=_parse_cuda_version(output),
        )

    return GpuInfo(
        available=True,
        reason="ok",
        driver_version=driver_version,
        cuda_version=_parse_cuda_version(output),
        devices=tuple(devices),
    )


def _query_gpu_devices() -> list[dict[str, Any]]:
    """Return per-device details from ``nvidia-smi --query-gpu``, if supported."""
    fields = "name,memory.total,compute_cap"
    ok, output = _run_quiet(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        timeout=_NVIDIA_SMI_TIMEOUT_S,
    )
    if not ok:
        return []

    devices: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, memory_mb, compute_cap = parts
        try:
            memory_total_mb: float | None = float(memory_mb)
        except ValueError:
            memory_total_mb = None
        devices.append(
            {
                "name": name,
                "memory_total_mb": memory_total_mb,
                "compute_capability": compute_cap or None,
            }
        )
    return devices


def detect_disk(path: str | Path = ".") -> DiskInfo | None:
    """Report capacity for ``path``'s filesystem, or ``None`` if unreadable."""
    target = Path(path).expanduser()
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        log_event(_log, logging.DEBUG, "disk probe failed", path=str(target), error=str(exc))
        return None

    gib = 1024**3
    total_gb = usage.total / gib
    used_fraction = (usage.used / usage.total * 100.0) if usage.total else 0.0
    return DiskInfo(
        path=str(target),
        total_gb=round(total_gb, 2),
        free_gb=round(usage.free / gib, 2),
        used_percent=round(used_fraction, 1),
    )


def _package_info(name: str) -> PackageInfo:
    """Look up an installed package's version without importing heavy modules.

    We use ``importlib.metadata`` rather than ``import`` so that probing for
    ``torch`` does not load hundreds of megabytes of native code.
    """
    import importlib.metadata as md

    try:
        version = md.version(name)
    except md.PackageNotFoundError:
        return PackageInfo(name=name, version=None, available=False)
    except Exception as exc:  # pragma: no cover - defensive
        log_event(_log, logging.DEBUG, "package probe failed", package=name, error=str(exc))
        return PackageInfo(name=name, version=None, available=False)
    return PackageInfo(name=name, version=version, available=True)


def _total_memory_gb() -> float | None:
    """Best-effort total RAM in GiB using ``/proc/meminfo`` when available."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    return round(kb / (1024**2), 2)
        except (OSError, ValueError, IndexError):
            pass
    return None


def collect_environment(disk_path: str | Path = ".") -> EnvironmentReport:
    """Collect a full environment snapshot.

    Args:
        disk_path: Path whose filesystem should be measured for free space.
    """
    return EnvironmentReport(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        cpu_count=os.cpu_count() or 0,
        total_memory_gb=_total_memory_gb(),
        gpu=detect_gpu(),
        disk=detect_disk(disk_path),
        packages=tuple(_package_info(name) for name in _TRACKED_PACKAGES),
    )


def _json_default(value: Any) -> Any:  # pragma: no cover - helper for callers
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def environment_to_json(report: EnvironmentReport, indent: int = 2) -> str:
    """Serialize an environment report to pretty JSON."""
    return json.dumps(report.to_dict(), indent=indent, default=_json_default)
