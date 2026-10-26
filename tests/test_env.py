"""Tests for environment inspection.

These tests must pass on a machine with no GPU, which is exactly the case
InferBench is designed to tolerate.
"""

from __future__ import annotations

from pathlib import Path

from inferbench.env import (
    collect_environment,
    detect_disk,
    detect_gpu,
    environment_to_json,
)


class TestDetectGpu:
    def test_never_raises_and_is_self_consistent(self) -> None:
        info = detect_gpu()
        assert isinstance(info.available, bool)
        # A reason is always present, whether available or not.
        assert info.reason
        if info.available:
            assert info.device_count >= 1
        else:
            # Unavailable must not carry fabricated device data.
            assert info.devices == ()

    def test_unavailable_gpu_reports_reason(self) -> None:
        info = detect_gpu()
        if not info.available:  # expected in CPU-only CI
            assert len(info.reason) > 0


class TestDetectDisk:
    def test_reports_tmp(self) -> None:
        info = detect_disk(Path("/tmp"))
        assert info is not None
        assert info.total_gb > 0
        assert info.free_gb >= 0
        assert 0.0 <= info.used_percent <= 100.0

    def test_missing_path_returns_none(self) -> None:
        info = detect_disk("/nonexistent/path/that/cannot/exist")
        assert info is None


class TestCollectEnvironment:
    def test_full_snapshot_is_serializable(self) -> None:
        report = collect_environment(disk_path=".")
        assert report.python_version
        assert report.cpu_count >= 1
        assert report.platform
        # Must round-trip through JSON for metadata storage.
        as_json = environment_to_json(report)
        assert '"python_version"' in as_json
        assert '"gpu"' in as_json

    def test_reports_tracked_packages(self) -> None:
        report = collect_environment()
        names = {pkg.name for pkg in report.packages}
        assert "pydantic" in names
        # pydantic is a hard dependency, so it must be present.
        pydantic = next(pkg for pkg in report.packages if pkg.name == "pydantic")
        assert pydantic.available is True
        assert pydantic.version
