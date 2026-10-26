"""Tests for the command-line interface.

Uses Typer's CliRunner so no subprocess is spawned.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner, Result

from inferbench import __version__
from inferbench.cli import app

runner = CliRunner()


def _combined(result: Result) -> str:
    """Return stdout and stderr concatenated for assertions on either stream."""
    return (result.stdout or "") + (result.stderr or "")


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_spec_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "models", "run", "compare", "report", "validate"):
        assert command in result.stdout


class TestDoctor:
    def test_runs_without_gpu(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        output = result.stdout
        assert "InferBench environment" in output
        assert "Accelerator" in output
        # The accelerator section must be explicit about availability.
        assert "UNAVAILABLE" in output or "available" in output

    def test_json_output_is_valid(self) -> None:
        result = runner.invoke(app, ["doctor", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert "gpu" in payload
        assert "python_version" in payload
        assert isinstance(payload["gpu"]["available"], bool)

    def test_accepts_disk_path(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["doctor", "--disk-path", str(tmp_path)])
        assert result.exit_code == 0


class TestUnimplementedCommands:
    def test_run_states_phase(self) -> None:
        result = runner.invoke(app, ["run", "--config", "configs/baseline.yaml"])
        assert result.exit_code == 2
        assert "Phase 1" in _combined(result)

    def test_compare_states_phase(self) -> None:
        result = runner.invoke(app, ["compare", "a", "b"])
        assert result.exit_code == 2
        assert "Phase" in _combined(result)

    def test_validate_states_phase(self) -> None:
        result = runner.invoke(app, ["validate", "run-1"])
        assert result.exit_code == 2
        assert "Phase" in _combined(result)

    def test_models_states_phase(self) -> None:
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 2
        assert "Phase" in _combined(result)
