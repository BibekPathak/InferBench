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


class TestRunCommand:
    def test_run_end_to_end_smoke(self, tmp_path: Path) -> None:
        """`inferbench run` must produce a stored run and a report."""
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            "run_name: cli-smoke\n"
            "backend:\n  kind: mock\n  model: mock-model\n"
            "workload:\n  kind: fixed\n  num_requests: 4\n"
            "  prompt_tokens: 32\n  output_tokens: 4\n"
            "load:\n  concurrency: 2\n  warmup_requests: 1\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "runs"

        result = runner.invoke(
            app,
            ["run", "--config", str(config_path), "--output-dir", str(out_dir), "--no-charts"],
        )
        assert result.exit_code == 0, _combined(result)
        assert "Run complete" in result.stdout
        assert "synthetic" in result.stdout
        assert "Report:" in result.stdout

        # Artifacts exist on disk.
        runs = [p for p in out_dir.iterdir() if p.is_dir()]
        assert len(runs) == 1
        assert (runs[0] / "raw.jsonl").is_file()
        assert (runs[0] / "report.md").is_file()

    def test_run_json_output(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            "run_name: cli-json\n"
            "workload:\n  num_requests: 3\n  prompt_tokens: 16\n  output_tokens: 2\n"
            "load:\n  concurrency: 1\n  warmup_requests: 0\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "runs"),
                "--json",
                "--no-report",
            ],
        )
        assert result.exit_code == 0, _combined(result)
        payload = json.loads(result.stdout)
        assert payload["throughput"]["successful_requests"] == 3

    def test_run_missing_config_errors_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["run", "--config", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0


class TestReportCommand:
    def test_report_for_stored_run(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "runs"
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            "run_name: report-cli\n"
            "workload:\n  num_requests: 3\n  prompt_tokens: 16\n  output_tokens: 3\n"
            "load:\n  concurrency: 1\n  warmup_requests: 0\n",
            encoding="utf-8",
        )
        run_result = runner.invoke(
            app,
            ["run", "--config", str(config_path), "--output-dir", str(out_dir), "--no-report"],
        )
        assert run_result.exit_code == 0, _combined(run_result)

        run_id = next(p.name for p in out_dir.iterdir() if p.is_dir())
        report_result = runner.invoke(
            app, ["report", run_id, "--output-dir", str(out_dir), "--no-charts"]
        )
        assert report_result.exit_code == 0, _combined(report_result)
        assert "Report written" in report_result.stdout

    def test_report_unknown_run_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["report", "ghost-run", "--output-dir", str(tmp_path), "--no-charts"]
        )
        assert result.exit_code != 0
