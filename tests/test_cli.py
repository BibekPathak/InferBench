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


class TestModelsCommand:
    def test_lists_profiles(self) -> None:
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert "Model profiles" in result.stdout
        assert "mock" in result.stdout

    def test_json_output_is_valid(self) -> None:
        result = runner.invoke(app, ["models", "--json"])
        assert result.exit_code == 0
        profiles = json.loads(result.stdout)
        assert any(p["name"] == "mock" for p in profiles)

    def test_custom_profile_file(self, tmp_path: Path) -> None:
        profile_file = tmp_path / "profiles.yaml"
        profile_file.write_text(
            "custom:\n  backend_kind: vllm\n  model: org/custom\n", encoding="utf-8"
        )
        result = runner.invoke(app, ["models", "--profiles", str(profile_file)])
        assert result.exit_code == 0
        assert "custom" in result.stdout


def _write_config(tmp_path: Path, name: str, concurrency: int = 1, prompt_tokens: int = 64) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"run_name: {name}\n"
        "backend:\n  kind: mock\n  model: mock-model\n"
        f"workload:\n  kind: fixed\n  num_requests: 4\n  prompt_tokens: {prompt_tokens}\n"
        "  output_tokens: 4\n  seed: 0\n"
        f"load:\n  concurrency: {concurrency}\n  warmup_requests: 0\n  repetitions: 2\n",
        encoding="utf-8",
    )
    return path


def _produce_runs(tmp_path: Path) -> tuple[Path, str, str]:
    """Run two compatible experiments and return (out_dir, run_a, run_b)."""
    out_dir = tmp_path / "runs"
    for name, concurrency in (("run-a", 1), ("run-b", 4)):
        result = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(_write_config(tmp_path, name, concurrency)),
                "--output-dir",
                str(out_dir),
                "--no-report",
                "--no-charts",
            ],
        )
        assert result.exit_code == 0, _combined(result)
    a = next(p.name for p in out_dir.iterdir() if p.is_dir() and "run-a" in p.name)
    b = next(p.name for p in out_dir.iterdir() if p.is_dir() and "run-b" in p.name)
    return out_dir, a, b


class TestValidateCommand:
    def test_valid_run_exits_zero(self, tmp_path: Path) -> None:
        out_dir, a, _ = _produce_runs(tmp_path)
        result = runner.invoke(app, ["validate", a, "--output-dir", str(out_dir)])
        assert result.exit_code == 0, _combined(result)
        assert "VALID" in result.stdout

    def test_json_output(self, tmp_path: Path) -> None:
        out_dir, a, _ = _produce_runs(tmp_path)
        result = runner.invoke(app, ["validate", a, "--output-dir", str(out_dir), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["valid"] is True

    def test_corrupt_run_exits_nonzero(self, tmp_path: Path) -> None:
        out_dir, a, _ = _produce_runs(tmp_path)
        # Delete the raw evidence: the run can no longer be audited.
        (out_dir / a / "raw.jsonl").unlink()
        result = runner.invoke(app, ["validate", a, "--output-dir", str(out_dir)])
        assert result.exit_code == 1
        assert "INVALID" in result.stdout

    def test_unknown_run_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["validate", "ghost", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0


class TestCompareCommand:
    def test_compatible_runs_compare(self, tmp_path: Path) -> None:
        out_dir, a, b = _produce_runs(tmp_path)
        result = runner.invoke(app, ["compare", a, b, "--output-dir", str(out_dir)])
        assert result.exit_code == 0, _combined(result)
        assert "requests/sec" in result.stdout
        assert "concurrency" in result.stdout  # the axis under comparison

    def test_json_output(self, tmp_path: Path) -> None:
        out_dir, a, b = _produce_runs(tmp_path)
        result = runner.invoke(app, ["compare", a, b, "--output-dir", str(out_dir), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["run_a"] == a

    def test_incompatible_runs_refused(self, tmp_path: Path) -> None:
        out_dir, a, _ = _produce_runs(tmp_path)
        other = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(_write_config(tmp_path, "run-long", prompt_tokens=512)),
                "--output-dir",
                str(out_dir),
                "--no-report",
                "--no-charts",
            ],
        )
        assert other.exit_code == 0, _combined(other)
        c = next(p.name for p in out_dir.iterdir() if p.is_dir() and "run-long" in p.name)

        result = runner.invoke(app, ["compare", a, c, "--output-dir", str(out_dir)])
        assert result.exit_code != 0

    def test_force_overrides_gate(self, tmp_path: Path) -> None:
        out_dir, a, _ = _produce_runs(tmp_path)
        runner.invoke(
            app,
            [
                "run",
                "--config",
                str(_write_config(tmp_path, "run-long2", prompt_tokens=512)),
                "--output-dir",
                str(out_dir),
                "--no-report",
                "--no-charts",
            ],
        )
        c = next(p.name for p in out_dir.iterdir() if p.is_dir() and "run-long2" in p.name)

        result = runner.invoke(app, ["compare", a, c, "--output-dir", str(out_dir), "--force"])
        assert result.exit_code == 0, _combined(result)
        assert "overridden" in _combined(result).lower()


class TestExperimentCommand:
    def _experiment_file(self, tmp_path: Path, name: str = "exp.yaml") -> Path:
        path = tmp_path / name
        path.write_text(
            "name: A\n"
            "varies: load.concurrency\n"
            "base:\n"
            "  run_name: expA\n"
            "  backend:\n    kind: mock\n    model: mock-model\n"
            "  workload:\n    kind: fixed\n    num_requests: 3\n"
            "    prompt_tokens: 32\n    output_tokens: 3\n    seed: 0\n"
            "  load:\n    concurrency: 1\n    warmup_requests: 0\n    repetitions: 2\n"
            "variants:\n"
            "  - name: c1\n    overrides:\n      load: {concurrency: 1}\n"
            "  - name: c4\n    overrides:\n      load: {concurrency: 4}\n",
            encoding="utf-8",
        )
        return path

    def test_sweep_runs_all_variants(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "runs"
        result = runner.invoke(
            app,
            [
                "experiment",
                "--config",
                str(self._experiment_file(tmp_path)),
                "--output-dir",
                str(out_dir),
                "--no-charts",
            ],
        )
        assert result.exit_code == 0, _combined(result)
        assert "Experiment complete" in result.stdout
        assert "2 / 2" in result.stdout
        assert "synthetic" in result.stdout

        # One directory per variant, plus the study report.
        run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
        assert len(run_dirs) == 2
        assert (out_dir / "experiment-A.md").is_file()
        assert (out_dir / "experiment-A.json").is_file()

    def test_json_output(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "experiment",
                "--config",
                str(self._experiment_file(tmp_path)),
                "--output-dir",
                str(tmp_path / "runs"),
                "--no-report",
                "--json",
            ],
        )
        assert result.exit_code == 0, _combined(result)
        payload = json.loads(result.stdout)
        assert payload["experiment"] == "A"
        assert payload["complete"] is True
        assert payload["succeeded"] == 2

    def test_report_contains_trend_table(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "runs"
        runner.invoke(
            app,
            [
                "experiment",
                "--config",
                str(self._experiment_file(tmp_path)),
                "--output-dir",
                str(out_dir),
                "--no-charts",
            ],
        )
        report = (out_dir / "experiment-A.md").read_text()
        assert "Results by variant" in report
        assert "load.concurrency" in report
        assert "SYNTHETIC" in report

    def test_invalid_experiment_errors_cleanly(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "name: bad\n"
            "varies: load.concurrency\n"
            "variants:\n"
            "  - name: a\n    overrides:\n      load: {concurrency: 1}\n"
            "  - name: b\n    overrides:\n      load: {concurrency: 1}\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, ["experiment", "--config", str(bad)])
        # Two identical variants cannot vary anything; the guard refuses it.
        assert result.exit_code != 0
