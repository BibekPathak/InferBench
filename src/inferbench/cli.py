"""InferBench command-line interface.

Phase 0 implements ``doctor`` and ``version`` in full. The remaining commands
(``models``, ``run``, ``compare``, ``report``, ``validate``) are registered as
explicit "not yet implemented" commands so that the CLI surface is discoverable
and honest about its state, rather than failing with an opaque error.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated, Any

import typer

from inferbench import __version__
from inferbench.config import load_config
from inferbench.env import collect_environment, environment_to_json
from inferbench.errors import InferBenchError
from inferbench.logging_utils import configure_logging, get_logger, log_event

__all__ = ["app", "main"]

app = typer.Typer(
    name="inferbench",
    help="Real-world LLM inference benchmarking and optimization platform.",
    no_args_is_help=True,
    add_completion=False,
)

_log = get_logger("cli")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"inferbench {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the InferBench version and exit.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Configure global CLI behavior."""
    configure_logging(level="DEBUG" if verbose else "INFO", fmt="text")


@app.command()
def version() -> None:
    """Print the InferBench version."""
    typer.echo(f"inferbench {__version__}")


@app.command()
def doctor(
    disk_path: Annotated[
        str,
        typer.Option(
            "--disk-path",
            help="Filesystem to measure for free space (used to gate model downloads).",
        ),
    ] = ".",
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the full environment report as JSON."),
    ] = False,
) -> None:
    """Inspect the environment: Python, CPU, memory, GPU, CUDA, disk, packages.

    Never fails on a machine without a GPU: accelerator fields are reported as
    unavailable with an explanation.
    """
    report = collect_environment(disk_path=disk_path)

    if as_json:
        typer.echo(environment_to_json(report))
        return

    typer.echo("InferBench environment")
    typer.echo("=" * 60)
    typer.echo(f"  version          {__version__}")
    typer.echo(f"  python           {report.python_version} ({report.python_implementation})")
    typer.echo(f"  platform         {report.platform}")
    typer.echo(f"  machine          {report.machine}")
    typer.echo(f"  cpus             {report.cpu_count}")

    mem = report.total_memory_gb
    typer.echo(f"  memory           {mem if mem is not None else 'unknown'} GiB")

    typer.echo("")
    typer.echo("Accelerator")
    typer.echo("-" * 60)
    if report.gpu.available:
        typer.echo(f"  status           available ({report.gpu.device_count} device(s))")
        typer.echo(f"  driver           {report.gpu.driver_version}")
        typer.echo(f"  cuda             {report.gpu.cuda_version}")
        for index, device in enumerate(report.gpu.devices):
            typer.echo(
                f"  gpu[{index}]          {device.get('name')} "
                f"({device.get('memory_total_mb')} MiB, cc {device.get('compute_capability')})"
            )
    else:
        typer.echo("  status           UNAVAILABLE")
        typer.echo(f"  reason           {report.gpu.reason}")
        typer.echo("  note             GPU-specific experiments will be skipped or marked pending.")

    typer.echo("")
    typer.echo("Disk")
    typer.echo("-" * 60)
    if report.disk is not None:
        typer.echo(
            f"  {report.disk.path}: {report.disk.free_gb} GiB free of "
            f"{report.disk.total_gb} GiB ({report.disk.used_percent}% used)"
        )
    else:
        typer.echo("  unknown (could not stat path)")

    typer.echo("")
    typer.echo("Packages")
    typer.echo("-" * 60)
    for package in report.packages:
        marker = package.version if package.available else "not installed"
        typer.echo(f"  {package.name:<14} {marker}")

    typer.echo("")
    typer.echo("Next steps")
    typer.echo("-" * 60)
    if report.gpu.available:
        typer.echo("  GPU detected: real backend experiments are possible.")
        typer.echo("  Example: inferbench run --config configs/baseline.yaml")
    else:
        typer.echo("  No GPU: run the CPU-only smoke path with the mock backend.")
        typer.echo("  Example: inferbench run --config configs/baseline.yaml")
        typer.echo("  Real vLLM/GPU measurements require a GPU host.")


@app.command()
def models(
    profiles: Annotated[
        str | None,
        typer.Option("--profiles", help="Optional YAML file defining or overriding profiles."),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit profiles as JSON."),
    ] = False,
) -> None:
    """List configured or supported model profiles."""
    from inferbench.models import list_profiles

    entries = list_profiles(profiles)

    if as_json:
        typer.echo(json.dumps([p.to_dict() for p in entries], indent=2))
        return

    typer.echo("Model profiles")
    typer.echo("=" * 72)
    for profile in entries:
        gpu = "GPU" if profile.requires_gpu else "CPU-ok"
        typer.echo(f"  {profile.name:<22} [{gpu}] {profile.model}")
        if profile.description:
            typer.echo(f"  {'':<22} {profile.description}")
        details = []
        if profile.parameters_billions is not None:
            details.append(f"{profile.parameters_billions:g}B")
        if profile.dtype:
            details.append(profile.dtype)
        if profile.quantization:
            details.append(profile.quantization)
        if details:
            typer.echo(f"  {'':<22} ({', '.join(details)})")
        typer.echo("")

    typer.echo("Profiles describe what to run; whether a model is actually served is")
    typer.echo("validated by the backend at run time.")


@app.command()
def run(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to a run config file.")],
    output_dir: Annotated[
        str | None,
        typer.Option("--output-dir", "-o", help="Override the config's output directory."),
    ] = None,
    report: Annotated[
        bool,
        typer.Option("--report/--no-report", help="Generate a Markdown report after the run."),
    ] = True,
    charts: Annotated[
        bool,
        typer.Option("--charts/--no-charts", help="Render charts (requires matplotlib)."),
    ] = True,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the run summary as JSON."),
    ] = False,
) -> None:
    """Execute an experiment defined by --config."""
    from pathlib import Path

    from inferbench.backends import create_backend
    from inferbench.env import collect_environment
    from inferbench.runner import Runner
    from inferbench.storage import ExperimentStore

    run_config = load_config(config)
    if output_dir:
        run_config.output_dir = Path(output_dir)

    store = ExperimentStore(run_config.output_dir)
    try:
        environment = collect_environment(disk_path=run_config.output_dir).to_dict()
        backend = create_backend(run_config.backend)
        runner = Runner(run_config, backend, store, environment=environment)
        result = asyncio.run(_run_and_teardown(runner, backend))
    finally:
        store.close()

    if as_json:
        typer.echo(json.dumps(result.summary, indent=2, default=str))
    else:
        _print_run_summary(result)

    if report:
        from inferbench.analysis import generate_report

        store = ExperimentStore(run_config.output_dir)
        try:
            path = generate_report(store, result.run_id, make_charts=charts)
        finally:
            store.close()
        typer.echo(f"Report: {path}")


async def _run_and_teardown(runner: Any, backend: Any) -> Any:
    """Run the experiment, always tearing the backend down afterwards."""
    try:
        return await runner.run()
    finally:
        await backend.teardown()


def _print_run_summary(result: Any) -> None:
    summary = result.summary
    tp = summary.get("throughput", {})
    lat = summary.get("latency", {})
    fail = summary.get("failures", {})

    typer.echo("")
    typer.echo(f"Run complete: {result.run_id}")
    typer.echo("-" * 60)
    typer.echo(f"  data source        {summary.get('data_source')}")
    typer.echo(f"  completed requests {tp.get('successful_requests')}")
    typer.echo(f"  failed requests    {tp.get('failed_requests')}")
    typer.echo(f"  wall time          {tp.get('wall_time_s')} s")
    typer.echo(f"  requests/sec       {tp.get('requests_per_s')}")
    typer.echo(f"  output tokens/sec  {tp.get('output_tokens_per_s')}")

    ttft = lat.get("ttft", {})
    e2e = lat.get("e2e", {})
    tpot = lat.get("tpot", {})
    typer.echo(
        f"  TTFT p50/p95       {ttft.get('p50')} / {ttft.get('p95')} s  (n={ttft.get('count')})"
    )
    typer.echo(
        f"  TPOT p50/p95       {tpot.get('p50')} / {tpot.get('p95')} s  (n={tpot.get('count')})"
    )
    typer.echo(
        f"  E2E  p50/p95       {e2e.get('p50')} / {e2e.get('p95')} s  (n={e2e.get('count')})"
    )
    typer.echo(f"  failure rate       {fail.get('failure_rate')}")

    if summary.get("data_source") != "measured":
        typer.echo("")
        typer.echo("  NOTE: synthetic (mock) backend — not real GPU/model performance.")

    for warning in summary.get("warnings", []):
        typer.echo(f"  WARNING: {warning}")
    typer.echo(f"  artifacts          {result.run_dir}")


@app.command()
def compare(
    run_a: Annotated[str, typer.Argument(help="First (baseline) run id.")],
    run_b: Annotated[str, typer.Argument(help="Second run id.")],
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Directory holding the experiment store."),
    ] = "runs",
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the comparison as JSON."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Compare even when the fairness gate blocks it (results may be misleading).",
        ),
    ] = False,
) -> None:
    """Compare two compatible runs."""
    from inferbench.analysis import compare_runs, format_comparison
    from inferbench.storage import ExperimentStore

    store = ExperimentStore(output_dir)
    try:
        try:
            result = compare_runs(store, run_a, run_b)
        except InferBenchError as exc:
            if not force:
                raise
            # Report the refusal, then proceed explicitly at the user's request.
            typer.echo(f"Fairness gate blocked this comparison: {exc}", err=True)
            typer.echo("Proceeding due to --force. Interpret with caution.", err=True)
            result = _forced_comparison(store, run_a, run_b)
    finally:
        store.close()

    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        typer.echo(format_comparison(result))


def _forced_comparison(store: Any, run_a: str, run_b: str) -> Any:
    """Build a comparison while recording that the fairness gate was overridden."""
    from inferbench.analysis import ComparisonResult, check_compatibility
    from inferbench.analysis.compare import MetricDelta

    compatibility = check_compatibility(store.get_metadata(run_a), store.get_metadata(run_b))
    summary_a = store.get_summary(run_a)
    summary_b = store.get_summary(run_b)

    def delta(name: str, unit: str, a: Any, b: Any, lower: bool = True) -> Any:
        absolute = None
        percent = None
        if a is not None and b is not None:
            absolute = b - a
            if a != 0:
                percent = (b - a) / a * 100.0
        return MetricDelta(name, unit, a, b, absolute, percent, lower)

    lat_a, lat_b = summary_a.get("latency", {}), summary_b.get("latency", {})
    tp_a, tp_b = summary_a.get("throughput", {}), summary_b.get("throughput", {})
    fl_a, fl_b = summary_a.get("failures", {}), summary_b.get("failures", {})

    deltas = (
        delta("TTFT p50", "s", lat_a.get("ttft", {}).get("p50"), lat_b.get("ttft", {}).get("p50")),
        delta("TTFT p95", "s", lat_a.get("ttft", {}).get("p95"), lat_b.get("ttft", {}).get("p95")),
        delta("TPOT p50", "s", lat_a.get("tpot", {}).get("p50"), lat_b.get("tpot", {}).get("p50")),
        delta("E2E p50", "s", lat_a.get("e2e", {}).get("p50"), lat_b.get("e2e", {}).get("p50")),
        delta("E2E p95", "s", lat_a.get("e2e", {}).get("p95"), lat_b.get("e2e", {}).get("p95")),
        delta("E2E p99", "s", lat_a.get("e2e", {}).get("p99"), lat_b.get("e2e", {}).get("p99")),
        delta(
            "requests/sec", "req/s", tp_a.get("requests_per_s"), tp_b.get("requests_per_s"), False
        ),
        delta(
            "output tokens/sec",
            "tok/s",
            tp_a.get("output_tokens_per_s"),
            tp_b.get("output_tokens_per_s"),
            False,
        ),
        delta("failure rate", "", fl_a.get("failure_rate"), fl_b.get("failure_rate")),
    )

    return ComparisonResult(
        run_a=run_a,
        run_b=run_b,
        compatibility=compatibility,
        deltas=deltas,
        notes=[
            "fairness gate was overridden with --force; the runs are not "
            "methodologically comparable and the deltas may be misleading",
            *compatibility.blockers,
        ],
    )


@app.command()
def report(
    run_id: Annotated[str, typer.Argument(help="Run id to report on.")],
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Directory holding the experiment store."),
    ] = "runs",
    output: Annotated[
        str | None,
        typer.Option(
            "--output", help="Write the report to this path instead of the run directory."
        ),
    ] = None,
    charts: Annotated[
        bool,
        typer.Option("--charts/--no-charts", help="Render charts (requires matplotlib)."),
    ] = True,
) -> None:
    """Generate a Markdown report for a run."""
    from inferbench.analysis import generate_report
    from inferbench.storage import ExperimentStore

    store = ExperimentStore(output_dir)
    try:
        path = generate_report(store, run_id, output_path=output, make_charts=charts)
    finally:
        store.close()
    typer.echo(f"Report written: {path}")


@app.command()
def experiment(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to an experiment file.")],
    output_dir: Annotated[
        str | None,
        typer.Option("--output-dir", "-o", help="Override the output directory."),
    ] = None,
    report: Annotated[
        bool,
        typer.Option("--report/--no-report", help="Generate a study report after the sweep."),
    ] = True,
    charts: Annotated[
        bool,
        typer.Option("--charts/--no-charts", help="Render charts (requires matplotlib)."),
    ] = True,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the study result as JSON."),
    ] = False,
) -> None:
    """Execute a multi-run experiment defined by --config."""
    from pathlib import Path

    from inferbench.backends import create_backend
    from inferbench.env import collect_environment
    from inferbench.experiment import generate_study_report, load_experiment
    from inferbench.experiment.runner import StudyRunner
    from inferbench.storage import ExperimentStore

    experiment_config = load_experiment(config)
    if output_dir:
        experiment_config.base.output_dir = Path(output_dir)

    store = ExperimentStore(experiment_config.base.output_dir)
    try:
        environment = collect_environment(disk_path=experiment_config.base.output_dir).to_dict()
        # All variants share the base backend; StudyRunner verifies this.
        backend = create_backend(experiment_config.base.backend)
        study = asyncio.run(
            StudyRunner(experiment_config, backend, store, environment=environment).run()
        )
    finally:
        store.close()

    if as_json:
        typer.echo(json.dumps(study.to_dict(), indent=2, default=str))
    else:
        _print_study_summary(study)

    if report:
        store = ExperimentStore(experiment_config.base.output_dir)
        try:
            path = generate_study_report(store, study, make_charts=charts)
        finally:
            store.close()
        typer.echo(f"Study report: {path}")

    # A partial study exits non-zero so scripts notice it.
    if not study.is_complete:
        raise typer.Exit(code=1)


def _print_study_summary(study: Any) -> None:
    typer.echo("")
    typer.echo(f"Experiment complete: {study.experiment}")
    typer.echo("-" * 60)
    typer.echo(f"  varies     {study.varies or 'n/a'}")
    typer.echo(f"  completed  {study.succeeded} / {study.attempted}")
    for run in study.runs:
        summary = run.summary
        throughput = summary.get("throughput", {})
        latency = summary.get("latency", {})
        variant = run.metadata.run_name.split("--", 1)[-1]
        typer.echo(
            f"    {variant:<16} {throughput.get('requests_per_s')} req/s  "
            f"TTFT p50={latency.get('ttft', {}).get('p50')}  "
            f"E2E p95={latency.get('e2e', {}).get('p95')}"
        )
    for name, reason in study.failures.items():
        typer.echo(f"    {name:<16} FAILED: {reason}")
    if study.runs and study.runs[0].metadata.data_source != "measured":
        typer.echo("")
        typer.echo("  NOTE: synthetic (mock) backend — not real GPU/model performance.")


@app.command()
def validate(
    run_id: Annotated[str, typer.Argument(help="Run id to validate.")],
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Directory holding the experiment store."),
    ] = "runs",
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit findings as JSON."),
    ] = False,
) -> None:
    """Check result completeness and metadata integrity for a run."""
    from inferbench.analysis import format_report, validate_run
    from inferbench.storage import ExperimentStore

    store = ExperimentStore(output_dir)
    try:
        report = validate_run(store, run_id)
    finally:
        store.close()

    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        typer.echo(format_report(report))

    # A non-zero exit makes validation usable in scripts and CI.
    if not report.is_valid:
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point wrapper that turns expected errors into clean exits."""
    try:
        app()
    except InferBenchError as exc:
        log_event(
            _log,
            logging.ERROR,
            "command failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
