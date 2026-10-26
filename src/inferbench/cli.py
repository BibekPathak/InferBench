"""InferBench command-line interface.

Phase 0 implements ``doctor`` and ``version`` in full. The remaining commands
(``models``, ``run``, ``compare``, ``report``, ``validate``) are registered as
explicit "not yet implemented" commands so that the CLI surface is discoverable
and honest about its state, rather than failing with an opaque error.
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from inferbench import __version__
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


def _not_implemented(command: str, phase: str) -> None:
    typer.echo(
        f"'{command}' is planned for {phase} and is not implemented yet.\n"
        f"See docs/roadmap.md for the phase plan.",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command()
def models() -> None:
    """List configured or supported model profiles."""
    _not_implemented("models", "Phase 2")


@app.command()
def run(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to a run config file.")],
) -> None:
    """Execute an experiment defined by --config."""
    _not_implemented("run", "Phase 1")


@app.command()
def compare(
    run_a: Annotated[str, typer.Argument(help="First run id.")],
    run_b: Annotated[str, typer.Argument(help="Second run id.")],
) -> None:
    """Compare two compatible runs."""
    _not_implemented("compare", "Phase 2")


@app.command()
def report(
    run_id: Annotated[str, typer.Argument(help="Run id to report on.")],
) -> None:
    """Generate a Markdown report for a run."""
    _not_implemented("report", "Phase 1")


@app.command()
def validate(
    run_id: Annotated[str, typer.Argument(help="Run id to validate.")],
) -> None:
    """Check result completeness and metadata integrity for a run."""
    _not_implemented("validate", "Phase 2")


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
