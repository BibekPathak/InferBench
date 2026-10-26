"""Reporting: turn a stored run into a human-readable Markdown report.

Design rules that keep reports honest:

- The data provenance banner (``measured`` / ``synthetic``) is always emitted
  and always near the top. A synthetic report says so in the first lines.
- Failures and warnings are shown next to the numbers they affect, never
  buried.
- Numbers are formatted with a fixed, documented precision so reports are
  diffable.
- Charts are optional: if matplotlib is absent, text tables are used and the
  report says charts were skipped, rather than failing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferbench.errors import RunNotFoundError
from inferbench.storage import ExperimentStore

__all__ = ["generate_report", "render_markdown"]

_CLAIM = "> **Provenance: {source}.** {explanation}\n"


def _fmt(value: Any, unit: str = "", digits: int = 4) -> str:
    """Format a number for display, or ``n/a`` when unavailable."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value}{unit}"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def render_markdown(
    metadata: dict[str, Any],
    summary: dict[str, Any],
    *,
    charts: list[str] | None = None,
    chart_notes: list[str] | None = None,
) -> str:
    """Render a complete report from stored metadata and summary dicts."""
    source = summary.get("data_source") or metadata.get("data_source", "unknown")
    is_synthetic = source != "measured"

    lines: list[str] = []
    lines.append(f"# InferBench report — {metadata.get('run_name', 'run')}")
    lines.append("")
    lines.append(
        _CLAIM.format(
            source=source.upper(),
            explanation=(
                "These numbers come from a synthetic mock backend, not a real "
                "serving engine. They validate the *pipeline* (workload, "
                "timing, metrics, storage) and **must not** be read as GPU or "
                "model performance."
                if is_synthetic
                else "These numbers were measured against a real serving backend."
            ),
        )
    )

    # --- Run identity -----------------------------------------------------
    lines.append("## Run")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| run id | `{metadata.get('run_id')}` |")
    lines.append(f"| experiment | {metadata.get('experiment')} |")
    lines.append(f"| backend | {metadata.get('backend_kind')} |")
    lines.append(f"| backend version | {metadata.get('backend_version') or 'unavailable'} |")
    lines.append(f"| model | `{metadata.get('model')}` |")
    lines.append(f"| model revision | {metadata.get('model_revision') or 'unavailable'} |")
    lines.append(f"| data source | **{source}** |")
    lines.append(f"| tool version | {metadata.get('tool_version')} |")
    lines.append(f"| incomplete | {_fmt(metadata.get('incomplete'))} |")
    if metadata.get("incomplete_reason"):
        lines.append(f"| incomplete reason | {metadata['incomplete_reason']} |")
    lines.append("")

    # --- Environment ------------------------------------------------------
    env = metadata.get("environment") or {}
    gpu = env.get("gpu") or {}
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Python: {env.get('python_version', 'unknown')}")
    lines.append(f"- CPU: {env.get('cpu_count', 'unknown')} cores")
    if gpu.get("available"):
        devices = gpu.get("devices") or []
        names = ", ".join(str(d.get("name")) for d in devices) or "unknown"
        lines.append(
            f"- GPU: {names} (driver {gpu.get('driver_version')}, CUDA {gpu.get('cuda_version')})"
        )
    else:
        lines.append(f"- GPU: **unavailable** — {gpu.get('reason', 'not reported')}")
    lines.append("")

    # --- Workload & load --------------------------------------------------
    config = metadata.get("config") or {}
    workload = config.get("workload") or {}
    load = config.get("load") or {}
    lines.append("## Workload and load")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| workload kind | {workload.get('kind')} |")
    lines.append(f"| requests per repetition | {workload.get('num_requests')} |")
    lines.append(f"| prompt tokens (target) | {workload.get('prompt_tokens')} |")
    lines.append(f"| output tokens (target) | {workload.get('output_tokens')} |")
    lines.append(f"| seed | {workload.get('seed')} |")
    lines.append(f"| arrival pattern | {load.get('pattern')} |")
    lines.append(f"| concurrency | {load.get('concurrency')} |")
    lines.append(f"| warm-up requests | {load.get('warmup_requests')} |")
    lines.append(f"| repetitions | {load.get('repetitions')} |")
    lines.append("")

    # --- Throughput -------------------------------------------------------
    tp = summary.get("throughput") or {}
    lines.append("## Throughput")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| successful requests | {_fmt(tp.get('successful_requests'))} |")
    lines.append(f"| failed requests | {_fmt(tp.get('failed_requests'))} |")
    lines.append(f"| output tokens | {_fmt(tp.get('output_tokens'))} |")
    lines.append(f"| measured wall time | {_fmt(tp.get('wall_time_s'), ' s')} |")
    lines.append(f"| **requests/sec (completed)** | {_fmt(tp.get('requests_per_s'))} |")
    lines.append(f"| offered requests/sec | {_fmt(tp.get('offered_requests_per_s'))} |")
    lines.append(f"| **output tokens/sec** | {_fmt(tp.get('output_tokens_per_s'))} |")
    per_rep = summary.get("per_repetition_rps") or []
    if per_rep:
        lines.append(f"| per-repetition req/s | {', '.join(_fmt(v) for v in per_rep)} |")
    lines.append("")

    # --- Latency ----------------------------------------------------------
    lat = summary.get("latency") or {}
    lines.append("## Latency")
    lines.append("")
    lines.append("| Metric | count | p50 | p90 | p95 | p99 | mean |")
    lines.append("|--------|-------|-----|-----|-----|-----|------|")
    for label, key in (
        ("TTFT (s)", "ttft"),
        ("TPOT (s)", "tpot"),
        ("inter-token (s)", "inter_token"),
        ("end-to-end (s)", "e2e"),
    ):
        dist = lat.get(key) or {}
        no_samples = dist.get("count", 0) == 0
        lines.append(
            f"| {label} | {_fmt(dist.get('count'))} | {_fmt(dist.get('p50'))} | "
            f"{_fmt(dist.get('p90'))} | {_fmt(dist.get('p95'))} | {_fmt(dist.get('p99'))} | "
            f"{_fmt(dist.get('mean'))} |"
        )
        if no_samples and key in ("ttft", "tpot"):
            lines.append(
                f"| _{label}: no samples — insufficient successful requests_ | | | | | | |"
            )
    lines.append("")
    lines.append(
        f"TTFT measured on {_fmt(lat.get('requests_with_ttft'))} request(s); "
        f"TPOT on {_fmt(lat.get('requests_with_tpot'))} request(s) "
        "(needs ≥2 output tokens and a successful status)."
    )
    lines.append("")

    # --- Failures ---------------------------------------------------------
    fail = summary.get("failures") or {}
    lines.append("## Failures")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| success | {_fmt(fail.get('successes'))} |")
    lines.append(f"| timeout | {_fmt(fail.get('timeouts'))} |")
    lines.append(f"| cancelled | {_fmt(fail.get('cancelled'))} |")
    lines.append(f"| error | {_fmt(fail.get('errors'))} |")
    lines.append(f"| interrupted | {_fmt(fail.get('interrupted'))} |")
    lines.append(f"| **failure rate** | {_pct(fail.get('failure_rate'))} |")
    lines.append("")

    # --- Warnings ---------------------------------------------------------
    warnings = summary.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    # --- Charts -----------------------------------------------------------
    lines.append("## Charts")
    lines.append("")
    if charts:
        for chart in charts:
            lines.append(f"![{Path(chart).stem}]({Path(chart).name})")
            lines.append("")
    else:
        lines.append("_No charts generated._")
        lines.append("")
    for note in chart_notes or []:
        lines.append(f"- {note}")
    if chart_notes:
        lines.append("")

    # --- Aggregation method ----------------------------------------------
    agg = summary.get("aggregation") or {}
    lines.append("## Method")
    lines.append("")
    lines.append(f"- Percentiles: {agg.get('percentile_method', 'nearest-rank')}")
    lines.append(f"- TTFT: {agg.get('ttft', 'dispatch → first token')}")
    lines.append(f"- TPOT: {agg.get('tpot', '(e2e − ttft) / (tokens − 1)')}")
    lines.append(f"- Inter-token: {agg.get('inter_token', 'per-token arrival gaps')}")
    lines.append(f"- Throughput: {agg.get('throughput', 'successes / measured window')}")
    lines.append(f"- Warm-up excluded: {_fmt(agg.get('warmup_excluded'))}")
    lines.append("")

    # --- Limitations ------------------------------------------------------
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Client-observed timings include network and client scheduling effects; "
        "they are not the server's internal decode time."
    )
    lines.append(
        "- Prompt token counts are only reported when the backend provides them; "
        "otherwise fields are `n/a` rather than estimated."
    )
    lines.append(
        "- GPU utilization and memory telemetry are unavailable unless collected separately."
    )
    if is_synthetic:
        lines.append(
            "- **This run used a synthetic backend. No conclusion about real "
            "hardware, models, or serving engines may be drawn from it.**"
        )
    lines.append("")
    return "\n".join(lines)


def generate_report(
    store: ExperimentStore,
    run_id: str,
    *,
    output_path: str | Path | None = None,
    make_charts: bool = True,
) -> Path:
    """Generate (and write) a Markdown report for ``run_id``.

    Args:
        store: Open experiment store.
        run_id: Run to report on.
        output_path: Where to write the report. Defaults to
            ``<run_dir>/report.md``.
        make_charts: Attempt chart rendering when matplotlib is installed.

    Raises:
        RunNotFoundError: The run or its summary is missing.
    """
    if not store.has_run(run_id):
        raise RunNotFoundError(f"No such run: {run_id!r}")

    metadata = store.get_metadata(run_id).to_dict()
    summary = store.get_summary(run_id)

    run_dir = store.run_dir(run_id)
    charts: list[str] = []
    chart_notes: list[str] = []
    if make_charts:
        from inferbench.analysis.charts import try_make_charts

        charts, chart_notes = try_make_charts(store, run_id, run_dir)

    markdown = render_markdown(metadata, summary, charts=charts, chart_notes=chart_notes)

    destination = Path(output_path) if output_path else run_dir / "report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")
    return destination
