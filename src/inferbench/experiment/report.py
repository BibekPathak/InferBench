"""Experiment reporting: trends across variants, not just a table of runs.

A study report answers a different question from a single-run report. A single
run says "this is how fast it was". A study says "here is how the speed changed
as the variable changed, and here is what that suggests about the bottleneck".

The report therefore:

- orders variants along the varied axis (numerically when the values are
  numbers, so concurrency 1, 2, 4, 8 reads in that order rather than
  alphabetically as "1, 16, 2, 4");
- presents a trend table with the headline metrics per variant;
- identifies the **saturation knee** for concurrency sweeps — the point past
  which more offered load stops buying throughput — while stating that this is
  an observation, not proof of cause;
- carries the same provenance banner as single-run reports, so a synthetic
  study is never mistaken for a real one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inferbench.errors import StorageError
from inferbench.experiment.runner import StudyResult
from inferbench.storage import ExperimentStore

__all__ = ["render_study_markdown", "generate_study_report", "VariantSummary"]

_PERCENT_FIELDS = ("ttft", "tpot", "e2e")


@dataclass(frozen=True, slots=True)
class VariantSummary:
    """The headline numbers for one variant, ready for a trend table."""

    name: str
    run_id: str
    #: Value of the varied config field, used for ordering.
    varies_value: Any
    ttft_p50: float | None
    ttft_p95: float | None
    tpot_p50: float | None
    e2e_p50: float | None
    e2e_p95: float | None
    requests_per_s: float | None
    output_tokens_per_s: float | None
    failure_rate: float | None
    successes: int
    total: int
    per_repetition_rps: tuple[float, ...]


def _variant_name(run_name: str) -> str:
    """Extract the variant label from ``<experiment>--<variant>``."""
    return run_name.split("--", 1)[-1] if "--" in run_name else run_name


def _numeric(value: Any) -> float | None:
    """Coerce a varied value to a number for ordering, if it looks numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*$", value)
        if match:
            return float(match.group(1))
    return None


def _extract_varies_value(store: ExperimentStore, run_id: str, varies: str) -> Any:
    """Read the varied field's value out of a run's stored config."""
    if not varies:
        return None
    section, _, field = varies.partition(".")
    metadata = store.get_metadata(run_id)
    section_value = (metadata.config or {}).get(section) or {}
    if not isinstance(section_value, dict):
        return None
    return section_value.get(field)


def collect_variant_summaries(store: ExperimentStore, study: StudyResult) -> list[VariantSummary]:
    """Build one summary per completed variant, ordered along the varied axis."""
    summaries: list[VariantSummary] = []

    for run in study.runs:
        try:
            summary = store.get_summary(run.run_id)
        except StorageError:
            # A run without a summary cannot contribute to a trend table; it is
            # surfaced by `validate` instead of being silently averaged in.
            continue
        summary["run_id"] = run.run_id
        latency = summary.get("latency", {})
        throughput = summary.get("throughput", {})
        failures = summary.get("failures", {})
        summaries.append(
            VariantSummary(
                name=_variant_name(run.metadata.run_name),
                run_id=run.run_id,
                varies_value=_extract_varies_value(store, run.run_id, study.varies),
                ttft_p50=latency.get("ttft", {}).get("p50"),
                ttft_p95=latency.get("ttft", {}).get("p95"),
                tpot_p50=latency.get("tpot", {}).get("p50"),
                e2e_p50=latency.get("e2e", {}).get("p50"),
                e2e_p95=latency.get("e2e", {}).get("p95"),
                requests_per_s=throughput.get("requests_per_s"),
                output_tokens_per_s=throughput.get("output_tokens_per_s"),
                failure_rate=failures.get("failure_rate"),
                successes=failures.get("successes", 0) or 0,
                total=failures.get("total", 0) or 0,
                per_repetition_rps=tuple(summary.get("per_repetition_rps") or ()),
            )
        )

    # Order numerically along the varied axis when possible, else by label.
    numeric_keys = [_numeric(s.varies_value) for s in summaries]
    if summaries and all(key is not None for key in numeric_keys):
        summaries.sort(key=lambda s: _numeric(s.varies_value) or 0.0)
    else:
        summaries.sort(key=lambda s: str(s.varies_value))
    return summaries


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _find_saturation_knee(summaries: list[VariantSummary]) -> int | None:
    """Index past which throughput stops improving meaningfully.

    Returns the index of the last variant that improved throughput by at least
    10% over its predecessor, or ``None`` when the sweep shows no knee (fewer
    than two points, or throughput still climbing).

    This is a heuristic for *description*, not a claim about mechanism: it
    identifies where the curve flattens, and the report says so.
    """
    points = [
        (s.varies_value, s.requests_per_s)
        for s in summaries
        if s.requests_per_s is not None and _numeric(s.varies_value) is not None
    ]
    if len(points) < 2:
        return None

    last_improvement = 0
    for index in range(1, len(points)):
        previous = points[index - 1][1]
        current = points[index][1]
        if previous is None or current is None or previous <= 0:
            continue
        if (current - previous) / previous >= 0.10:
            last_improvement = index
        else:
            # First variant that fails to improve is the knee.
            return index - 1
    # Throughput kept improving across the whole sweep.
    return last_improvement


def render_study_markdown(
    study: StudyResult,
    summaries: list[VariantSummary],
    *,
    metadata: dict[str, Any] | None = None,
    charts: list[str] | None = None,
    chart_notes: list[str] | None = None,
) -> str:
    """Render a study report as Markdown."""
    metadata = metadata or {}
    source = metadata.get("data_source", "synthetic")
    is_synthetic = source != "measured"

    lines: list[str] = []
    lines.append(f"# InferBench experiment report — {study.experiment}")
    lines.append("")
    lines.append(
        f"> **Provenance: {source.upper()}.** "
        + (
            "These numbers come from a synthetic mock backend, not a real serving "
            "engine. They demonstrate the experimental method and the reporting "
            "path, and **must not** be read as GPU or model performance."
            if is_synthetic
            else "These numbers were measured against a real serving backend."
        )
    )
    lines.append("")

    # --- Design -----------------------------------------------------------
    lines.append("## Experiment design")
    lines.append("")
    lines.append(f"- **Experiment:** {study.experiment}")
    if study.varies:
        lines.append(f"- **Variable under study:** `{study.varies}`")
    lines.append(f"- **Variants attempted:** {study.attempted}")
    lines.append(f"- **Variants completed:** {study.succeeded}")
    if study.failures:
        lines.append(f"- **Variants failed:** {len(study.failures)}")
    if metadata.get("model"):
        lines.append(f"- **Model:** `{metadata['model']}`")
    if metadata.get("backend_kind"):
        lines.append(f"- **Backend:** {metadata['backend_kind']}")
    lines.append("")

    if study.failures:
        lines.append("### Failed variants")
        lines.append("")
        for name, reason in study.failures.items():
            lines.append(f"- `{name}`: {reason}")
        lines.append("")
        lines.append(
            "_Results below cover only the variants that completed; the study is incomplete._"
        )
        lines.append("")

    if not summaries:
        lines.append("No completed variants to report.")
        lines.append("")
        return "\n".join(lines)

    # --- Trend table ------------------------------------------------------
    axis_label = study.varies or "variant"
    lines.append("## Results by variant")
    lines.append("")
    lines.append(
        f"| {axis_label} | TTFT p50 (s) | TTFT p95 (s) | TPOT p50 (s) | "
        "E2E p50 (s) | req/s | tok/s | failure rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for summary in summaries:
        lines.append(
            f"| {_fmt(summary.varies_value)} "
            f"| {_fmt(summary.ttft_p50)} | {_fmt(summary.ttft_p95)} "
            f"| {_fmt(summary.tpot_p50)} | {_fmt(summary.e2e_p50)} "
            f"| {_fmt(summary.requests_per_s)} | {_fmt(summary.output_tokens_per_s)} "
            f"| {_fmt(summary.failure_rate)} |"
        )
    lines.append("")

    # --- Repetition variability ------------------------------------------
    if any(len(s.per_repetition_rps) > 1 for s in summaries):
        lines.append("### Run-to-run variability")
        lines.append("")
        lines.append("| variant | per-repetition req/s | spread |")
        lines.append("|---|---|---|")
        for summary in summaries:
            values = summary.per_repetition_rps
            if len(values) > 1:
                spread = max(values) - min(values)
                rendered = ", ".join(_fmt(v) for v in values)
                lines.append(f"| {_fmt(summary.varies_value)} | {rendered} | {_fmt(spread)} |")
            else:
                lines.append(f"| {_fmt(summary.varies_value)} | single repetition | n/a |")
        lines.append("")

    # --- Observations -----------------------------------------------------
    lines.append("## Observations")
    lines.append("")

    first, last = summaries[0], summaries[-1]
    if first.ttft_p50 is not None and last.ttft_p50 is not None and first.ttft_p50 > 0:
        change = (last.ttft_p50 - first.ttft_p50) / first.ttft_p50 * 100.0
        lines.append(
            f"- **TTFT p50** moved from {_fmt(first.ttft_p50)} s at "
            f"{axis_label}={_fmt(first.varies_value)} to {_fmt(last.ttft_p50)} s at "
            f"{axis_label}={_fmt(last.varies_value)} ({change:+.1f}%)."
        )
    if first.tpot_p50 is not None and last.tpot_p50 is not None and first.tpot_p50 > 0:
        change = (last.tpot_p50 - first.tpot_p50) / first.tpot_p50 * 100.0
        lines.append(
            f"- **TPOT p50** moved from {_fmt(first.tpot_p50)} s to "
            f"{_fmt(last.tpot_p50)} s ({change:+.1f}%)."
        )
    if (
        first.requests_per_s is not None
        and last.requests_per_s is not None
        and first.requests_per_s > 0
    ):
        change = (last.requests_per_s - first.requests_per_s) / first.requests_per_s * 100.0
        lines.append(
            f"- **Throughput** moved from {_fmt(first.requests_per_s)} req/s to "
            f"{_fmt(last.requests_per_s)} req/s ({change:+.1f}%)."
        )

    # Tail latency growth is the signature of approaching saturation.
    if first.e2e_p95 is not None and last.e2e_p95 is not None and first.e2e_p95 > 0:
        change = (last.e2e_p95 - first.e2e_p95) / first.e2e_p95 * 100.0
        tail_note = (
            " Tail growth out of proportion to throughput gain is the usual "
            "signature of saturation."
            if change > 5.0
            else " The tail did not grow appreciably, so this sweep shows no sign "
            "of saturation within the range measured."
        )
        lines.append(
            f"- **E2E p95** (tail) moved from {_fmt(first.e2e_p95)} s to "
            f"{_fmt(last.e2e_p95)} s ({change:+.1f}%).{tail_note}"
        )

    knee = _find_saturation_knee(summaries)
    if knee is not None and knee < len(summaries) - 1:
        knee_point = summaries[knee]
        lines.append(
            f"- **Apparent flattening** at {axis_label}={_fmt(knee_point.varies_value)}: "
            f"beyond this point, throughput improved by less than 10% per step. "
            "This describes the curve; it does not by itself establish the cause."
        )

    any_failures = [s for s in summaries if (s.successes or 0) < (s.total or 0)]
    if any_failures:
        lines.append(
            f"- **{len(any_failures)} variant(s) had failed requests.** Throughput "
            "counts only completed requests, so those variants did less work than "
            "the offered load suggests."
        )

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

    # --- Interpretation and limits ---------------------------------------
    lines.append("## Interpretation and limitations")
    lines.append("")
    lines.append(
        "- The observations above are **descriptive**. They state what the "
        "measurements show; they do not establish which component caused it."
    )
    lines.append(
        "- Only one variable is varied by construction, and the loader verifies "
        "this, so the comparison is controlled."
    )
    if is_synthetic:
        lines.append(
            "- **The backend was synthetic.** The mock's timing model responds to "
            "load in a programmed way, so these trends reflect the model, not real "
            "hardware."
        )
    lines.append(
        "- Client-observed timings include network and scheduling effects; they are "
        "not the server's internal decode time."
    )
    lines.append("")

    # --- Method -----------------------------------------------------------
    lines.append("## Method")
    lines.append("")
    lines.append(f"- Variable under study: `{study.varies or 'n/a'}`")
    lines.append("- Percentiles: nearest-rank on the sorted sample")
    lines.append("- Throughput: completed requests / measured wall-clock window")
    lines.append("- Warm-up: excluded from all measured statistics")
    lines.append("")
    lines.append("| variant | run id |")
    lines.append("|---|---|")
    for summary in summaries:
        lines.append(f"| {summary.name} | `{summary.run_id}` |")
    lines.append("")
    return "\n".join(lines)


def generate_study_report(
    store: ExperimentStore,
    study: StudyResult,
    *,
    output_path: str | Path | None = None,
    make_charts: bool = True,
) -> Path:
    """Generate and write a study report, returning the path written."""
    summaries = collect_variant_summaries(store, study)

    # Reuse the first run's metadata for shared facts (model, backend, env).
    metadata: dict[str, Any] = {}
    if study.runs:
        metadata = study.runs[0].metadata.to_dict()

    # Write alongside the runs: a study is identified by its experiment name
    # plus the first run id, so repeated studies do not overwrite each other.
    if output_path is not None:
        destination = Path(output_path)
    else:
        first_run_dir = Path(study.runs[0].run_dir) if study.runs else store.root
        destination = first_run_dir.parent / f"experiment-{study.experiment}.md"

    charts: list[str] = []
    chart_notes: list[str] = []
    if make_charts and summaries:
        from inferbench.experiment.charts import try_make_study_charts

        charts, chart_notes = try_make_study_charts(study, summaries, destination.parent)

    markdown = render_study_markdown(
        study, summaries, metadata=metadata, charts=charts, chart_notes=chart_notes
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")

    # Also persist a machine-readable form so a study can be diffed or replotted.
    json_path = destination.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "experiment": study.experiment,
                "varies": study.varies,
                "complete": study.is_complete,
                "failures": study.failures,
                "variants": [
                    {
                        "name": s.name,
                        "run_id": s.run_id,
                        "varies_value": s.varies_value,
                        "ttft_p50": s.ttft_p50,
                        "ttft_p95": s.ttft_p95,
                        "tpot_p50": s.tpot_p50,
                        "e2e_p50": s.e2e_p50,
                        "e2e_p95": s.e2e_p95,
                        "requests_per_s": s.requests_per_s,
                        "output_tokens_per_s": s.output_tokens_per_s,
                        "failure_rate": s.failure_rate,
                        "per_repetition_rps": list(s.per_repetition_rps),
                    }
                    for s in summaries
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return destination
