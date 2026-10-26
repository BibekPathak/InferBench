"""Run comparison with fairness gating.

Comparing two benchmark runs is only meaningful if they answered the *same
question*. This module encodes that as an explicit gate: before producing any
comparison, it checks that the two runs used compatible workloads, load
profiles, and protocols. When they are incompatible it refuses, with a message
naming the specific incompatibility — because a silently unfair comparison is
worse than no comparison.

Three classes of difference are distinguished:

- **Blocking** — the runs are not comparable at all (e.g. different prompt/output
  token distributions, different measurement protocol). Comparison is refused.
- **Warned** — the runs are comparable but differ in a way the reader must know
  about (e.g. different models, different backend, different repetition count).
- **Expected** — the axis of the experiment itself (e.g. different concurrency),
  which is the *point* of the comparison and is not flagged.

Uncertainty is reported rather than hidden: when a run has multiple
repetitions, the spread across repetitions is used to derive a bootstrap
confidence interval on throughput, and the comparison says whether the observed
difference exceeds that spread.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from inferbench.analysis.validate import validate_run
from inferbench.errors import IncomparableRunsError
from inferbench.storage import ExperimentStore
from inferbench.types import RunMetadata

__all__ = [
    "CompatibilityReport",
    "MetricDelta",
    "ComparisonResult",
    "check_compatibility",
    "compare_runs",
    "bootstrap_ci",
]

#: Relative tolerance when deciding whether two token-length distributions are
#: "the same". Distributions from the same config differ only by sampling noise
#: in randomized workloads, so a small band is appropriate.
_LENGTH_TOLERANCE = 0.05

#: Default bootstrap settings.
_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 12345


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """The verdict on whether two runs may be compared."""

    comparable: bool
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparable": self.comparable,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


def _distribution(metadata: RunMetadata, key: str) -> dict[str, Any]:
    """Pull a token-length distribution out of stored run metadata.

    For a fixed workload the min and max are the single configured value; for
    randomized/mixed workloads they are the configured bounds. This describes
    what the run *intended* to send, which is the right basis for deciding
    whether two runs asked the same question.
    """
    workload = (metadata.config or {}).get("workload", {})
    if workload.get("kind") == "fixed":
        fixed_key = "prompt_tokens" if key == "prompt" else "output_tokens"
        value = workload.get(fixed_key)
        if value is not None:
            return {"min": value, "max": value, "mean": value}

    min_key = f"{key}_tokens_min"
    max_key = f"{key}_tokens_max"
    fallback_key = f"{key}_tokens"
    return {
        "min": workload.get(min_key) or workload.get(fallback_key),
        "max": workload.get(max_key) or workload.get(fallback_key),
    }


def _within_tolerance(a: float | None, b: float | None, tolerance: float) -> bool:
    """Whether two magnitudes agree within a relative tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a == 0 and b == 0:
        return True
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale <= tolerance


def check_compatibility(a: RunMetadata, b: RunMetadata) -> CompatibilityReport:
    """Decide whether two runs may be compared, and how much to trust it."""
    blockers: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    config_a = a.config or {}
    config_b = b.config or {}
    workload_a = config_a.get("workload", {})
    workload_b = config_b.get("workload", {})
    load_a = config_a.get("load", {})
    load_b = config_b.get("load", {})

    # --- Blocking: different prompt/output distributions ------------------
    for label in ("prompt", "output"):
        dist_a = _distribution(a, label)
        dist_b = _distribution(b, label)
        min_differs = not _within_tolerance(dist_a.get("min"), dist_b.get("min"), _LENGTH_TOLERANCE)
        max_differs = not _within_tolerance(dist_a.get("max"), dist_b.get("max"), _LENGTH_TOLERANCE)
        if min_differs or max_differs:
            blockers.append(
                f"different {label} token distribution: "
                f"{dist_a.get('min')}–{dist_a.get('max')} vs "
                f"{dist_b.get('min')}–{dist_b.get('max')}"
            )

    # --- Blocking: different workload shape or request count --------------
    if workload_a.get("kind") != workload_b.get("kind"):
        blockers.append(
            f"different workload kind: {workload_a.get('kind')!r} vs {workload_b.get('kind')!r}"
        )

    requests_a = workload_a.get("num_requests")
    requests_b = workload_b.get("num_requests")
    if requests_a != requests_b:
        blockers.append(
            f"different request counts: {requests_a} vs {requests_b}; "
            "throughput and tail latencies are not comparable across different sample sizes"
        )

    # --- Blocking: different measurement protocol -------------------------
    warmup_a = load_a.get("warmup_requests")
    warmup_b = load_b.get("warmup_requests")
    if warmup_a != warmup_b:
        warnings.append(
            f"different warm-up counts: {warmup_a} vs {warmup_b}; "
            "cold-start effects may differ between runs"
        )

    seed_a = workload_a.get("seed")
    seed_b = workload_b.get("seed")
    if seed_a != seed_b:
        notes.append(
            f"different seeds ({seed_a} vs {seed_b}); prompts differ, "
            "which is expected for randomized workloads"
        )

    # --- Warnings: environment and configuration differences --------------
    if a.model != b.model:
        warnings.append(f"different models: {a.model!r} vs {b.model!r}")
    if a.backend_kind != b.backend_kind:
        warnings.append(f"different backends: {a.backend_kind!r} vs {b.backend_kind!r}")
    if a.backend_version != b.backend_version:
        warnings.append(
            f"different backend versions: {a.backend_version!r} vs {b.backend_version!r}"
        )

    gpu_a = (a.environment or {}).get("gpu") or {}
    gpu_b = (b.environment or {}).get("gpu") or {}
    if gpu_a.get("available") != gpu_b.get("available"):
        warnings.append("one run used a GPU and the other did not")

    if (a.data_source == "measured") != (b.data_source == "measured"):
        other = b.data_source if a.data_source == "measured" else a.data_source
        warnings.append(
            f"one run is measured and the other is {other}; "
            "the comparison mixes real and synthetic data"
        )

    # --- Notes: the experimental axis, not a defect -----------------------
    concurrency_a = load_a.get("concurrency")
    concurrency_b = load_b.get("concurrency")
    if concurrency_a != concurrency_b:
        notes.append(
            f"different concurrency ({concurrency_a} vs {concurrency_b}): "
            "this is the axis under comparison"
        )

    pattern_a = load_a.get("pattern")
    pattern_b = load_b.get("pattern")
    if pattern_a != pattern_b:
        warnings.append(f"different arrival patterns: {pattern_a!r} vs {pattern_b!r}")

    repetitions_a = load_a.get("repetitions")
    repetitions_b = load_b.get("repetitions")
    if repetitions_a != repetitions_b:
        warnings.append(
            f"different repetition counts ({repetitions_a} vs {repetitions_b}); "
            "uncertainty estimates are not equally well supported"
        )
    if (repetitions_a or 1) < 2 or (repetitions_b or 1) < 2:
        notes.append(
            "at least one run has a single repetition, so run-to-run variability "
            "cannot be estimated from it"
        )

    return CompatibilityReport(
        comparable=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """A single metric compared across two runs."""

    name: str
    unit: str
    #: ``None`` when the metric is undefined for a run (e.g. no successful
    #: requests), which is reported as unavailable rather than as zero.
    value_a: float | None
    value_b: float | None
    #: ``b - a``.
    absolute_delta: float | None
    #: Percent change from a to b; ``None`` when a is zero or missing.
    percent_delta: float | None
    #: Whether this direction is an improvement. ``None`` when no judgement is
    #: meaningful (e.g. failure rate is not "better" in a directionless sense).
    lower_is_better: bool = True

    @property
    def improved(self) -> bool | None:
        if self.absolute_delta is None:
            return None
        if abs(self.absolute_delta) < 1e-12:
            return False
        return (self.absolute_delta < 0) if self.lower_is_better else (self.absolute_delta > 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "a": self.value_a,
            "b": self.value_b,
            "delta": self.absolute_delta,
            "percent_delta": self.percent_delta,
            "lower_is_better": self.lower_is_better,
            "improved": self.improved,
        }


def _delta(
    name: str,
    unit: str,
    a: float | None,
    b: float | None,
    *,
    lower_is_better: bool = True,
) -> MetricDelta:
    absolute = None
    percent = None
    if a is not None and b is not None:
        absolute = b - a
        if a != 0:
            percent = (b - a) / a * 100.0
    return MetricDelta(
        name=name,
        unit=unit,
        value_a=a,
        value_b=b,
        absolute_delta=absolute,
        percent_delta=percent,
        lower_is_better=lower_is_better,
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] | None = None,
    confidence: float = 0.95,
    resamples: int = _BOOTSTRAP_RESAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval for a statistic.

    Deterministic for a given seed, which keeps reports reproducible. Returns
    ``None`` when there are too few samples for an interval to mean anything —
    reporting nothing is better than reporting a fabricated precision.

    Args:
        values: Sample values.
        statistic: Function computing the statistic of interest from a sample;
            defaults to the mean.
        confidence: Confidence level in ``(0, 1)``.
        resamples: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Raises:
        ValueError: ``confidence`` is outside ``(0, 1)`` or there are too few
            resamples.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if resamples < 100:
        raise ValueError(f"resamples must be >= 100, got {resamples}")
    if len(values) < 2:
        return None

    stat = statistic or (lambda sample: sum(sample) / len(sample))
    rng = random.Random(seed)
    n = len(values)
    estimates: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        try:
            estimates.append(float(stat(sample)))
        except (ValueError, ZeroDivisionError):
            continue
    if not estimates:
        return None

    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, int(math.floor(alpha * len(estimates))))
    upper_index = min(len(estimates) - 1, int(math.ceil((1.0 - alpha) * len(estimates))) - 1)
    return estimates[lower_index], estimates[upper_index]


def _throughput_ci(
    records: Sequence[Any], repetitions: int, wall_time_s: float
) -> tuple[float, float] | None:
    """CI on requests/sec, resampling at the *repetition* level when possible.

    Resampling repetitions (not individual requests) is the honest unit of
    uncertainty for a benchmark: repetitions are the independent trials. With a
    single repetition there is no estimate to make.
    """
    if repetitions < 2:
        return None

    by_rep: dict[int, list[float]] = {}
    for record in records:
        if record.is_warmup:
            continue
        by_rep.setdefault(record.repetition, []).append(1.0 if record.is_success else 0.0)

    per_rep_rps = [sum(successes) / wall_time_s * repetitions for successes in by_rep.values()]
    if len(per_rep_rps) < 2:
        return None
    return bootstrap_ci(per_rep_rps)


@dataclass(slots=True)
class ComparisonResult:
    """The full outcome of comparing two runs."""

    run_a: str
    run_b: str
    compatibility: CompatibilityReport
    deltas: tuple[MetricDelta, ...]
    #: Bootstrap CI for run A's requests/sec, when derivable.
    ci_a: tuple[float, float] | None = None
    ci_b: tuple[float, float] | None = None
    #: Whether the difference in requests/sec exceeds the combined spread.
    difference_exceeds_spread: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_a": self.run_a,
            "run_b": self.run_b,
            "compatibility": self.compatibility.to_dict(),
            "deltas": [d.to_dict() for d in self.deltas],
            "ci_a": self.ci_a,
            "ci_b": self.ci_b,
            "difference_exceeds_spread": self.difference_exceeds_spread,
            "notes": list(self.notes),
        }


def compare_runs(store: ExperimentStore, run_a: str, run_b: str) -> ComparisonResult:
    """Compare two stored runs, refusing an unfair comparison.

    Raises:
        RunNotFoundError: Either run is unknown.
        IncomparableRunsError: The runs are not comparable; the message lists
            every blocking difference.
    """
    metadata_a = store.get_metadata(run_a)
    metadata_b = store.get_metadata(run_b)

    compatibility = check_compatibility(metadata_a, metadata_b)
    if not compatibility.comparable:
        joined = "; ".join(compatibility.blockers)
        raise IncomparableRunsError(f"Runs {run_a!r} and {run_b!r} are not comparable: {joined}")

    summary_a = store.get_summary(run_a)
    summary_b = store.get_summary(run_b)

    latency_a = summary_a.get("latency", {})
    latency_b = summary_b.get("latency", {})
    throughput_a = summary_a.get("throughput", {})
    throughput_b = summary_b.get("throughput", {})
    failures_a = summary_a.get("failures", {})
    failures_b = summary_b.get("failures", {})

    deltas = (
        _delta(
            "TTFT p50",
            "s",
            latency_a.get("ttft", {}).get("p50"),
            latency_b.get("ttft", {}).get("p50"),
        ),
        _delta(
            "TTFT p95",
            "s",
            latency_a.get("ttft", {}).get("p95"),
            latency_b.get("ttft", {}).get("p95"),
        ),
        _delta(
            "TPOT p50",
            "s",
            latency_a.get("tpot", {}).get("p50"),
            latency_b.get("tpot", {}).get("p50"),
        ),
        _delta(
            "E2E p50",
            "s",
            latency_a.get("e2e", {}).get("p50"),
            latency_b.get("e2e", {}).get("p50"),
        ),
        _delta(
            "E2E p95",
            "s",
            latency_a.get("e2e", {}).get("p95"),
            latency_b.get("e2e", {}).get("p95"),
        ),
        _delta(
            "E2E p99",
            "s",
            latency_a.get("e2e", {}).get("p99"),
            latency_b.get("e2e", {}).get("p99"),
        ),
        _delta(
            "requests/sec",
            "req/s",
            throughput_a.get("requests_per_s"),
            throughput_b.get("requests_per_s"),
            lower_is_better=False,
        ),
        _delta(
            "output tokens/sec",
            "tok/s",
            throughput_a.get("output_tokens_per_s"),
            throughput_b.get("output_tokens_per_s"),
            lower_is_better=False,
        ),
        _delta(
            "failure rate",
            "",
            failures_a.get("failure_rate"),
            failures_b.get("failure_rate"),
        ),
    )

    notes = list(compatibility.notes)

    # Uncertainty: only meaningful with repeated runs.
    records_a = store.load_raw_records(run_a)
    records_b = store.load_raw_records(run_b)
    wall_a = float(throughput_a.get("wall_time_s") or 0.0)
    wall_b = float(throughput_b.get("wall_time_s") or 0.0)

    ci_a = (
        _throughput_ci(records_a, int(summary_a.get("repetitions", 1)), wall_a)
        if wall_a > 0
        else None
    )
    ci_b = (
        _throughput_ci(records_b, int(summary_b.get("repetitions", 1)), wall_b)
        if wall_b > 0
        else None
    )

    exceeds: bool | None = None
    rps_a = throughput_a.get("requests_per_s")
    rps_b = throughput_b.get("requests_per_s")
    if ci_a is not None and ci_b is not None and rps_a is not None and rps_b is not None:
        # The difference is credible only if the intervals do not overlap.
        overlaps = not (ci_b[0] > ci_a[1] or ci_a[0] > ci_b[1])
        exceeds = not overlaps
        if not exceeds:
            notes.append(
                "throughput confidence intervals overlap; the observed difference "
                "is within run-to-run variability and should not be treated as real"
            )
        else:
            notes.append(
                "throughput confidence intervals do not overlap; the difference "
                "exceeds observed run-to-run variability"
            )
    else:
        notes.append(
            "no confidence interval available (a single repetition, or too few "
            "samples); report the difference as a point estimate only"
        )

    # Record whether each run validated cleanly, so a comparison cannot quietly
    # rest on a broken result.
    for label, run_id in (("A", run_a), ("B", run_b)):
        report = validate_run(store, run_id)
        if not report.is_valid:
            notes.append(
                f"run {label} ({run_id}) has {len(report.errors)} validation error(s); "
                "interpret this comparison with caution"
            )

    return ComparisonResult(
        run_a=run_a,
        run_b=run_b,
        compatibility=compatibility,
        deltas=deltas,
        ci_a=ci_a,
        ci_b=ci_b,
        difference_exceeds_spread=exceeds,
        notes=notes,
    )


def _fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    if unit == "":
        return f"{value:.4f}"
    return f"{value:.4f} {unit}".strip()


def format_comparison(result: ComparisonResult) -> str:
    """Render a comparison as human-readable text."""
    lines: list[str] = []
    lines.append(f"Comparison: {result.run_a}  →  {result.run_b}")
    lines.append("=" * 72)

    if result.compatibility.warnings:
        lines.append("")
        lines.append("Warnings")
        lines.append("-" * 72)
        for warning in result.compatibility.warnings:
            lines.append(f"  ! {warning}")

    lines.append("")
    lines.append(f"{'metric':<20} {'run A':>14} {'run B':>14} {'delta':>14} {'%':>10}")
    lines.append("-" * 72)
    for delta in result.deltas:
        delta_text = "n/a" if delta.absolute_delta is None else f"{delta.absolute_delta:+.4f}"
        percent_text = "n/a" if delta.percent_delta is None else f"{delta.percent_delta:+.1f}%"
        lines.append(
            f"{delta.name:<20} {_fmt(delta.value_a, delta.unit):>14} "
            f"{_fmt(delta.value_b, delta.unit):>14} {delta_text:>14} {percent_text:>10}"
        )

    if result.ci_a is not None or result.ci_b is not None:
        lines.append("")
        lines.append("Uncertainty (95% bootstrap CI on requests/sec)")
        lines.append("-" * 72)
        ci_a_text = "n/a" if result.ci_a is None else f"{result.ci_a[0]:.4f} – {result.ci_a[1]:.4f}"
        ci_b_text = "n/a" if result.ci_b is None else f"{result.ci_b[0]:.4f} – {result.ci_b[1]:.4f}"
        lines.append(f"  run A: {ci_a_text}")
        lines.append(f"  run B: {ci_b_text}")

    if result.notes:
        lines.append("")
        lines.append("Notes")
        lines.append("-" * 72)
        for note in result.notes:
            lines.append(f"  - {note}")

    lines.append("")
    return "\n".join(lines)
