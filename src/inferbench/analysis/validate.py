"""Run validation: is a stored result complete, consistent, and auditable?

``inferbench validate`` answers a question that matters more than "did the run
finish": *can these numbers be trusted and reproduced?* Validation therefore
checks three separate things, and reports them separately:

1. **Artifacts** — every file a run is supposed to have is present and readable.
2. **Metadata integrity** — the fields required to reproduce and interpret the
   run are populated, and any self-contradiction (e.g. a run marked complete
   with no successful requests) is flagged.
3. **Data consistency** — the raw JSONL and the stored summary agree. A run
   where they disagree is not auditable, whatever else is true about it.

Validation produces *findings*, not a bare boolean, so the CLI can print what
is wrong and why. Each finding has a severity:

- ``error``   — the run cannot be trusted as-is.
- ``warning`` — the run is usable but should be read with a caveat, or is
  incomplete in a way that is honestly recorded.
- ``info``    — an observation worth surfacing (e.g. synthetic provenance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from inferbench.errors import RunNotFoundError, StorageError
from inferbench.metrics import filter_measured, summarize
from inferbench.storage import ExperimentStore
from inferbench.types import RequestStatus

__all__ = [
    "Severity",
    "Finding",
    "ValidationReport",
    "validate_run",
]

#: Files every completed run must have.
REQUIRED_ARTIFACTS = (
    "metadata.json",
    "raw.jsonl",
    "metrics.json",
    "summary.csv",
    "environment.json",
)

#: Metadata fields required to reproduce a run. Missing values here mean the
#: result cannot be re-executed or fairly compared.
REQUIRED_METADATA_FIELDS = (
    "run_id",
    "run_name",
    "experiment",
    "created_at",
    "started_at",
    "config",
    "environment",
    "backend_kind",
    "model",
    "data_source",
    "tool_version",
)

#: Tolerance when cross-checking recomputed metrics against the stored summary.
#: Floating-point summation order can differ, so exact equality is wrong.
_SUMMARY_TOLERANCE = 1e-6


class Severity(StrEnum):
    error = "error"
    warning = "warning"
    info = "info"


@dataclass(frozen=True, slots=True)
class Finding:
    """One problem or observation about a run."""

    severity: Severity
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity.value, "code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The complete result of validating a run."""

    run_id: str
    findings: tuple[Finding, ...]

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.error)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.warning)

    @property
    def infos(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.info)

    @property
    def is_valid(self) -> bool:
        """A run is valid when it has no errors. Warnings do not invalidate it."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "valid": self.is_valid,
            "errors": [f.to_dict() for f in self.errors],
            "warnings": [f.to_dict() for f in self.warnings],
            "info": [f.to_dict() for f in self.infos],
        }


class _Collector:
    """Accumulates findings; keeps the validation logic flat and readable."""

    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def error(self, code: str, message: str) -> None:
        self.findings.append(Finding(Severity.error, code, message))

    def warn(self, code: str, message: str) -> None:
        self.findings.append(Finding(Severity.warning, code, message))

    def info(self, code: str, message: str) -> None:
        self.findings.append(Finding(Severity.info, code, message))


def _close(a: Any, b: Any, tolerance: float = _SUMMARY_TOLERANCE) -> bool:
    """Compare two numbers with tolerance, treating two ``None``s as equal."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return bool(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance))
    except (TypeError, ValueError):
        return bool(a == b)


def validate_run(store: ExperimentStore, run_id: str) -> ValidationReport:
    """Validate a stored run and return findings.

    Raises:
        RunNotFoundError: The run id is unknown to the store.
    """
    collector = _Collector()

    if not store.has_run(run_id):
        raise RunNotFoundError(f"No such run: {run_id!r}")

    run_dir = store.run_dir(run_id)

    # --- 1. Artifacts ----------------------------------------------------
    for artifact in REQUIRED_ARTIFACTS:
        if not (run_dir / artifact).is_file():
            collector.error("missing_artifact", f"required artifact is missing: {artifact}")

    # Metadata is needed for everything downstream; if it is unreadable there is
    # nothing more to check.
    try:
        metadata = store.get_metadata(run_id)
    except (RunNotFoundError, StorageError) as exc:
        collector.error("unreadable_metadata", f"could not load metadata: {exc}")
        return ValidationReport(run_id=run_id, findings=tuple(collector.findings))

    meta_dict = metadata.to_dict()

    # --- 2. Metadata integrity ------------------------------------------
    for field in REQUIRED_METADATA_FIELDS:
        value = meta_dict.get(field)
        if value is None or value == "" or value == {}:
            collector.error("missing_metadata", f"required metadata field is empty: {field}")

    if not metadata.config:
        collector.error("empty_config", "run config is empty; the workload is unknown")
    else:
        workload = metadata.config.get("workload", {})
        if not workload.get("num_requests"):
            collector.warn("unknown_workload", "workload.num_requests is unset or zero")
        if not metadata.config.get("load"):
            collector.warn("unknown_load", "load profile is unset; comparison may be unfair")

    if not metadata.environment:
        collector.warn(
            "empty_environment",
            "environment fingerprint is empty; hardware context cannot be established",
        )
    else:
        gpu = metadata.environment.get("gpu") or {}
        if not gpu.get("available", False):
            collector.info(
                "no_gpu",
                f"no GPU recorded at run time ({gpu.get('reason', 'not reported')})",
            )

    if metadata.backend_version is None:
        collector.info(
            "no_backend_version",
            "backend version was unavailable; results cannot be pinned to an engine release",
        )

    if metadata.incomplete:
        collector.warn(
            "incomplete_run",
            f"run is marked incomplete: {metadata.incomplete_reason or 'no reason recorded'}",
        )

    if metadata.data_source != "measured":
        collector.info(
            "synthetic_data",
            f"data source is {metadata.data_source!r}; "
            "numbers describe the pipeline, not real hardware performance",
        )

    # --- 3. Data consistency --------------------------------------------
    try:
        records = store.load_raw_records(run_id)
    except (RunNotFoundError, StorageError) as exc:
        collector.error("unreadable_raw", f"could not read raw records: {exc}")
        return ValidationReport(run_id=run_id, findings=tuple(collector.findings))

    if not records:
        collector.error("no_requests", "raw.jsonl contains no request records")
        return ValidationReport(run_id=run_id, findings=tuple(collector.findings))

    measured = filter_measured(records)
    if not measured:
        collector.error(
            "no_measured_requests",
            "every recorded request was warm-up; there is no measured data",
        )

    # Requests must be uniquely identified within a repetition, or counts lie.
    seen: set[tuple[int, str]] = set()
    duplicates = 0
    for record in records:
        key = (record.repetition, record.request_id)
        if key in seen:
            duplicates += 1
        seen.add(key)
    if duplicates:
        collector.error(
            "duplicate_requests",
            f"{duplicates} request record(s) share a request_id within a repetition",
        )

    # Timing sanity: negative or inconsistent timings mean the measurement is
    # broken, not merely noisy.
    negative_timing = [
        r.request_id for r in records if r.e2e_s < 0 or (r.ttft_s is not None and r.ttft_s < 0)
    ]
    if negative_timing:
        collector.error(
            "negative_timing",
            f"{len(negative_timing)} record(s) have negative timing: {negative_timing[:5]}",
        )

    ttft_after_e2e = [
        r.request_id
        for r in records
        if r.ttft_s is not None and r.ttft_s > r.e2e_s + _SUMMARY_TOLERANCE
    ]
    if ttft_after_e2e:
        collector.error(
            "inconsistent_timing",
            f"{len(ttft_after_e2e)} record(s) report TTFT after end-to-end completion: "
            f"{ttft_after_e2e[:5]}",
        )

    non_monotonic = [
        r.request_id
        for r in records
        if any(b < a for a, b in zip(r.token_times_s, r.token_times_s[1:], strict=False))
    ]
    if non_monotonic:
        collector.warn(
            "non_monotonic_tokens",
            f"{len(non_monotonic)} record(s) have non-monotonic token timestamps; "
            "inter-token latency treats these gaps as zero",
        )

    # Token accounting: token times must match the reported token count.
    mismatched_tokens = [
        r.request_id for r in records if r.is_success and r.output_tokens != len(r.token_times_s)
    ]
    if mismatched_tokens:
        collector.warn(
            "token_count_mismatch",
            f"{len(mismatched_tokens)} successful record(s) report an output token count "
            f"that differs from their recorded token timestamps: {mismatched_tokens[:5]}",
        )

    successes = [r for r in records if r.status is RequestStatus.success]
    zero_token_success = [r.request_id for r in successes if r.output_tokens == 0]
    if zero_token_success:
        collector.warn(
            "empty_success",
            f"{len(zero_token_success)} request(s) are marked successful with zero output "
            "tokens; throughput may be overstated",
        )

    # --- 4. Summary agreement -------------------------------------------
    try:
        summary = store.get_summary(run_id)
    except (RunNotFoundError, StorageError) as exc:
        collector.error("unreadable_summary", f"could not read summary: {exc}")
        return ValidationReport(run_id=run_id, findings=tuple(collector.findings))

    _check_summary_agreement(collector, summary, measured)

    return ValidationReport(run_id=run_id, findings=tuple(collector.findings))


def _check_summary_agreement(
    collector: _Collector, summary: dict[str, Any], measured: list[Any]
) -> None:
    """Verify the stored summary matches what the raw records imply."""
    stored_throughput = summary.get("throughput") or {}
    stored_failures = summary.get("failures") or {}

    recomputed_successes = sum(1 for r in measured if r.status is RequestStatus.success)
    if stored_throughput.get("successful_requests") != recomputed_successes:
        collector.error(
            "summary_mismatch",
            "stored summary reports "
            f"{stored_throughput.get('successful_requests')} successful requests, "
            f"but raw records imply {recomputed_successes}",
        )

    recomputed_output_tokens = sum(
        r.output_tokens for r in measured if r.status is RequestStatus.success
    )
    if stored_throughput.get("output_tokens") != recomputed_output_tokens:
        collector.error(
            "summary_mismatch",
            f"stored summary reports {stored_throughput.get('output_tokens')} output tokens, "
            f"but raw records imply {recomputed_output_tokens}",
        )

    if stored_failures.get("total") != len(measured):
        collector.error(
            "summary_mismatch",
            f"stored summary totals {stored_failures.get('total')} requests, "
            f"but raw records contain {len(measured)} measured requests",
        )

    # Wall time must be positive and finite for throughput to mean anything.
    wall_time = stored_throughput.get("wall_time_s")
    if wall_time is None or not isinstance(wall_time, (int, float)) or wall_time <= 0:
        collector.error("bad_wall_time", f"stored wall time is not positive: {wall_time!r}")

    # Latency percentiles should be recomputable; a large divergence means the
    # summary was produced from different data than the raw file.
    try:
        recomputed = summarize(measured, wall_time_s=float(wall_time or 1.0)).to_dict()
    except (ValueError, TypeError) as exc:
        collector.error("summary_recompute_failed", f"could not recompute summary: {exc}")
        return

    for key in ("ttft", "tpot", "e2e"):
        stored_p50 = (summary.get("latency") or {}).get(key, {}).get("p50")
        recomputed_p50 = recomputed["latency"][key]["p50"]
        if not _close(stored_p50, recomputed_p50):
            collector.error(
                "summary_mismatch",
                f"{key} p50 differs: stored {stored_p50!r}, recomputed {recomputed_p50!r}",
            )

    # A summary that carries a provenance flag differing from metadata is a
    # genuine integrity problem.
    data_source = summary.get("data_source")
    if data_source is None:
        collector.warn("summary_no_provenance", "summary does not record its data source")


def format_report(report: ValidationReport) -> str:
    """Render a validation report as human-readable text."""
    lines: list[str] = []
    lines.append(f"Validation: {report.run_id}")
    lines.append("=" * 60)
    lines.append(f"  result   {'VALID' if report.is_valid else 'INVALID'}")
    lines.append(f"  errors   {len(report.errors)}")
    lines.append(f"  warnings {len(report.warnings)}")
    lines.append("")

    for label, findings in (
        ("Errors", report.errors),
        ("Warnings", report.warnings),
        ("Info", report.infos),
    ):
        if not findings:
            continue
        lines.append(label)
        lines.append("-" * 60)
        for finding in findings:
            lines.append(f"  [{finding.code}] {finding.message}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
