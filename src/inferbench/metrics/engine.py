"""Metric computation from raw request records.

Everything here is a pure function of a sequence of
:class:`~inferbench.types.RequestRecord`. No I/O, no globals, no hidden state —
which is what makes the numbers testable against analytic ground truth.

Percentile method
-----------------
Nearest-rank on the sorted sample: for sorted values of size ``n`` and
percentile ``p`` (0–100), the reported value is the element at index
``ceil(p / 100 * n) - 1``, clamped to ``[0, n-1]``. This is stated in
``docs/methodology.md`` and is chosen because it always returns an observed
value (never an interpolated one that no request actually recorded).

What is deliberately *not* computed
-----------------------------------
- No queueing/prefill/decode split unless the backend genuinely reported the
  needed timestamps. TTFT and decode time are distinguished because streaming
  provides both; deeper splits are not invented.
- No GPU metrics: those live in the telemetry module and report "unavailable"
  rather than being estimated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from inferbench.types import RequestRecord, RequestStatus

__all__ = [
    "Percentiles",
    "LatencyStats",
    "ThroughputStats",
    "FailureStats",
    "MetricsSummary",
    "percentile",
    "latency_stats",
    "throughput_stats",
    "failure_stats",
    "summarize",
    "filter_measured",
]


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile. Returns ``None`` for an empty sample.

    Args:
        values: Sample values (need not be sorted).
        p: Percentile in ``[0, 100]``.

    Raises:
        ValueError: ``p`` is outside ``[0, 100]``.
    """
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {p}")
    if not values:
        return None
    ordered = sorted(values)
    # ceil(p/100 * n) - 1, clamped. Using math.ceil keeps intent explicit.
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    index = min(rank - 1, len(ordered) - 1)
    return ordered[index]


@dataclass(frozen=True, slots=True)
class Percentiles:
    """A distribution summarized at the percentiles InferBench reports."""

    count: int
    min: float | None
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    max: float | None
    mean: float | None

    @classmethod
    def from_values(cls, values: Sequence[float]) -> Percentiles:
        if not values:
            return cls(
                count=0, min=None, p50=None, p90=None, p95=None, p99=None, max=None, mean=None
            )
        return cls(
            count=len(values),
            min=min(values),
            p50=percentile(values, 50),
            p90=percentile(values, 90),
            p95=percentile(values, 95),
            p99=percentile(values, 99),
            max=max(values),
            mean=sum(values) / len(values),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Latency distributions, split by phase where the data supports it."""

    ttft: Percentiles
    e2e: Percentiles
    tpot: Percentiles
    inter_token: Percentiles
    #: Client-observed request duration / server-observed span are the same
    #: here; separated for future backends that expose server timing.
    requests_with_ttft: int
    requests_with_tpot: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft": self.ttft.to_dict(),
            "e2e": self.e2e.to_dict(),
            "tpot": self.tpot.to_dict(),
            "inter_token": self.inter_token.to_dict(),
            "requests_with_ttft": self.requests_with_ttft,
            "requests_with_tpot": self.requests_with_tpot,
        }


@dataclass(frozen=True, slots=True)
class ThroughputStats:
    """Completed-work throughput over the measured wall-clock window."""

    successful_requests: int
    failed_requests: int
    output_tokens: int
    prompt_tokens: int
    wall_time_s: float
    requests_per_s: float
    output_tokens_per_s: float
    #: Offered load is reported separately so it is never confused with
    #: completed throughput.
    offered_requests_per_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FailureStats:
    """Failure accounting. Failures are never silently dropped."""

    total: int
    successes: int
    timeouts: int
    cancelled: int
    errors: int
    interrupted: int

    @property
    def failure_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.total - self.successes) / self.total

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_rate"] = round(self.failure_rate, 6)
        return data


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    """Everything InferBench reports about one measured window.

    ``aggregation`` documents the percentile method and window semantics so a
    stored summary is self-describing.
    """

    latency: LatencyStats
    throughput: ThroughputStats
    failures: FailureStats
    #: Number of repetitions that contributed.
    repetitions: int
    #: Per-repetition throughput, used to report variability across repeats.
    per_repetition_rps: tuple[float, ...] = ()
    aggregation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency": self.latency.to_dict(),
            "throughput": self.throughput.to_dict(),
            "failures": self.failures.to_dict(),
            "repetitions": self.repetitions,
            "per_repetition_rps": list(self.per_repetition_rps),
            "aggregation": self.aggregation,
        }


def filter_measured(records: Sequence[RequestRecord]) -> list[RequestRecord]:
    """Drop warm-up records, which must never enter measured statistics."""
    return [record for record in records if not record.is_warmup]


def _ttft_values(records: Sequence[RequestRecord]) -> list[float]:
    return [r.ttft_s for r in records if r.is_success and r.ttft_s is not None]


def _tpot(record: RequestRecord) -> float | None:
    """Decode-phase time per output token for one request.

    Defined only when at least two tokens were produced (a single token has no
    inter-token interval) and the request succeeded.
    """
    if not record.is_success or record.ttft_s is None:
        return None
    if record.output_tokens < 2:
        return None
    decode_time = record.e2e_s - record.ttft_s
    if decode_time <= 0:
        return None
    return decode_time / (record.output_tokens - 1)


def _inter_token_values(records: Sequence[RequestRecord]) -> list[float]:
    """Flatten per-request inter-token gaps into one distribution."""
    gaps: list[float] = []
    for record in records:
        if not record.is_success:
            continue
        times = record.token_times_s
        for previous, current in zip(times, times[1:], strict=False):
            gap = current - previous
            if gap >= 0:
                gaps.append(gap)
    return gaps


def latency_stats(records: Sequence[RequestRecord]) -> LatencyStats:
    """Compute latency distributions from measured records."""
    successes = [r for r in records if r.is_success]
    ttft = _ttft_values(records)
    tpot_values = [value for r in successes if (value := _tpot(r)) is not None]
    return LatencyStats(
        ttft=Percentiles.from_values(ttft),
        e2e=Percentiles.from_values([r.e2e_s for r in successes]),
        tpot=Percentiles.from_values(tpot_values),
        inter_token=Percentiles.from_values(_inter_token_values(records)),
        requests_with_ttft=len(ttft),
        requests_with_tpot=len(tpot_values),
    )


def throughput_stats(
    records: Sequence[RequestRecord],
    *,
    wall_time_s: float,
    measured_requests: int | None = None,
) -> ThroughputStats:
    """Compute throughput over a wall-clock window.

    Args:
        records: Measured (non-warm-up) records.
        wall_time_s: Duration of the measured window. Must be positive.
        measured_requests: Offered request count; defaults to ``len(records)``.

    Raises:
        ValueError: ``wall_time_s`` is not positive.
    """
    if wall_time_s <= 0:
        raise ValueError(f"wall_time_s must be positive, got {wall_time_s}")

    successes = [r for r in records if r.is_success]
    failures = len(records) - len(successes)
    output_tokens = sum(r.output_tokens for r in successes)
    prompt_tokens = sum((r.prompt_tokens or 0) for r in successes)
    offered = measured_requests if measured_requests is not None else len(records)

    return ThroughputStats(
        successful_requests=len(successes),
        failed_requests=failures,
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        wall_time_s=round(wall_time_s, 6),
        requests_per_s=round(len(successes) / wall_time_s, 4),
        output_tokens_per_s=round(output_tokens / wall_time_s, 4),
        offered_requests_per_s=round(offered / wall_time_s, 4),
    )


def failure_stats(records: Sequence[RequestRecord]) -> FailureStats:
    """Count terminal states. Every dispatched request must appear in exactly one."""
    counts: dict[RequestStatus, int] = dict.fromkeys(RequestStatus, 0)
    for record in records:
        counts[record.status] += 1
    return FailureStats(
        total=len(records),
        successes=counts[RequestStatus.success],
        timeouts=counts[RequestStatus.timeout],
        cancelled=counts[RequestStatus.cancelled],
        errors=counts[RequestStatus.error],
        interrupted=counts[RequestStatus.interrupted],
    )


def _repetition_rps(
    records: Sequence[RequestRecord],
    repetition_durations_s: dict[int, float] | None,
    fallback_wall_time_s: float,
) -> tuple[float, ...]:
    """Per-repetition requests/sec, to expose run-to-run variability.

    The runner owns repetition windows (it times each one), so it passes them
    in. When durations are unavailable the global wall time is used as a
    fallback, which is conservative rather than flattering.
    """
    by_rep: dict[int, list[RequestRecord]] = {}
    for record in records:
        by_rep.setdefault(record.repetition, []).append(record)

    values: list[float] = []
    for rep in sorted(by_rep):
        reps = by_rep[rep]
        successes = [r for r in reps if r.is_success]
        duration = (repetition_durations_s or {}).get(rep, fallback_wall_time_s)
        if duration <= 0:
            values.append(0.0)
        else:
            values.append(round(len(successes) / duration, 4))
    return tuple(values)


def summarize(
    records: Sequence[RequestRecord],
    *,
    wall_time_s: float,
    measured_requests: int | None = None,
    repetition_durations_s: dict[int, float] | None = None,
) -> MetricsSummary:
    """Produce a complete, self-describing summary of a measured window.

    Warm-up records should already be excluded (use :func:`filter_measured`);
    this function does not filter them again so that callers stay explicit.
    """
    repetitions = len({r.repetition for r in records})
    return MetricsSummary(
        latency=latency_stats(records),
        throughput=throughput_stats(
            records, wall_time_s=wall_time_s, measured_requests=measured_requests
        ),
        failures=failure_stats(records),
        repetitions=repetitions,
        per_repetition_rps=_repetition_rps(records, repetition_durations_s, wall_time_s),
        aggregation={
            "percentile_method": "nearest-rank on sorted sample",
            "ttft": "dispatch -> first content token",
            "tpot": "(e2e - ttft) / (output_tokens - 1), successes with >=2 tokens only",
            "inter_token": "per-token arrival gaps, all successes pooled",
            "throughput": "completed successes / measured wall-clock window",
            "warmup_excluded": True,
        },
    )
