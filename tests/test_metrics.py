"""Tests for the metrics engine.

The central requirement (Definition of Done #2) is that streaming-based latency
metrics are validated against *controlled* data with known-correct answers.
These tests construct synthetic token streams whose TTFT/TPOT/E2E/percentiles
can be computed by hand, then assert the engine agrees exactly.

Anything that cannot be validated analytically is not asserted with a magic
number.
"""

from __future__ import annotations

import math

import pytest

from inferbench.metrics import (
    Percentiles,
    failure_stats,
    filter_measured,
    latency_stats,
    percentile,
    summarize,
    throughput_stats,
)
from inferbench.types import RequestRecord, RequestStatus


def make_record(
    *,
    request_id: str = "r0",
    status: RequestStatus = RequestStatus.success,
    ttft_s: float | None = 0.1,
    output_tokens: int = 5,
    tpot: float = 0.02,
    prompt_tokens: int | None = 10,
    is_warmup: bool = False,
    repetition: int = 0,
    class_label: str = "fixed",
    token_times: tuple[float, ...] | None = None,
) -> RequestRecord:
    """Build a record with an internally consistent, analytic timing model.

    Default shape: TTFT 0.1 s then ``output_tokens`` tokens spaced ``tpot``
    apart. E2E is derived, so expectations are exact rather than approximate.
    """
    if token_times is None:
        if ttft_s is None or output_tokens <= 0:
            token_times = ()
        else:
            token_times = tuple(ttft_s + i * tpot for i in range(output_tokens))
    e2e = token_times[-1] if token_times else (ttft_s or 0.0)
    return RequestRecord(
        request_id=request_id,
        status=status,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens if token_times else 0,
        requested_output_tokens=output_tokens,
        ttft_s=ttft_s if token_times else None,
        e2e_s=e2e,
        token_times_s=token_times,
        class_label=class_label,
        repetition=repetition,
        is_warmup=is_warmup,
        started_at=0.0,
    )


class TestPercentile:
    """Nearest-rank percentile, verified against hand-computed ranks."""

    def test_single_value(self) -> None:
        assert percentile([7.0], 50) == 7.0
        assert percentile([7.0], 99) == 7.0

    def test_known_ranks(self) -> None:
        # 10 values 1..10; nearest-rank indices are ceil(p/100*n)-1.
        data = [float(i) for i in range(1, 11)]
        assert percentile(data, 0) == 1.0  # rank 1  -> index 0
        assert percentile(data, 50) == 5.0  # ceil(5)-1 = 4 -> value 5
        assert percentile(data, 90) == 9.0  # ceil(9)-1 = 8 -> value 9
        assert percentile(data, 95) == 10.0  # ceil(9.5)=10 -> index 9
        assert percentile(data, 99) == 10.0
        assert percentile(data, 100) == 10.0

    def test_returns_observed_value_never_interpolated(self) -> None:
        data = [1.0, 2.0, 100.0]
        for p in (10, 33, 50, 67, 90, 99):
            assert percentile(data, p) in data

    def test_empty_returns_none(self) -> None:
        assert percentile([], 50) is None

    def test_unsorted_input_is_sorted(self) -> None:
        assert percentile([10.0, 1.0, 5.0], 50) == 5.0

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            percentile([1.0], 101)
        with pytest.raises(ValueError):
            percentile([1.0], -1)


class TestPercentilesDataclass:
    def test_empty(self) -> None:
        dist = Percentiles.from_values([])
        assert dist.count == 0
        assert dist.p50 is None
        assert dist.mean is None

    def test_mean_is_arithmetic(self) -> None:
        dist = Percentiles.from_values([1.0, 2.0, 3.0, 4.0])
        assert dist.mean == 2.5
        assert dist.min == 1.0
        assert dist.max == 4.0


class TestTtftValidation:
    """TTFT must equal the injected first-token time, exactly."""

    def test_ttft_matches_known_value(self) -> None:
        records = [make_record(ttft_s=0.25, output_tokens=4, tpot=0.01)]
        stats = latency_stats(records)
        assert stats.ttft.p50 == pytest.approx(0.25)
        assert stats.requests_with_ttft == 1

    def test_ttft_distribution_across_requests(self) -> None:
        records = [
            make_record(request_id=f"r{i}", ttft_s=value, output_tokens=3, tpot=0.01)
            for i, value in enumerate((0.1, 0.2, 0.3, 0.4, 0.5))
        ]
        stats = latency_stats(records)
        # n=5: p50 -> ceil(2.5)-1=2 -> 0.3 ; p95 -> ceil(4.75)-1=4 -> 0.5
        assert stats.ttft.p50 == pytest.approx(0.3)
        assert stats.ttft.p95 == pytest.approx(0.5)
        assert stats.ttft.count == 5


class TestTpotValidation:
    """TPOT must equal the injected inter-token interval, exactly."""

    def test_tpot_matches_injected_interval(self) -> None:
        # 5 tokens, TTFT 0.1, spacing 0.03 -> decode = 4*0.03 = 0.12
        # TPOT = 0.12 / (5-1) = 0.03
        record = make_record(ttft_s=0.1, output_tokens=5, tpot=0.03)
        stats = latency_stats([record])
        assert stats.tpot.p50 == pytest.approx(0.03)

    def test_single_token_has_no_tpot(self) -> None:
        """A one-token response has no inter-token interval; TPOT is undefined."""
        record = make_record(ttft_s=0.1, output_tokens=1)
        stats = latency_stats([record])
        assert stats.requests_with_tpot == 0
        assert stats.tpot.p50 is None
        # But TTFT is still defined.
        assert stats.requests_with_ttft == 1

    def test_zero_output_tokens_has_no_tpot(self) -> None:
        record = make_record(ttft_s=None, output_tokens=0, token_times=())
        stats = latency_stats([record])
        assert stats.tpot.p50 is None


class TestInterTokenLatency:
    def test_uniform_spacing_is_flat(self) -> None:
        record = make_record(ttft_s=0.1, output_tokens=5, tpot=0.02)
        stats = latency_stats([record])
        # 4 gaps, all exactly 0.02
        assert stats.inter_token.count == 4
        assert stats.inter_token.p50 == pytest.approx(0.02)
        assert stats.inter_token.max == pytest.approx(0.02)

    def test_pooled_across_requests(self) -> None:
        records = [
            make_record(request_id="a", ttft_s=0.1, output_tokens=3, tpot=0.01),
            make_record(request_id="b", ttft_s=0.2, output_tokens=3, tpot=0.01),
        ]
        stats = latency_stats(records)
        assert stats.inter_token.count == 4  # 2 gaps from each request

    def test_non_monotonic_times_are_ignored(self) -> None:
        # A backend glitch that reports a token before its predecessor must not
        # produce a negative gap.
        record = make_record(token_times=(0.1, 0.2, 0.15, 0.3))
        stats = latency_stats([record])
        assert stats.inter_token.min is not None
        assert stats.inter_token.min >= 0.0


class TestE2eValidation:
    def test_e2e_matches_last_token_time(self) -> None:
        record = make_record(ttft_s=0.1, output_tokens=5, tpot=0.02)
        stats = latency_stats([record])
        # 0.1 + 4*0.02 = 0.18
        assert stats.e2e.p50 == pytest.approx(0.18)


class TestFailureAccounting:
    def test_every_status_counted(self) -> None:
        records = [
            make_record(request_id="ok", status=RequestStatus.success),
            make_record(request_id="to", status=RequestStatus.timeout, ttft_s=None),
            make_record(request_id="cn", status=RequestStatus.cancelled, ttft_s=None),
            make_record(request_id="er", status=RequestStatus.error, ttft_s=None),
            make_record(request_id="in", status=RequestStatus.interrupted, ttft_s=None),
        ]
        stats = failure_stats(records)
        assert stats.total == 5
        assert stats.successes == 1
        assert stats.timeouts == 1
        assert stats.cancelled == 1
        assert stats.errors == 1
        assert stats.interrupted == 1
        assert stats.failure_rate == pytest.approx(0.8)

    def test_no_records_has_zero_rate(self) -> None:
        assert failure_stats([]).failure_rate == 0.0

    def test_failures_excluded_from_latency(self) -> None:
        records = [
            make_record(request_id="ok", ttft_s=0.1, output_tokens=3, tpot=0.01),
            make_record(request_id="bad", status=RequestStatus.timeout, ttft_s=None),
        ]
        stats = latency_stats(records)
        assert stats.ttft.count == 1  # the timeout contributes nothing


class TestThroughput:
    def test_requests_and_tokens_per_second(self) -> None:
        records = [
            make_record(request_id=f"r{i}", output_tokens=10, ttft_s=0.05, tpot=0.01)
            for i in range(4)
        ]
        stats = throughput_stats(records, wall_time_s=2.0)
        assert stats.successful_requests == 4
        assert stats.requests_per_s == pytest.approx(2.0)
        assert stats.output_tokens == 40
        assert stats.output_tokens_per_s == pytest.approx(20.0)

    def test_failures_do_not_count_toward_throughput(self) -> None:
        records = [
            make_record(request_id="ok", output_tokens=10),
            make_record(request_id="bad", status=RequestStatus.error, ttft_s=None),
        ]
        stats = throughput_stats(records, wall_time_s=1.0)
        assert stats.successful_requests == 1
        assert stats.failed_requests == 1
        assert stats.requests_per_s == pytest.approx(1.0)

    def test_offered_load_separate_from_completed(self) -> None:
        """Offered load must not be conflated with completed throughput."""
        records = [make_record(request_id="ok", output_tokens=5)]
        stats = throughput_stats(records, wall_time_s=1.0, measured_requests=10)
        assert stats.offered_requests_per_s == pytest.approx(10.0)
        assert stats.requests_per_s == pytest.approx(1.0)
        assert stats.offered_requests_per_s > stats.requests_per_s

    def test_nonpositive_wall_time_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            throughput_stats([make_record()], wall_time_s=0.0)


class TestFilterMeasured:
    def test_warmup_excluded(self) -> None:
        records = [
            make_record(request_id="w", is_warmup=True),
            make_record(request_id="m", is_warmup=False),
        ]
        measured = filter_measured(records)
        assert [r.request_id for r in measured] == ["m"]


class TestSummarize:
    def test_end_to_end_summary_is_consistent(self) -> None:
        records = [
            make_record(request_id=f"r{i}", ttft_s=0.1, output_tokens=5, tpot=0.02, repetition=0)
            for i in range(4)
        ]
        summary = summarize(records, wall_time_s=1.0)
        assert summary.latency.ttft.p50 == pytest.approx(0.1)
        assert summary.latency.tpot.p50 == pytest.approx(0.02)
        assert summary.throughput.requests_per_s == pytest.approx(4.0)
        assert summary.failures.successes == 4
        assert summary.repetitions == 1

    def test_summary_is_serializable_and_self_describing(self) -> None:
        summary = summarize([make_record()], wall_time_s=1.0)
        data = summary.to_dict()
        # The percentile method must travel with the numbers.
        assert data["aggregation"]["percentile_method"]
        assert data["aggregation"]["warmup_excluded"] is True
        assert set(data) >= {"latency", "throughput", "failures", "repetitions"}

    def test_repetition_variability_reported(self) -> None:
        records = [
            make_record(request_id="r0", repetition=0),
            make_record(request_id="r1", repetition=1),
        ]
        summary = summarize(records, wall_time_s=2.0, repetition_durations_s={0: 1.0, 1: 1.0})
        assert len(summary.per_repetition_rps) == 2
        assert all(value == pytest.approx(1.0) for value in summary.per_repetition_rps)

    def test_no_successes_yields_none_not_zero(self) -> None:
        """Absent data must be None, never a fabricated 0."""
        records = [make_record(status=RequestStatus.timeout, ttft_s=None)]
        summary = summarize(records, wall_time_s=1.0)
        assert summary.latency.ttft.p50 is None
        assert summary.latency.e2e.p50 is None
        assert summary.throughput.successful_requests == 0


class TestAnalyticConsistency:
    """Cross-check: recomputing metrics from raw times matches the summary."""

    def test_ttft_recomputable_from_token_times(self) -> None:
        record = make_record(ttft_s=0.137, output_tokens=8, tpot=0.011)
        # TTFT is, by definition, the first token's timestamp.
        assert record.ttft_s == record.token_times_s[0]

    def test_tpot_recomputable_from_raw_times(self) -> None:
        record = make_record(ttft_s=0.2, output_tokens=6, tpot=0.05)
        assert record.ttft_s is not None
        decode = record.e2e_s - record.ttft_s
        expected = decode / (record.output_tokens - 1)
        stats = latency_stats([record])
        assert stats.tpot.p50 == pytest.approx(expected)

    def test_mean_matches_manual_average(self) -> None:
        values = [0.01, 0.02, 0.03, 0.04, 0.05]
        records = [
            make_record(request_id=f"r{i}", ttft_s=value, output_tokens=2, tpot=0.01)
            for i, value in enumerate(values)
        ]
        stats = latency_stats(records)
        assert stats.ttft.mean is not None
        assert stats.ttft.mean == pytest.approx(sum(values) / len(values))
        assert math.isclose(stats.ttft.mean, 0.03, rel_tol=1e-12)
