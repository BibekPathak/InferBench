"""Tests for run comparison and the fairness gate.

The fairness gate is the feature that makes comparison trustworthy, so most of
these tests are about *refusals*: two runs that differ in a way that makes the
comparison meaningless must be rejected, with the specific reason named.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from inferbench.analysis import (
    bootstrap_ci,
    check_compatibility,
    compare_runs,
    format_comparison,
)
from inferbench.errors import IncomparableRunsError, RunNotFoundError
from inferbench.storage import ExperimentStore
from inferbench.types import RequestRecord, RequestStatus, RunMetadata

StoreRun = Callable[..., str]
MakeMetadata = Callable[..., RunMetadata]
MakeRecords = Callable[..., list[RequestRecord]]


def _workload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "kind": "fixed",
        "num_requests": 4,
        "prompt_tokens": 128,
        "output_tokens": 32,
        "seed": 0,
    }
    base.update(overrides)
    return base


def _load(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "pattern": "closed_loop",
        "concurrency": 1,
        "warmup_requests": 0,
        "repetitions": 1,
    }
    base.update(overrides)
    return base


class TestCompatibilityGate:
    """check_compatibility must accept legitimate differences and block unfair ones."""

    def test_identical_runs_are_comparable(self, make_metadata: MakeMetadata) -> None:
        report = check_compatibility(make_metadata("a"), make_metadata("b"))
        assert report.comparable
        assert report.blockers == ()

    def test_concurrency_difference_is_allowed_and_noted(self, make_metadata: MakeMetadata) -> None:
        a = make_metadata("a", load=_load(concurrency=1))
        b = make_metadata("b", load=_load(concurrency=8))
        report = check_compatibility(a, b)
        assert report.comparable
        assert any("concurrency" in note for note in report.notes)
        # The experimental axis must not be framed as a problem.
        assert not any("concurrency" in w for w in report.warnings)

    def test_different_prompt_lengths_blocked(self, make_metadata: MakeMetadata) -> None:
        a = make_metadata("a", workload=_workload(prompt_tokens=128))
        b = make_metadata("b", workload=_workload(prompt_tokens=512))
        report = check_compatibility(a, b)
        assert not report.comparable
        assert any("prompt token distribution" in blocker for blocker in report.blockers)

    def test_different_output_lengths_blocked(self, make_metadata: MakeMetadata) -> None:
        a = make_metadata("a", workload=_workload(output_tokens=32))
        b = make_metadata("b", workload=_workload(output_tokens=256))
        report = check_compatibility(a, b)
        assert not report.comparable
        assert any("output token distribution" in blocker for blocker in report.blockers)

    def test_different_workload_kind_blocked(self, make_metadata: MakeMetadata) -> None:
        a = make_metadata("a", workload=_workload(kind="fixed"))
        b = make_metadata(
            "b",
            workload={
                "kind": "randomized",
                "num_requests": 4,
                "prompt_tokens_min": 64,
                "prompt_tokens_max": 256,
                "output_tokens": 32,
            },
        )
        report = check_compatibility(a, b)
        assert not report.comparable
        assert any("workload kind" in blocker for blocker in report.blockers)

    def test_different_request_counts_blocked(self, make_metadata: MakeMetadata) -> None:
        a = make_metadata("a", workload=_workload(num_requests=4))
        b = make_metadata("b", workload=_workload(num_requests=64))
        report = check_compatibility(a, b)
        assert not report.comparable
        assert any("request counts" in blocker for blocker in report.blockers)

    def test_different_models_warned_not_blocked(self, make_metadata: MakeMetadata) -> None:
        a = make_metadata("a", model="model-x")
        b = make_metadata("b", model="model-y")
        report = check_compatibility(a, b)
        assert report.comparable
        assert any("different models" in w for w in report.warnings)

    def test_measured_vs_synthetic_warned(self, make_metadata: MakeMetadata) -> None:
        a = make_metadata("a", data_source="measured")
        b = make_metadata("b", data_source="synthetic")
        report = check_compatibility(a, b)
        assert report.comparable
        assert any("real and synthetic" in w for w in report.warnings)

    def test_different_warmup_warned(self, make_metadata: MakeMetadata) -> None:
        a = make_metadata("a", load=_load(warmup_requests=0))
        b = make_metadata("b", load=_load(warmup_requests=8))
        report = check_compatibility(a, b)
        assert report.comparable
        assert any("warm-up" in w for w in report.warnings)

    def test_randomized_bounds_compare_against_same_bounds(
        self, make_metadata: MakeMetadata
    ) -> None:
        """Two randomized runs with the same bounds are compatible."""
        workload = {
            "kind": "randomized",
            "num_requests": 4,
            "prompt_tokens_min": 64,
            "prompt_tokens_max": 256,
            "output_tokens_min": 16,
            "output_tokens_max": 64,
        }
        a = make_metadata("a", workload=dict(workload))
        b = make_metadata("b", workload=dict(workload))
        assert check_compatibility(a, b).comparable

    def test_fixed_workload_uses_configured_value_as_distribution(
        self, make_metadata: MakeMetadata
    ) -> None:
        a = make_metadata("a", workload=_workload(prompt_tokens=128, output_tokens=32))
        b = make_metadata("b", workload=_workload(prompt_tokens=128, output_tokens=32))
        assert check_compatibility(a, b).comparable


class TestCompareRuns:
    def test_comparison_produces_deltas(self, tmp_path: Path, stored_run: StoreRun) -> None:
        with ExperimentStore(tmp_path) as store:
            a = stored_run(store, run_id="run-a", wall_time_s=1.0)
            b = stored_run(store, run_id="run-b", wall_time_s=2.0)
            result = compare_runs(store, a, b)

        assert result.run_a == a
        assert result.run_b == b
        names = {d.name for d in result.deltas}
        assert {"TTFT p50", "requests/sec", "output tokens/sec", "failure rate"} <= names

    def test_duplicate_run_id_raises(self, tmp_path: Path, stored_run: StoreRun) -> None:
        """Comparing a run against itself is a no-op, not an error."""
        with ExperimentStore(tmp_path) as store:
            a = stored_run(store, run_id="run-a")
            result = compare_runs(store, a, a)
        assert result.run_a == result.run_b

    def test_unknown_run_raises(self, tmp_path: Path, stored_run: StoreRun) -> None:
        with ExperimentStore(tmp_path) as store:
            a = stored_run(store, run_id="run-a")
            with pytest.raises(RunNotFoundError):
                compare_runs(store, a, "ghost")

    def test_incomparable_runs_refused_with_reason(
        self, tmp_path: Path, stored_run: StoreRun, make_metadata: MakeMetadata
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            a = stored_run(
                store,
                run_id="run-a",
                metadata=make_metadata("run-a", workload=_workload(prompt_tokens=128)),
            )
            b = stored_run(
                store,
                run_id="run-b",
                metadata=make_metadata("run-b", workload=_workload(prompt_tokens=512)),
            )
            with pytest.raises(IncomparableRunsError, match="not comparable"):
                compare_runs(store, a, b)

    def test_refusal_message_names_every_blocker(
        self, tmp_path: Path, stored_run: StoreRun, make_metadata: MakeMetadata
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            a = stored_run(
                store,
                run_id="run-a",
                metadata=make_metadata(
                    "run-a", workload=_workload(prompt_tokens=128, output_tokens=32)
                ),
            )
            b = stored_run(
                store,
                run_id="run-b",
                metadata=make_metadata(
                    "run-b", workload=_workload(prompt_tokens=512, output_tokens=256)
                ),
            )
            with pytest.raises(IncomparableRunsError) as excinfo:
                compare_runs(store, a, b)

        message = str(excinfo.value)
        assert "prompt token distribution" in message
        assert "output token distribution" in message

    def test_comparison_serializable(self, tmp_path: Path, stored_run: StoreRun) -> None:
        import json

        with ExperimentStore(tmp_path) as store:
            a = stored_run(store, run_id="run-a")
            b = stored_run(store, run_id="run-b")
            payload = compare_runs(store, a, b).to_dict()

        round_tripped = json.loads(json.dumps(payload, default=str))
        assert round_tripped["run_a"] == "run-a"
        assert isinstance(round_tripped["deltas"], list)

    def test_delta_direction_is_interpreted(self, tmp_path: Path, stored_run: StoreRun) -> None:
        """Lower latency and higher throughput must be recognisable as improvements."""
        with ExperimentStore(tmp_path) as store:
            a = stored_run(store, run_id="run-a", wall_time_s=2.0)
            b = stored_run(store, run_id="run-b", wall_time_s=1.0)
            result = compare_runs(store, a, b)

        rps = next(d for d in result.deltas if d.name == "requests/sec")
        assert rps.lower_is_better is False
        # B is faster, so throughput improved.
        assert rps.improved is True

        ttft = next(d for d in result.deltas if d.name == "TTFT p50")
        assert ttft.lower_is_better is True

    def test_failed_requests_reduce_reported_throughput(
        self, tmp_path: Path, stored_run: StoreRun, make_records: MakeRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            a = stored_run(store, run_id="run-a", records=make_records(count=4), wall_time_s=1.0)
            b = stored_run(
                store,
                run_id="run-b",
                records=make_records(count=4, statuses={0: RequestStatus.error}),
                wall_time_s=1.0,
            )
            result = compare_runs(store, a, b)

        rps = next(d for d in result.deltas if d.name == "requests/sec")
        assert rps.value_a == pytest.approx(4.0)
        assert rps.value_b == pytest.approx(3.0)

    def test_notes_flag_invalid_runs(self, tmp_path: Path, stored_run: StoreRun) -> None:
        """A comparison must not quietly rest on an invalid result."""
        with ExperimentStore(tmp_path) as store:
            a = stored_run(store, run_id="run-a")
            # Corrupt B's summary so it fails validation.
            b = stored_run(store, run_id="run-b")
            summary = store.get_summary(b)
            summary["throughput"]["successful_requests"] = 9999
            store.save_summary(b, summary)

            result = compare_runs(store, a, b)

        assert any("validation error" in note for note in result.notes)


class TestUncertainty:
    def test_bootstrap_ci_is_deterministic(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        first = bootstrap_ci(values, seed=7)
        second = bootstrap_ci(values, seed=7)
        assert first == second

    def test_bootstrap_ci_brackets_the_mean(self) -> None:
        values = [float(i) for i in range(1, 21)]
        low, high = bootstrap_ci(values, seed=1)  # type: ignore[misc]
        mean = sum(values) / len(values)
        assert low <= mean <= high

    def test_bootstrap_needs_two_samples(self) -> None:
        assert bootstrap_ci([1.0]) is None
        assert bootstrap_ci([]) is None

    def test_bootstrap_rejects_bad_confidence(self) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            bootstrap_ci([1.0, 2.0], confidence=1.5)

    def test_bootstrap_rejects_too_few_resamples(self) -> None:
        with pytest.raises(ValueError, match="resamples"):
            bootstrap_ci([1.0, 2.0], resamples=10)

    def test_single_repetition_yields_no_ci(
        self, tmp_path: Path, stored_run: StoreRun, make_records: MakeRecords
    ) -> None:
        """One repetition cannot support an uncertainty estimate."""
        with ExperimentStore(tmp_path) as store:
            a = stored_run(store, run_id="run-a", records=make_records(count=4, repetitions=1))
            b = stored_run(store, run_id="run-b", records=make_records(count=4, repetitions=1))
            result = compare_runs(store, a, b)

        assert result.ci_a is None
        assert result.ci_b is None
        assert result.difference_exceeds_spread is None
        assert any("no confidence interval" in note for note in result.notes)

    def test_multiple_repetitions_yield_ci(
        self, tmp_path: Path, stored_run: StoreRun, make_records: MakeRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            a = stored_run(store, run_id="run-a", records=make_records(count=4, repetitions=4))
            b = stored_run(store, run_id="run-b", records=make_records(count=4, repetitions=4))
            result = compare_runs(store, a, b)

        assert result.ci_a is not None
        assert result.ci_b is not None

    def test_ci_reported_in_text(
        self, tmp_path: Path, stored_run: StoreRun, make_records: MakeRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            a = stored_run(store, run_id="run-a", records=make_records(count=4, repetitions=3))
            b = stored_run(store, run_id="run-b", records=make_records(count=4, repetitions=3))
            text = format_comparison(compare_runs(store, a, b))

        assert "Uncertainty" in text
        assert "bootstrap" in text.lower()


class TestFormatting:
    def test_text_includes_warnings_and_notes(
        self, tmp_path: Path, stored_run: StoreRun, make_metadata: MakeMetadata
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            a = stored_run(store, run_id="run-a", metadata=make_metadata("run-a", model="model-x"))
            b = stored_run(store, run_id="run-b", metadata=make_metadata("run-b", model="model-y"))
            text = format_comparison(compare_runs(store, a, b))

        assert "Warnings" in text
        assert "different models" in text
        assert "Notes" in text
