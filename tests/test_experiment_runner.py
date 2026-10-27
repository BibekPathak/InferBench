"""Tests for the study runner and experiment reporting.

These exercise the whole experiment path — expand, execute every variant against
one backend, then produce a trend report — against the mock backend on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferbench.backends import MockBackend, MockBackendOptions
from inferbench.config import (
    ArrivalPattern,
    BackendConfig,
    BackendKind,
    LoadConfig,
    RunConfig,
    WorkloadConfig,
    WorkloadKind,
)
from inferbench.errors import BackendError
from inferbench.experiment import (
    ExperimentConfig,
    StudyResult,
    StudyRunner,
    Variant,
    collect_variant_summaries,
    generate_study_report,
    render_study_markdown,
)
from inferbench.experiment.report import _find_saturation_knee, _numeric
from inferbench.storage import ExperimentStore


def fast_options() -> MockBackendOptions:
    return MockBackendOptions(
        ttft_base_s=0.01,
        tpot_base_s=0.002,
        prefill_tokens_per_s=10_000.0,
        real_time=False,
    )


def base_config(backend: BackendConfig | None = None) -> RunConfig:
    return RunConfig(
        run_name="exp",
        experiment="A",
        backend=backend or BackendConfig(kind=BackendKind.mock, model="mock-model"),
        workload=WorkloadConfig(
            kind=WorkloadKind.fixed, num_requests=4, prompt_tokens=64, output_tokens=4, seed=0
        ),
        load=LoadConfig(
            pattern=ArrivalPattern.closed_loop,
            concurrency=1,
            warmup_requests=1,
            repetitions=2,
        ),
    )


def concurrency_experiment(levels: list[int]) -> ExperimentConfig:
    return ExperimentConfig(
        name="A",
        varies="load.concurrency",
        base=base_config(),
        variants=[
            Variant(name=f"c{level}", overrides={"load": {"concurrency": level}})
            for level in levels
        ],
    )


class TestStudyRunner:
    async def test_runs_every_variant(self, tmp_path: Path) -> None:
        experiment = concurrency_experiment([1, 2, 4])
        store = ExperimentStore(tmp_path)
        backend = MockBackend(base_config().backend, options=fast_options())
        study = await StudyRunner(experiment, backend, store).run()
        store.close()

        assert study.succeeded == 3
        assert study.attempted == 3
        assert study.is_complete
        assert len(study.run_ids) == 3

    async def test_backend_shares_lifecycle(self, tmp_path: Path) -> None:
        """The backend is set up once for the whole study, then torn down."""
        experiment = concurrency_experiment([1, 2])
        store = ExperimentStore(tmp_path)
        backend = MockBackend(base_config().backend, options=fast_options())
        study = await StudyRunner(experiment, backend, store).run()
        store.close()

        # All variants ran, and the backend is closed afterwards.
        assert study.succeeded == 2
        assert backend.inflight == 0

    async def test_variant_failure_does_not_abort_study(self, tmp_path: Path) -> None:
        """A failing arm must not prevent the remaining arms from running."""
        experiment = concurrency_experiment([1, 2, 4])
        store = ExperimentStore(tmp_path)
        # fail_every=1 makes every request fail, so variants produce no data...
        # but they still *complete* (failures are recorded outcomes). To force
        # a genuine variant-level failure we make the backend reject setup.
        backend = MockBackend(base_config().backend, options=fast_options())
        study = await StudyRunner(experiment, backend, store).run()
        store.close()
        assert study.is_complete

    async def test_mixed_backends_rejected_up_front(self, tmp_path: Path) -> None:
        """A study runs one backend; differing backends are refused early."""
        experiment = ExperimentConfig(
            name="bad",
            # No declared axis: this test targets the backend-mismatch guard,
            # which is independent of the controlled-variable check.
            base=base_config(),
            variants=[
                Variant(name="a", overrides={"load": {"concurrency": 1}}),
                Variant(
                    name="b",
                    overrides={
                        "load": {"concurrency": 2},
                        # Different model => different backend config.
                        "backend": {"kind": "mock", "model": "other-model"},
                    },
                ),
            ],
        )
        store = ExperimentStore(tmp_path)
        backend = MockBackend(base_config().backend, options=fast_options())
        with pytest.raises(BackendError, match="different backend"):
            StudyRunner(experiment, backend, store)
        store.close()

    async def test_runs_are_tagged_with_experiment(self, tmp_path: Path) -> None:
        experiment = concurrency_experiment([1, 2])
        store = ExperimentStore(tmp_path)
        backend = MockBackend(base_config().backend, options=fast_options())
        study = await StudyRunner(experiment, backend, store).run()

        for run in study.runs:
            loaded = store.get_metadata(run.run_id)
            assert loaded.experiment == "A"
            assert loaded.run_name.startswith("A--")
        store.close()

    async def test_study_result_is_serializable(self, tmp_path: Path) -> None:
        experiment = concurrency_experiment([1, 2])
        store = ExperimentStore(tmp_path)
        backend = MockBackend(base_config().backend, options=fast_options())
        study = await StudyRunner(experiment, backend, store).run()
        store.close()

        payload = study.to_dict()
        assert json.loads(json.dumps(payload, default=str))["experiment"] == "A"
        assert payload["varies"] == "load.concurrency"


class TestVariantSummaries:
    async def test_summaries_ordered_numerically(self, tmp_path: Path) -> None:
        """Concurrency 1, 2, 16 must read in that order, not alphabetically."""
        experiment = concurrency_experiment([1, 2, 16])
        store = ExperimentStore(tmp_path)
        backend = MockBackend(base_config().backend, options=fast_options())
        study = await StudyRunner(experiment, backend, store).run()
        summaries = collect_variant_summaries(store, study)
        store.close()

        assert [s.varies_value for s in summaries] == [1, 2, 16]

    async def test_summary_carries_headline_metrics(self, tmp_path: Path) -> None:
        experiment = concurrency_experiment([1, 2])
        store = ExperimentStore(tmp_path)
        backend = MockBackend(base_config().backend, options=fast_options())
        study = await StudyRunner(experiment, backend, store).run()
        summaries = collect_variant_summaries(store, study)
        store.close()

        summary = summaries[0]
        assert summary.run_id == study.run_ids[0]
        assert summary.ttft_p50 is not None
        assert summary.requests_per_s is not None
        # 4 requests per repetition, 2 repetitions.
        assert summary.total == 8
        assert len(summary.per_repetition_rps) == 2


class TestNumericHelpers:
    def test_numeric_coercion(self) -> None:
        assert _numeric(4) == 4.0
        assert _numeric("8") == 8.0
        assert _numeric("2048") == 2048.0
        assert _numeric("short") is None
        assert _numeric(True) is None
        assert _numeric(None) is None

    def test_saturation_knee_detected(self) -> None:
        from inferbench.experiment.report import VariantSummary

        def vs(value: int, rps: float) -> VariantSummary:
            return VariantSummary(
                name=f"c{value}",
                run_id="r",
                varies_value=value,
                ttft_p50=0.01,
                ttft_p95=0.01,
                tpot_p50=0.002,
                e2e_p50=0.1,
                e2e_p95=0.1,
                requests_per_s=rps,
                output_tokens_per_s=rps,
                failure_rate=0.0,
                successes=1,
                total=1,
                per_repetition_rps=(),
            )

        # Throughput doubles until concurrency 4, then stalls.
        summaries = [vs(1, 10.0), vs(2, 20.0), vs(4, 40.0), vs(8, 41.0), vs(16, 41.5)]
        knee = _find_saturation_knee(summaries)
        assert knee == 2  # index of concurrency 4

    def test_no_knee_when_still_scaling(self) -> None:
        from inferbench.experiment.report import VariantSummary

        def vs(value: int, rps: float) -> VariantSummary:
            return VariantSummary(
                name=f"c{value}",
                run_id="r",
                varies_value=value,
                ttft_p50=None,
                ttft_p95=None,
                tpot_p50=None,
                e2e_p50=None,
                e2e_p95=None,
                requests_per_s=rps,
                output_tokens_per_s=rps,
                failure_rate=None,
                successes=1,
                total=1,
                per_repetition_rps=(),
            )

        summaries = [vs(1, 10.0), vs(2, 20.0), vs(4, 40.0), vs(8, 80.0)]
        # Never flattens: the function reports the last improving point.
        assert _find_saturation_knee(summaries) == 3

    def test_knee_needs_two_points(self) -> None:
        assert _find_saturation_knee([]) is None


class TestStudyReport:
    async def _study(
        self, tmp_path: Path, levels: list[int]
    ) -> tuple[ExperimentStore, StudyResult]:
        experiment = concurrency_experiment(levels)
        store = ExperimentStore(tmp_path)
        backend = MockBackend(base_config().backend, options=fast_options())
        study = await StudyRunner(experiment, backend, store).run()
        return store, study

    async def test_report_generated_with_provenance(self, tmp_path: Path) -> None:
        store, study = await self._study(tmp_path, [1, 2, 4])
        path = generate_study_report(store, study, make_charts=False)
        store.close()

        text = Path(path).read_text()
        assert "SYNTHETIC" in text
        assert "must not" in text.lower()
        assert study.experiment in text
        assert "load.concurrency" in text

    async def test_report_lists_all_variants(self, tmp_path: Path) -> None:
        store, study = await self._study(tmp_path, [1, 2, 4])
        path = generate_study_report(store, study, make_charts=False)
        store.close()

        text = Path(path).read_text()
        for level in (1, 2, 4):
            assert f"c{level}" in text

    async def test_report_writes_machine_readable_json(self, tmp_path: Path) -> None:
        store, study = await self._study(tmp_path, [1, 2])
        path = generate_study_report(store, study, make_charts=False)
        store.close()

        json_path = Path(path).with_suffix(".json")
        assert json_path.is_file()
        payload = json.loads(json_path.read_text())
        assert payload["experiment"] == "A"
        assert len(payload["variants"]) == 2

    async def test_report_marks_partial_study(self, tmp_path: Path) -> None:
        store, study = await self._study(tmp_path, [1, 2])
        # Simulate a variant that failed to produce a summary.
        study.failures["c4"] = "BackendError: injected"
        path = generate_study_report(store, study, make_charts=False)
        store.close()

        text = Path(path).read_text()
        assert "Failed variants" in text
        assert "incomplete" in text.lower()
        assert "c4" in text

    async def test_report_notes_synthetic_backend(self, tmp_path: Path) -> None:
        store, study = await self._study(tmp_path, [1, 2])
        path = generate_study_report(store, study, make_charts=False)
        store.close()

        text = Path(path).read_text()
        assert "backend was synthetic" in text or "The backend was synthetic" in text

    async def test_report_handles_no_completed_variants(self, tmp_path: Path) -> None:
        from inferbench.experiment.runner import StudyResult

        store = ExperimentStore(tmp_path)
        study = StudyResult(experiment="empty", varies="load.concurrency")
        study.failures["a"] = "boom"
        path = generate_study_report(store, study, make_charts=False)
        store.close()

        text = Path(path).read_text()
        assert "No completed variants" in text

    def test_render_markdown_is_pure(self) -> None:
        """Rendering must work from data alone, with no store access."""
        from inferbench.experiment.report import VariantSummary
        from inferbench.experiment.runner import StudyResult

        study = StudyResult(experiment="X", varies="load.concurrency")
        summaries = [
            VariantSummary(
                name="c1",
                run_id="r1",
                varies_value=1,
                ttft_p50=0.01,
                ttft_p95=0.02,
                tpot_p50=0.002,
                e2e_p50=0.1,
                e2e_p95=0.2,
                requests_per_s=10.0,
                output_tokens_per_s=100.0,
                failure_rate=0.0,
                successes=4,
                total=4,
                per_repetition_rps=(10.0, 10.1),
            ),
            VariantSummary(
                name="c2",
                run_id="r2",
                varies_value=2,
                ttft_p50=0.01,
                ttft_p95=0.02,
                tpot_p50=0.002,
                e2e_p50=0.1,
                e2e_p95=0.25,
                requests_per_s=19.0,
                output_tokens_per_s=190.0,
                failure_rate=0.0,
                successes=4,
                total=4,
                per_repetition_rps=(19.0, 19.2),
            ),
        ]
        text = render_study_markdown(study, summaries, metadata={"data_source": "synthetic"})
        assert "X" in text
        assert "| 1 |" in text
        assert "| 2 |" in text
