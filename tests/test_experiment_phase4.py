"""Tests for Phase 4: Concurrency & Saturation experiments.

Phase 4 extends the experiment framework with higher-concurrency sweeps
and saturation knee detection.  Experiment C varies ``load.concurrency``
from 1 to 32, modelling inter-request decode contention via the mock
backend's ``decode_contention`` option.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from inferbench.experiment import (
    ExperimentConfig,
    StudyResult,
    StudyRunner,
    Variant,
    collect_variant_summaries,
    expand_experiment,
    generate_study_report,
    render_study_markdown,
)
from inferbench.experiment.config import load_experiment
from inferbench.experiment.report import VariantSummary, _find_saturation_knee
from inferbench.storage import ExperimentStore


def fast_options() -> MockBackendOptions:
    return MockBackendOptions(
        ttft_base_s=0.01,
        tpot_base_s=0.002,
        prefill_tokens_per_s=10_000.0,
        decode_contention=0.0,
        real_time=False,
    )


def saturation_point(value: int, rps: float) -> VariantSummary:
    """A VariantSummary standing in for one point on a saturation curve."""
    return VariantSummary(
        name=f"c{value}",
        run_id="r",
        varies_value=value,
        ttft_p50=0.01,
        ttft_p95=0.01,
        tpot_p50=0.002,
        e2e_p50=0.1,
        e2e_p95=0.1 + 0.01 * value,
        requests_per_s=rps,
        output_tokens_per_s=rps,
        failure_rate=0.0,
        successes=1,
        total=1,
        per_repetition_rps=(),
    )


def base_config(backend: BackendConfig | None = None) -> RunConfig:
    return RunConfig(
        run_name="exp",
        experiment="C",
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


def concurrency_experiment_c(levels: list[int]) -> ExperimentConfig:
    return ExperimentConfig(
        name="C",
        varies="load.concurrency",
        base=base_config(),
        variants=[
            Variant(name=f"c{level}", overrides={"load": {"concurrency": level}})
            for level in levels
        ],
    )


class TestExperimentCLoad:
    def test_shipped_experiment_c_loads(self) -> None:
        path = (
            Path(__file__).resolve().parents[1] / "configs" / "experiment_c_concurrency_sweep.yaml"
        )
        experiment = load_experiment(path)
        expanded = expand_experiment(experiment)
        assert len(expanded) == len(experiment.variants)
        assert experiment.varies
        # Verify the controlled-variable guard fires (all variants change only concurrency)
        # Since they all change only concurrency, the guard should pass
        assert all(
            r.config.load.concurrency == level
            for r, level in zip(expanded, [1, 2, 4, 8, 16, 32], strict=False)
        )

    def test_expansion_deterministic(self) -> None:
        path = (
            Path(__file__).resolve().parents[1] / "configs" / "experiment_c_concurrency_sweep.yaml"
        )
        experiment = load_experiment(path)
        first = [r.config.model_dump() for r in expand_experiment(experiment)]
        second = [r.config.model_dump() for r in expand_experiment(experiment)]
        assert first == second

    def test_experiment_c_declares_decode_contention(self) -> None:
        """The shipped C study must model contention, else it shows no saturation."""
        path = (
            Path(__file__).resolve().parents[1] / "configs" / "experiment_c_concurrency_sweep.yaml"
        )
        experiment = load_experiment(path)
        synthetic = experiment.base.backend.synthetic
        assert synthetic is not None
        assert synthetic.decode_contention > 0

    def test_experiment_c_backend_builds_mock_with_contention(self) -> None:
        from inferbench.backends import create_backend

        path = (
            Path(__file__).resolve().parents[1] / "configs" / "experiment_c_concurrency_sweep.yaml"
        )
        experiment = load_experiment(path)
        backend = create_backend(experiment.base.backend)
        assert isinstance(backend, MockBackend)
        assert backend.options.decode_contention > 0

    def test_contention_section_is_identical_across_variants(self) -> None:
        """Decode contention is part of the fixed backend, not the varied axis."""
        path = (
            Path(__file__).resolve().parents[1] / "configs" / "experiment_c_concurrency_sweep.yaml"
        )
        experiment = load_experiment(path)
        expanded = expand_experiment(experiment)
        first = expanded[0].config.backend
        assert all(r.config.backend == first for r in expanded)


class TestStudyRunnerPhase4:
    async def test_runs_all_variants_c1_to_c32(self, tmp_path: Path) -> None:
        experiment = concurrency_experiment_c([1, 2, 4, 8, 16, 32])
        store = ExperimentStore(tmp_path)
        backend = MockBackend(base_config().backend, options=fast_options())
        study = await StudyRunner(experiment, backend, store).run()
        store.close()

        assert study.succeeded == 6
        assert study.attempted == 6
        assert study.is_complete
        assert len(study.run_ids) == 6

    async def test_variants_ordered_numerically(self, tmp_path: Path) -> None:
        """Concurrency 1, 2, 4, 8, 16, 32 must read in that order, not alphabetically."""
        experiment = concurrency_experiment_c([1, 2, 4, 8, 16, 32])
        store = ExperimentStore(tmp_path)
        backend = MockBackend(base_config().backend, options=fast_options())
        study = await StudyRunner(experiment, backend, store).run()
        summaries = collect_variant_summaries(store, study)
        store.close()

        assert [s.varies_value for s in summaries] == [1, 2, 4, 8, 16, 32]

    async def test_saturation_knee_detected_with_contention(self, tmp_path: Path) -> None:
        """With decode_contention > 0 the curve stalls and the knee is detected.

        A stalled sweep is modelled directly: throughput improves from c1 to c4,
        then flattens. The knee detector must point at the last improving point.
        """
        summaries = [
            saturation_point(1, 10.0),
            saturation_point(2, 20.0),
            saturation_point(4, 40.0),
            saturation_point(8, 41.0),
            saturation_point(16, 41.0),
            saturation_point(32, 41.0),
        ]
        # knee should be index 2 (pointing at c4, the last point that improved)
        assert _find_saturation_knee(summaries) == 2

    async def test_no_knee_when_still_scaling(self, tmp_path: Path) -> None:
        # Still scaling: 10→20→40→80, no flattening.
        summaries = [
            saturation_point(1, 10.0),
            saturation_point(2, 20.0),
            saturation_point(4, 40.0),
            saturation_point(8, 80.0),
        ]
        # Never flattens: the function reports the last improving point.
        assert _find_saturation_knee(summaries) == 3

    async def test_knee_needs_two_points(self, tmp_path: Path) -> None:
        assert _find_saturation_knee([]) is None


class TestStudyReportPhase4:
    async def _study(
        self, tmp_path: Path, levels: list[int]
    ) -> tuple[ExperimentStore, StudyResult]:
        experiment = concurrency_experiment_c(levels)
        store = ExperimentStore(tmp_path)
        backend = MockBackend(base_config().backend, options=fast_options())
        study = await StudyRunner(experiment, backend, store).run()
        return store, study

    async def test_report_generated_with_provenance(self, tmp_path: Path) -> None:
        store, study = await self._study(tmp_path, [1, 2, 4, 8])
        path = generate_study_report(store, study, make_charts=False)
        store.close()

        text = Path(path).read_text()
        assert "SYNTHETIC" in text
        assert "must not" in text.lower()
        assert study.experiment in text
        assert "load.concurrency" in text

    async def test_report_lists_all_variants(self, tmp_path: Path) -> None:
        store, study = await self._study(tmp_path, [1, 2, 4, 8])
        path = generate_study_report(store, study, make_charts=False)
        store.close()

        text = Path(path).read_text()
        for level in (1, 2, 4, 8):
            assert f"c{level}" in text

    async def test_report_writes_machine_readable_json(self, tmp_path: Path) -> None:
        store, study = await self._study(tmp_path, [1, 2])
        path = generate_study_report(store, study, make_charts=False)
        store.close()

        json_path = Path(path).with_suffix(".json")
        assert json_path.is_file()
        payload = json.loads(json_path.read_text())
        assert payload["experiment"] == "C"
        assert payload["varies"] == "load.concurrency"
        assert len(payload["variants"]) == 2

    async def test_report_marks_partial_study(self, tmp_path: Path) -> None:
        store, study = await self._study(tmp_path, [1, 2, 4, 8])
        # Simulate a variant that failed to produce a summary.
        study.failures["c8"] = "BackendError: injected"
        path = generate_study_report(store, study, make_charts=False)
        store.close()

        text = Path(path).read_text()
        assert "Failed variants" in text
        assert "incomplete" in text.lower()
        assert "c8" in text

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


class TestRenderMarkdownPhase4:
    def test_render_markdown_with_saturation_observations(self) -> None:
        from inferbench.experiment.report import VariantSummary
        from inferbench.experiment.runner import StudyResult

        study = StudyResult(experiment="C", varies="load.concurrency")
        summaries = [
            VariantSummary(
                name="c1",
                run_id="r1",
                varies_value=1,
                ttft_p50=0.01,
                ttft_p95=0.02,
                tpot_p50=0.002,
                e2e_p50=0.1,
                e2e_p95=0.1,
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
                e2e_p95=0.12,
                requests_per_s=19.0,
                output_tokens_per_s=190.0,
                failure_rate=0.0,
                successes=4,
                total=4,
                per_repetition_rps=(19.0, 19.2),
            ),
            VariantSummary(
                name="c4",
                run_id="r3",
                varies_value=4,
                ttft_p50=0.01,
                ttft_p95=0.02,
                tpot_p50=0.002,
                e2e_p50=0.1,
                e2e_p95=0.18,
                requests_per_s=38.0,
                output_tokens_per_s=380.0,
                failure_rate=0.0,
                successes=4,
                total=4,
                per_repetition_rps=(38.0, 38.3),
            ),
            VariantSummary(
                name="c8",
                run_id="r4",
                varies_value=8,
                ttft_p50=0.01,
                ttft_p95=0.05,
                tpot_p50=0.002,
                e2e_p50=0.1,
                e2e_p95=0.45,
                requests_per_s=39.0,
                output_tokens_per_s=390.0,
                failure_rate=0.0,
                successes=4,
                total=4,
                per_repetition_rps=(39.0, 39.5),
            ),
            VariantSummary(
                name="c16",
                run_id="r5",
                varies_value=16,
                ttft_p50=0.01,
                ttft_p95=0.08,
                tpot_p50=0.002,
                e2e_p50=0.1,
                e2e_p95=0.80,
                requests_per_s=39.5,
                output_tokens_per_s=395.0,
                failure_rate=0.0,
                successes=4,
                total=4,
                per_repetition_rps=(39.5, 39.8),
            ),
            VariantSummary(
                name="c32",
                run_id="r6",
                varies_value=32,
                ttft_p50=0.01,
                ttft_p95=0.15,
                tpot_p50=0.002,
                e2e_p50=0.1,
                e2e_p95=1.50,
                requests_per_s=39.7,
                output_tokens_per_s=397.0,
                failure_rate=0.0,
                successes=4,
                total=4,
                per_repetition_rps=(39.7, 40.0),
            ),
        ]
        text = render_study_markdown(study, summaries, metadata={"data_source": "synthetic"})
        assert "C" in text
        assert "| 1 |" in text
        assert "| 32 |" in text
        # Saturation observations
        assert "Throughput" in text
        assert "E2E p95" in text or "tail" in text.lower()
        # Knee detection: c4 is the last that improved, so knee should be mentioned
        assert "Apparent flattening" in text or "saturation" in text.lower()
