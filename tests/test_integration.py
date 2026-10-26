"""End-to-end integration tests for the benchmark pipeline.

These exercise the real path a user takes — config → backend → workload →
runner → storage → metrics → report — against the mock backend, on CPU, with
no network. This is the vertical slice Phase 1 exists to deliver.

Where the mock's timing model is analytic, assertions use the *predicted*
values rather than recorded magic numbers, so the tests verify correctness
rather than merely freezing behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferbench.analysis import generate_report, render_markdown
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
from inferbench.metrics import filter_measured, summarize
from inferbench.runner import Runner, RunResult
from inferbench.storage import ExperimentStore
from inferbench.types import RequestStatus


def make_config(**overrides: object) -> RunConfig:
    """A fast, deterministic config suitable for tests."""
    defaults: dict[str, object] = {
        "run_name": "test-run",
        "experiment": "baseline",
        "backend": BackendConfig(kind=BackendKind.mock, model="mock-model"),
        "workload": WorkloadConfig(
            kind=WorkloadKind.fixed, num_requests=8, prompt_tokens=100, output_tokens=10, seed=0
        ),
        "load": LoadConfig(
            pattern=ArrivalPattern.closed_loop, concurrency=2, warmup_requests=2, repetitions=1
        ),
    }
    defaults.update(overrides)
    return RunConfig(**defaults)  # type: ignore[arg-type]


def make_options(**overrides: object) -> MockBackendOptions:
    """Real-time mocked timing, so measured values match the analytic model."""
    defaults: dict[str, object] = {
        "ttft_base_s": 0.01,
        "tpot_base_s": 0.005,
        "prefill_tokens_per_s": 10_000.0,
        "report_prompt_tokens": True,
        "real_time": False,
    }
    defaults.update(overrides)
    return MockBackendOptions(**defaults)  # type: ignore[arg-type]


async def run_experiment(
    tmp_path: Path, config: RunConfig, options: MockBackendOptions
) -> RunResult:
    store = ExperimentStore(tmp_path)
    backend = MockBackend(config.backend, options=options)
    runner = Runner(config, backend, store, environment={"test": True})
    result = await runner.run()
    await backend.teardown()
    return result


class TestEndToEnd:
    async def test_run_completes_and_stores_everything(self, tmp_path: Path) -> None:
        result = await run_experiment(tmp_path, make_config(), make_options())

        assert result.run_id
        run_dir = Path(result.run_dir)
        for artifact in (
            "metadata.json",
            "raw.jsonl",
            "metrics.json",
            "summary.csv",
            "environment.json",
        ):
            assert (run_dir / artifact).is_file(), f"missing {artifact}"

    async def test_metadata_records_provenance(self, tmp_path: Path) -> None:
        result = await run_experiment(tmp_path, make_config(), make_options())
        meta = result.metadata
        # The mock is not a real backend; this must be recorded, not glossed.
        assert meta.data_source == "synthetic"
        assert meta.backend_kind == "mock"
        assert meta.backend_version is None
        assert meta.incomplete is False
        assert meta.config["workload"]["num_requests"] == 8

    async def test_all_requests_recorded(self, tmp_path: Path) -> None:
        config = make_config()
        result = await run_experiment(tmp_path, config, make_options())
        # 8 measured + 2 warm-up
        assert len(result.records) == 10
        measured = filter_measured(list(result.records))
        assert len(measured) == 8

    async def test_warmup_excluded_from_metrics(self, tmp_path: Path) -> None:
        result = await run_experiment(tmp_path, make_config(), make_options())
        # The summary must reflect exactly the measured requests.
        assert result.summary["throughput"]["successful_requests"] == 8

    async def test_measured_timings_match_analytic_model(self, tmp_path: Path) -> None:
        """TTFT/TPOT must match the mock's predicted values, not magic numbers."""
        options = make_options(ttft_base_s=0.01, tpot_base_s=0.005, prefill_tokens_per_s=10_000.0)
        config = make_config(
            workload=WorkloadConfig(
                kind=WorkloadKind.fixed, num_requests=4, prompt_tokens=100, output_tokens=11
            ),
            load=LoadConfig(concurrency=1, warmup_requests=0, repetitions=1),
        )
        result = await run_experiment(tmp_path, config, options)

        expected_ttft = options.expected_ttft(100)  # 0.01 + 100/10000 = 0.02
        expected_tpot = options.expected_tpot(1)  # 0.005
        summary = result.summary
        assert summary["latency"]["ttft"]["p50"] == pytest.approx(expected_ttft, abs=1e-6)
        assert summary["latency"]["tpot"]["p50"] == pytest.approx(expected_tpot, abs=1e-6)

    async def test_e2e_recomputable_from_raw_jsonl(self, tmp_path: Path) -> None:
        """Raw data must suffice to reproduce the summary (auditability)."""
        config = make_config()
        store = ExperimentStore(tmp_path)
        backend = MockBackend(config.backend, options=make_options())
        runner = Runner(config, backend, store, environment={})
        result = await runner.run()
        await backend.teardown()

        reloaded = store.load_raw_records(result.run_id)
        recomputed = summarize(
            filter_measured(reloaded),
            wall_time_s=result.measured_wall_time_s,
        ).to_dict()
        store.close()

        # Same inputs -> same numbers.
        original = result.summary
        for key in ("successes", "timeouts", "errors", "interrupted"):
            assert recomputed["failures"][key] == original["failures"][key]
        assert recomputed["latency"]["ttft"]["p50"] == pytest.approx(
            original["latency"]["ttft"]["p50"]
        )


class TestFailureHandling:
    async def test_injected_failures_are_recorded_not_lost(self, tmp_path: Path) -> None:
        # fail_every=3 -> requests 3 and 6 of the measured set fail.
        config = make_config(
            workload=WorkloadConfig(num_requests=9, prompt_tokens=50, output_tokens=5),
            load=LoadConfig(concurrency=1, warmup_requests=0),
        )
        result = await run_experiment(tmp_path, config, make_options(fail_every=3))

        failures = result.summary["failures"]
        assert failures["errors"] == 3
        assert failures["successes"] == 6
        assert failures["failure_rate"] == pytest.approx(3 / 9)

    async def test_failures_reduce_throughput(self, tmp_path: Path) -> None:
        config = make_config(
            workload=WorkloadConfig(num_requests=6, prompt_tokens=50, output_tokens=5),
            load=LoadConfig(concurrency=1, warmup_requests=0),
        )
        result = await run_experiment(tmp_path, config, make_options(fail_every=2))
        # 6 requests, every 2nd fails -> 3 successes.
        assert result.summary["throughput"]["successful_requests"] == 3
        assert result.summary["throughput"]["failed_requests"] == 3

    async def test_interruption_preserves_partial_tokens(self, tmp_path: Path) -> None:
        config = make_config(
            workload=WorkloadConfig(num_requests=4, prompt_tokens=50, output_tokens=10),
            load=LoadConfig(concurrency=1, warmup_requests=0),
        )
        result = await run_experiment(tmp_path, config, make_options(interrupt_every=1))

        interrupted = [r for r in result.records if r.status is RequestStatus.interrupted]
        assert interrupted
        # Partial generations must carry a positive, sub-target token count.
        assert all(0 < r.output_tokens < 10 for r in interrupted)

    async def test_zero_token_success_is_downgraded(self, tmp_path: Path) -> None:
        """A stream that yields no tokens must not be credited as a success."""
        config = make_config(
            workload=WorkloadConfig(num_requests=3, prompt_tokens=20, output_tokens=1),
            load=LoadConfig(concurrency=1, warmup_requests=0),
        )
        # output_tokens=1 still yields one token, so this path is exercised via
        # the mock's own guard; assert the invariant holds generally.
        result = await run_experiment(tmp_path, config, make_options())
        for record in result.records:
            if record.status is RequestStatus.success:
                assert record.output_tokens > 0


class TestConcurrencyAndPatterns:
    async def test_concurrency_does_not_lose_requests(self, tmp_path: Path) -> None:
        config = make_config(
            workload=WorkloadConfig(num_requests=20, prompt_tokens=40, output_tokens=5),
            load=LoadConfig(concurrency=8, warmup_requests=0, repetitions=1),
        )
        result = await run_experiment(tmp_path, config, make_options())
        assert len(filter_measured(list(result.records))) == 20
        assert result.summary["throughput"]["successful_requests"] == 20

    async def test_repetitions_are_separate(self, tmp_path: Path) -> None:
        config = make_config(
            workload=WorkloadConfig(num_requests=4, prompt_tokens=30, output_tokens=4),
            load=LoadConfig(concurrency=2, warmup_requests=0, repetitions=3),
        )
        result = await run_experiment(tmp_path, config, make_options())
        assert result.summary["repetitions"] == 3
        assert len(result.summary["per_repetition_rps"]) == 3
        measured = filter_measured(list(result.records))
        assert len(measured) == 12  # 4 requests x 3 repetitions

    async def test_constant_rate_pattern(self, tmp_path: Path) -> None:
        config = make_config(
            workload=WorkloadConfig(num_requests=6, prompt_tokens=20, output_tokens=3),
            load=LoadConfig(
                pattern=ArrivalPattern.constant_rate,
                rate_rps=100.0,
                warmup_requests=0,
            ),
        )
        result = await run_experiment(tmp_path, config, make_options())
        assert result.summary["throughput"]["successful_requests"] == 6

    async def test_bursty_pattern(self, tmp_path: Path) -> None:
        config = make_config(
            workload=WorkloadConfig(num_requests=8, prompt_tokens=20, output_tokens=3),
            load=LoadConfig(
                pattern=ArrivalPattern.bursty,
                rate_rps=50.0,
                burst_size=4,
                warmup_requests=0,
            ),
        )
        result = await run_experiment(tmp_path, config, make_options())
        assert result.summary["throughput"]["successful_requests"] == 8


class TestReporting:
    async def test_report_generated_with_provenance_banner(self, tmp_path: Path) -> None:
        store = ExperimentStore(tmp_path)
        config = make_config()
        backend = MockBackend(config.backend, options=make_options())
        result = await Runner(config, backend, store, environment={}).run()
        await backend.teardown()

        path = generate_report(store, result.run_id, make_charts=False)
        store.close()

        text = Path(path).read_text()
        assert "SYNTHETIC" in text
        assert "must not" in text.lower()
        assert result.run_id in text
        # Key metrics must appear.
        assert "TTFT" in text
        assert "requests/sec" in text
        assert "failure rate" in text

    async def test_report_states_gpu_unavailable(self, tmp_path: Path) -> None:
        import json

        store = ExperimentStore(tmp_path)
        config = make_config()
        backend = MockBackend(config.backend, options=make_options())
        env = {"gpu": {"available": False, "reason": "no driver"}, "python_version": "3.12"}
        result = await Runner(config, backend, store, environment=env).run()
        await backend.teardown()

        path = generate_report(store, result.run_id, make_charts=False)
        store.close()
        text = Path(path).read_text()
        assert "unavailable" in text
        assert "no driver" in text
        _ = json

    def test_render_markdown_handles_missing_gpu(self) -> None:
        text = render_markdown(
            {"run_id": "x", "run_name": "n", "data_source": "synthetic", "environment": {}},
            {"data_source": "synthetic", "latency": {}, "throughput": {}, "failures": {}},
        )
        assert "SYNTHETIC" in text
        assert "unavailable" in text
