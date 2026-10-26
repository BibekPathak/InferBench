"""Tests for experiment storage.

The storage layer's job is to preserve enough information to reproduce and
audit a result, so these tests check round-tripping fidelity — especially that
raw records can be read back and every metric recomputed from them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from inferbench.errors import RunNotFoundError
from inferbench.metrics import summarize
from inferbench.storage import ExperimentStore
from inferbench.types import RequestRecord, RequestStatus, RunMetadata


def record(request_id: str = "r0", **overrides: object) -> RequestRecord:
    base = {
        "request_id": request_id,
        "status": RequestStatus.success,
        "prompt_tokens": 32,
        "output_tokens": 4,
        "requested_output_tokens": 8,
        "ttft_s": 0.1,
        "e2e_s": 0.16,
        "token_times_s": (0.1, 0.12, 0.14, 0.16),
        "class_label": "fixed",
        "repetition": 0,
        "is_warmup": False,
        "started_at": 1234.5,
        "error": None,
        "finish_reason": "length",
        "backend_request_id": "mock-1",
    }
    base.update(overrides)
    return RequestRecord(**base)  # type: ignore[arg-type]


def metadata(run_id: str = "run-1") -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        run_name="test",
        experiment="baseline",
        created_at=1.0,
        started_at=2.0,
        finished_at=None,
        config={"workload": {"num_requests": 4}},
        environment={"gpu": {"available": False}},
        backend_kind="mock",
        backend_version=None,
        model="mock-model",
        data_source="synthetic",
        tool_version="0.1.0",
        notes="test run",
    )


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[ExperimentStore]:
    """An open store rooted at a temporary directory."""
    with ExperimentStore(tmp_path) as s:
        yield s


class TestLifecycle:
    def test_creates_root_and_database(self, tmp_path: Path) -> None:
        with ExperimentStore(tmp_path / "out") as store:
            assert (tmp_path / "out" / "inferbench.sqlite3").is_file()
            assert store.list_runs() == []

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        with ExperimentStore(tmp_path) as store:
            store.save_metadata(metadata(), {})
        # Reopening must work and see the run.
        with ExperimentStore(tmp_path) as store:
            assert store.has_run("run-1")


class TestMetadata:
    def test_round_trip(self, store: ExperimentStore) -> None:
        original = metadata()
        store.save_metadata(original, {"disk": {"free_gb": 1.0}})
        loaded = store.get_metadata("run-1")

        assert loaded.run_id == original.run_id
        assert loaded.backend_kind == "mock"
        assert loaded.data_source == "synthetic"
        assert loaded.notes == "test run"
        assert loaded.environment["gpu"]["available"] is False

    def test_writes_human_readable_artifacts(self, store: ExperimentStore) -> None:
        store.save_metadata(metadata(), {"cpu": 4})
        run_dir = store.run_dir("run-1")
        assert (run_dir / "metadata.json").is_file()
        assert (run_dir / "environment.json").is_file()
        env = json.loads((run_dir / "environment.json").read_text())
        assert env["cpu"] == 4

    def test_unknown_run_raises(self, store: ExperimentStore) -> None:
        with pytest.raises(RunNotFoundError):
            store.get_metadata("ghost")

    def test_metadata_update_marks_complete(self, store: ExperimentStore) -> None:
        meta = metadata()
        meta.incomplete = True
        meta.incomplete_reason = "in progress"
        store.save_metadata(meta, {})

        meta.incomplete = False
        meta.incomplete_reason = None
        meta.finished_at = 99.0
        store.save_metadata(meta, {})

        loaded = store.get_metadata("run-1")
        assert loaded.incomplete is False
        assert loaded.finished_at == 99.0


class TestRawRecords:
    def test_jsonl_round_trip(self, store: ExperimentStore) -> None:
        store.save_metadata(metadata(), {})
        records = [record("a"), record("b", status=RequestStatus.timeout, ttft_s=None)]
        store.save_raw_records("run-1", records)

        loaded = store.load_raw_records("run-1")
        assert len(loaded) == 2
        assert loaded[0].request_id == "a"
        assert loaded[0].status is RequestStatus.success
        assert loaded[1].status is RequestStatus.timeout
        assert loaded[1].ttft_s is None
        assert loaded[0].token_times_s == (0.1, 0.12, 0.14, 0.16)

    def test_raw_file_is_one_object_per_line(self, store: ExperimentStore) -> None:
        store.save_metadata(metadata(), {})
        store.save_raw_records("run-1", [record("a"), record("b")])
        path = store.run_dir("run-1") / "raw.jsonl"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["request_id"] == "a"

    def test_metrics_recomputable_from_raw(self, store: ExperimentStore) -> None:
        """The core auditability guarantee: raw data alone suffices."""
        store.save_metadata(metadata(), {})
        records = [record(f"r{i}") for i in range(5)]
        store.save_raw_records("run-1", records)

        reloaded = store.load_raw_records("run-1")
        summary = summarize(reloaded, wall_time_s=1.0)
        assert summary.throughput.successful_requests == 5
        assert summary.latency.ttft.p50 == pytest.approx(0.1)
        assert summary.latency.tpot.p50 == pytest.approx(0.02)

    def test_rewrite_replaces_previous(self, store: ExperimentStore) -> None:
        store.save_metadata(metadata(), {})
        store.save_raw_records("run-1", [record("a")])
        store.save_raw_records("run-1", [record("b"), record("c")])
        assert len(store.load_raw_records("run-1")) == 2

    def test_missing_raw_raises(self, store: ExperimentStore) -> None:
        store.save_metadata(metadata(), {})
        (store.run_dir("run-1")).mkdir(exist_ok=True)
        with pytest.raises(RunNotFoundError, match="Raw records missing"):
            store.load_raw_records("run-1")

    def test_unknown_run_raw_raises(self, store: ExperimentStore) -> None:
        with pytest.raises(RunNotFoundError):
            store.load_raw_records("nonexistent")


class TestSummary:
    def test_round_trip_and_exports(self, store: ExperimentStore) -> None:
        store.save_metadata(metadata(), {})
        store.save_raw_records("run-1", [record("a")])
        summary = summarize(store.load_raw_records("run-1"), wall_time_s=1.0).to_dict()
        store.save_summary("run-1", summary)

        loaded = store.get_summary("run-1")
        assert loaded["throughput"]["successful_requests"] == 1

        run_dir = store.run_dir("run-1")
        assert (run_dir / "metrics.json").is_file()
        csv_text = (run_dir / "summary.csv").read_text()
        assert csv_text.startswith("metric,value")
        assert "latency.ttft.p50" in csv_text

    def test_missing_summary_raises(self, store: ExperimentStore) -> None:
        store.save_metadata(metadata(), {})
        with pytest.raises(RunNotFoundError, match="no summary"):
            store.get_summary("run-1")


class TestListing:
    def test_lists_runs_newest_first(self, store: ExperimentStore) -> None:
        early = metadata("run-old")
        early.created_at = 1.0
        late = metadata("run-new")
        late.created_at = 2.0
        store.save_metadata(early, {})
        store.save_metadata(late, {})

        rows = store.list_runs()
        assert [row["run_id"] for row in rows] == ["run-new", "run-old"]

    def test_has_run(self, store: ExperimentStore) -> None:
        assert store.has_run("run-1") is False
        store.save_metadata(metadata(), {})
        assert store.has_run("run-1") is True
