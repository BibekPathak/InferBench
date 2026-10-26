"""Shared pytest fixtures for the InferBench test suite.

The ``stored_run`` helper is used by the validation and comparison tests: it
writes a complete, internally consistent run into a store so tests can then
corrupt a specific aspect and assert that validation catches it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from inferbench.metrics import filter_measured, summarize
from inferbench.storage import ExperimentStore
from inferbench.types import RequestRecord, RequestStatus, RunMetadata

FIXTURES_DIR = Path(__file__).parent / "fixtures"

#: Signature of the ``stored_run`` fixture factory.
StoredRunFactory = Callable[..., str]


@pytest.fixture()
def fixtures_dir() -> Path:
    """Directory holding static test fixtures."""
    return FIXTURES_DIR


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Callable[..., Path]:
    """Factory writing a config file with given content and returning its path."""

    def _write(content: str, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    return _write


def _record(
    request_id: str,
    *,
    status: RequestStatus = RequestStatus.success,
    ttft_s: float | None = 0.05,
    output_tokens: int = 5,
    repetition: int = 0,
) -> RequestRecord:
    """A record with self-consistent analytic timing."""
    if status.is_success and ttft_s is not None:
        token_times = tuple(ttft_s + i * 0.01 for i in range(output_tokens))
    else:
        token_times = ()
    return RequestRecord(
        request_id=request_id,
        status=status,
        prompt_tokens=32,
        output_tokens=output_tokens if token_times else 0,
        requested_output_tokens=output_tokens,
        ttft_s=ttft_s if token_times else None,
        e2e_s=token_times[-1] if token_times else 0.05,
        token_times_s=token_times,
        class_label="fixed",
        repetition=repetition,
        is_warmup=False,
        started_at=1000.0 + repetition,
        error=None if status.is_success else "injected",
        finish_reason="length" if status.is_success else None,
    )


def build_records(
    *,
    count: int = 4,
    repetitions: int = 1,
    statuses: dict[int, RequestStatus] | None = None,
) -> list[RequestRecord]:
    """Build a consistent set of measured records."""
    statuses = statuses or {}
    records: list[RequestRecord] = []
    for repetition in range(repetitions):
        for index in range(count):
            records.append(
                _record(
                    f"r{repetition}-{index}",
                    status=statuses.get(index, RequestStatus.success),
                    repetition=repetition,
                )
            )
    return records


def build_metadata(
    run_id: str,
    *,
    workload: dict[str, Any] | None = None,
    load: dict[str, Any] | None = None,
    model: str = "mock-model",
    data_source: str = "synthetic",
    incomplete: bool = False,
) -> RunMetadata:
    """Build run metadata with a complete, comparable default shape."""
    return RunMetadata(
        run_id=run_id,
        run_name=run_id,
        experiment="test",
        created_at=1000.0,
        started_at=1001.0,
        finished_at=1002.0,
        config={
            "workload": workload
            or {"kind": "fixed", "num_requests": 4, "prompt_tokens": 128, "output_tokens": 32},
            "load": load
            or {"pattern": "closed_loop", "concurrency": 1, "warmup_requests": 0, "repetitions": 1},
        },
        environment={"gpu": {"available": False, "reason": "no driver"}, "python_version": "3.12"},
        backend_kind="mock",
        backend_version=None,
        model=model,
        data_source=data_source,
        tool_version="0.1.0",
        incomplete=incomplete,
    )


@pytest.fixture()
def make_records() -> Callable[..., list[RequestRecord]]:
    """Expose :func:`build_records` to tests without cross-module imports."""
    return build_records


@pytest.fixture()
def make_metadata() -> Callable[..., RunMetadata]:
    """Expose :func:`build_metadata` to tests without cross-module imports."""
    return build_metadata


@pytest.fixture()
def stored_run() -> Callable[..., str]:
    """Factory that stores a complete run and returns its id.

    Usage::

        run_id = stored_run(store, records=build_records(), metadata=build_metadata("r1"))

    Any component can be omitted to get a consistent default.
    """

    def _store(
        store: ExperimentStore,
        *,
        records: list[RequestRecord] | None = None,
        metadata: RunMetadata | None = None,
        run_id: str | None = None,
        wall_time_s: float = 1.0,
        write_summary: bool = True,
    ) -> str:
        # Auto-number the default id so repeated calls in one store do not
        # collide, while explicit ids stay honored.
        if run_id is None:
            counter = _store.counter  # type: ignore[attr-defined]
            _store.counter = counter + 1  # type: ignore[attr-defined]
            run_id = f"run-{counter}"
        records = records if records is not None else build_records()
        metadata = metadata if metadata is not None else build_metadata(run_id)
        store.save_metadata(metadata, metadata.environment)
        store.save_raw_records(run_id, records)
        if write_summary:
            measured = filter_measured(records)
            summary = summarize(measured, wall_time_s=wall_time_s).to_dict()
            summary["run_id"] = run_id
            summary["data_source"] = metadata.data_source
            store.save_summary(run_id, summary)
        return run_id

    _store.counter = 1  # type: ignore[attr-defined]
    return _store
