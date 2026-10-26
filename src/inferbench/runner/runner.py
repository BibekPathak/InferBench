"""Experiment orchestration.

The runner is where benchmark *correctness* is enforced:

- **Warm-up is separated from measurement.** Warm-up requests are executed and
  recorded (marked ``is_warmup``) but excluded from every measured statistic.
- **Every dispatched request produces exactly one record.** Timeouts,
  cancellations, disconnects, and server errors all terminate in a recorded
  status rather than vanishing.
- **The measurement window is timed explicitly.** Throughput divides completed
  work by the observed window, so a slow start cannot inflate req/s.
- **Repetitions are independent.** Each repetition is timed separately so
  variability across repeats can be reported.

Concurrency model: a closed-loop pool of ``concurrency`` workers for the
closed-loop pattern; a rate-limited dispatcher for constant-rate and bursty
patterns. Both share the same per-request execution path, so metrics are
comparable across patterns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from inferbench.backends.base import (
    BackendAdapter,
    CompletionSignal,
    FailureSignal,
    TokenChunk,
)
from inferbench.config import ArrivalPattern, RunConfig
from inferbench.errors import BackendError
from inferbench.logging_utils import get_logger, log_event
from inferbench.metrics import filter_measured, summarize
from inferbench.storage import ExperimentStore
from inferbench.types import (
    RequestRecord,
    RequestSpec,
    RequestStatus,
    RunMetadata,
    new_run_id,
)
from inferbench.version_info import tool_version

__all__ = ["RunResult", "Runner"]

_log = get_logger("runner")


@dataclass(slots=True)
class RunResult:
    """Everything a completed run produced, in memory and on disk."""

    run_id: str
    metadata: RunMetadata
    records: tuple[RequestRecord, ...]
    summary: dict[str, Any]
    run_dir: str
    #: Wall-clock duration of the measured window(s), excluding warm-up.
    measured_wall_time_s: float
    #: Non-fatal problems encountered (e.g. probe failures). Surfaced in reports.
    warnings: list[str] = field(default_factory=list)


class Runner:
    """Executes a :class:`RunConfig` against a backend and stores the results."""

    def __init__(
        self,
        config: RunConfig,
        backend: BackendAdapter,
        store: ExperimentStore,
        *,
        environment: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.store = store
        self.environment = environment or {}
        self._warnings: list[str] = []
        # Records produced by tasks cancelled mid-stream. Drained by the
        # executor so a shutdown still accounts for partially completed work.
        self._pending_records: list[RequestRecord] = []

    # --- public API -------------------------------------------------------

    async def run(self) -> RunResult:
        """Execute the configured experiment end to end.

        Raises:
            BackendError: The backend could not be set up (the run must not
                start). Per-request failures are recorded, not raised.
        """
        run_id = new_run_id(self.config.run_name)
        created_at = time.time()
        log_event(_log, logging.INFO, "run starting", run_id=run_id, run_name=self.config.run_name)

        await self.backend.setup()

        metadata = RunMetadata(
            run_id=run_id,
            run_name=self.config.run_name,
            experiment=self.config.experiment,
            created_at=created_at,
            started_at=time.time(),
            finished_at=None,
            config=self.config.model_dump(mode="json"),
            environment=self.environment,
            backend_kind=self.backend.name,
            backend_version=self.backend.version,
            model=self.config.backend.model,
            data_source=self._data_source(),
            tool_version=tool_version(),
            notes=self.config.notes,
            backend_details={},
            incomplete=True,
            incomplete_reason="run in progress",
        )

        try:
            details = await self.backend.describe()
            metadata.backend_details = details
            revision = details.get("model_revision")
            if isinstance(revision, str):
                metadata.model_revision = revision
        except BackendError as exc:
            # Describing the backend is best-effort; its absence is recorded
            # rather than fatal.
            self._warnings.append(f"backend.describe() failed: {exc}")

        self.store.save_metadata(metadata, self.environment)
        log_event(_log, logging.INFO, "run registered", run_id=run_id)

        records: list[RequestRecord] = []
        repetition_durations: dict[int, float] = {}

        try:
            records.extend(await self._run_warmup(run_id))
            measured, repetition_durations = await self._run_measured(run_id)
            records.extend(measured)
        except asyncio.CancelledError:
            metadata.incomplete = True
            metadata.incomplete_reason = "run cancelled"
            metadata.finished_at = time.time()
            self.store.save_metadata(metadata, self.environment)
            if records:
                self.store.save_raw_records(run_id, records)
            raise
        except Exception as exc:
            metadata.incomplete = True
            metadata.incomplete_reason = f"{type(exc).__name__}: {exc}"
            metadata.finished_at = time.time()
            self.store.save_metadata(metadata, self.environment)
            if records:
                self.store.save_raw_records(run_id, records)
            raise

        measured_records = filter_measured(records)
        wall_time = sum(repetition_durations.values())
        if wall_time <= 0:
            self._warnings.append("measured wall time was zero; throughput is not meaningful")

        summary_obj = summarize(
            measured_records,
            wall_time_s=max(wall_time, 1e-9),
            measured_requests=len(measured_records),
            repetition_durations_s=repetition_durations,
        )
        summary = summary_obj.to_dict()
        summary["warnings"] = list(self._warnings)
        summary["data_source"] = metadata.data_source
        summary["run_id"] = run_id
        summary["run_name"] = metadata.run_name
        summary["model"] = metadata.model
        summary["backend_kind"] = metadata.backend_kind
        summary["wall_time_s_total"] = round(wall_time, 6)

        self.store.save_raw_records(run_id, records)
        self.store.save_summary(run_id, summary)

        metadata.incomplete = False
        metadata.incomplete_reason = None
        metadata.finished_at = time.time()
        self.store.save_metadata(metadata, self.environment)

        log_event(
            _log,
            20,
            "run complete",
            run_id=run_id,
            requests=len(measured_records),
            failures=summary["failures"]["total"] - summary["failures"]["successes"],
        )

        return RunResult(
            run_id=run_id,
            metadata=metadata,
            records=tuple(records),
            summary=summary,
            run_dir=str(self.store.run_dir(run_id)),
            measured_wall_time_s=wall_time,
            warnings=list(self._warnings),
        )

    # --- phases -----------------------------------------------------------

    async def _run_warmup(self, run_id: str) -> list[RequestRecord]:
        """Issue and record warm-up requests (excluded from measurement)."""
        count = self.config.load.warmup_requests
        if count <= 0:
            return []

        # Warm-up reuses the first workload specs so the backend sees realistic
        # prompt shapes, but at lower concurrency to reach steady state quickly.
        specs = self._warmup_specs(count)
        log_event(_log, logging.INFO, "warmup starting", run_id=run_id, requests=len(specs))

        concurrency = max(1, min(self.config.load.concurrency, len(specs) or 1))
        results = await self._execute_pool(specs, concurrency)
        log_event(_log, logging.INFO, "warmup complete", run_id=run_id)
        return results

    async def _run_measured(self, run_id: str) -> tuple[list[RequestRecord], dict[int, float]]:
        """Execute all repetitions, timing each measured window separately."""
        from inferbench.workload import generate_workload

        plan = generate_workload(self.config.workload)
        base_specs = list(plan.specs)
        repetitions = self.config.load.repetitions
        all_records: list[RequestRecord] = []
        durations: dict[int, float] = {}

        log_event(
            _log,
            20,
            "measurement starting",
            run_id=run_id,
            repetitions=repetitions,
            requests_per_rep=len(base_specs),
            pattern=self.config.load.pattern.value,
        )

        for repetition in range(repetitions):
            specs = [
                RequestSpec(
                    request_id=f"r{repetition}-{spec.request_id}",
                    prompt=spec.prompt,
                    target_prompt_tokens=spec.target_prompt_tokens,
                    max_output_tokens=spec.max_output_tokens,
                    class_label=spec.class_label,
                    repetition=repetition,
                    is_warmup=False,
                )
                for spec in base_specs
            ]

            start = time.perf_counter()
            if self.config.load.pattern is ArrivalPattern.closed_loop:
                records = await self._execute_pool(specs, self.config.load.concurrency)
            else:
                records = await self._execute_rate_limited(specs)
            durations[repetition] = time.perf_counter() - start
            all_records.extend(records)

        return all_records, durations

    def _warmup_specs(self, count: int) -> list[RequestSpec]:
        from inferbench.workload import generate_workload

        plan = generate_workload(self.config.workload)
        base = list(plan.specs)
        # Cycle through the workload so warm-up covers the same shape mix.
        specs: list[RequestSpec] = []
        for index in range(count):
            source = base[index % len(base)]
            specs.append(
                RequestSpec(
                    request_id=f"warmup-{index:05d}",
                    prompt=source.prompt,
                    target_prompt_tokens=source.target_prompt_tokens,
                    max_output_tokens=source.max_output_tokens,
                    class_label=source.class_label,
                    repetition=-1,
                    is_warmup=True,
                )
            )
        return specs

    # --- execution strategies --------------------------------------------

    async def _execute_pool(
        self, specs: list[RequestSpec], concurrency: int
    ) -> list[RequestRecord]:
        """Closed-loop execution: ``concurrency`` workers pull from a queue."""
        if not specs:
            return []

        queue: asyncio.Queue[RequestSpec] = asyncio.Queue()
        for spec in specs:
            queue.put_nowait(spec)

        records: list[RequestRecord] = []
        lock = asyncio.Lock()

        async def worker() -> None:
            while True:
                try:
                    spec = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = await self._execute_one(spec)
                finally:
                    queue.task_done()
                async with lock:
                    records.append(record)

        workers = [asyncio.create_task(worker()) for _ in range(max(1, concurrency))]
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

        # Restore a deterministic order so storage output is stable.
        records.sort(key=lambda r: r.request_id)
        return records

    async def _execute_rate_limited(self, specs: list[RequestSpec]) -> list[RequestRecord]:
        """Open-loop execution: release requests on a constant/bursty schedule."""
        rate = self.config.load.rate_rps
        if rate is None or rate <= 0:
            # Config validation prevents this, but never hang if it happens.
            self._warnings.append("rate_rps unset for open-loop pattern; falling back to pool")
            return await self._execute_pool(specs, self.config.load.concurrency)

        start = time.perf_counter()
        tasks: list[asyncio.Task[RequestRecord]] = []
        burst_size = self.config.load.burst_size

        for index, spec in enumerate(specs):
            if self.config.load.pattern is ArrivalPattern.bursty:
                # Release bursts at the burst's own rate; the target rate
                # controls inter-burst spacing.
                within_burst = index % burst_size
                burst_index = index // burst_size
                scheduled = burst_index / rate + within_burst * (1.0 / (rate * burst_size))
            else:
                scheduled = index / rate

            delay = scheduled - (time.perf_counter() - start)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(self._execute_one(spec)))

        if not tasks:
            return []

        try:
            results = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        records = list(results)
        records.sort(key=lambda r: r.request_id)
        return records

    # --- single request ---------------------------------------------------

    async def _execute_one(self, spec: RequestSpec) -> RequestRecord:
        """Execute one request and convert its stream into a raw record.

        Guarantees exactly one :class:`RequestRecord` is returned for every
        call, whatever happens on the stream.
        """
        started_at = time.time()
        token_times: list[float] = []
        ttft: float | None = None
        output_tokens = 0
        prompt_tokens: int | None = None
        finish_reason: str | None = None
        backend_request_id: str | None = None
        status = RequestStatus.error
        error: str | None = None
        last_t = 0.0

        try:
            stream = self.backend.stream(spec)
            async for event in stream:
                if isinstance(event, TokenChunk):
                    output_tokens += 1
                    if ttft is None:
                        ttft = event.t
                    token_times.append(event.t)
                    last_t = event.t
                elif isinstance(event, CompletionSignal):
                    output_tokens = event.output_tokens
                    prompt_tokens = event.prompt_tokens
                    finish_reason = event.finish_reason
                    backend_request_id = event.backend_request_id
                    last_t = event.t
                    status = RequestStatus.success
                elif isinstance(event, FailureSignal):
                    output_tokens = event.output_tokens or output_tokens
                    backend_request_id = event.backend_request_id
                    last_t = event.t
                    status = _status_from_signal(event.status)
                    error = event.message
        except asyncio.CancelledError:
            # Cancellation is a recorded outcome for the record, but must still
            # propagate so the caller can shut down.
            status = RequestStatus.cancelled
            error = "request cancelled"
            e2e = max(last_t, 0.0)
            record = self._build_record(
                spec,
                started_at=started_at,
                status=status,
                ttft=ttft,
                e2e=e2e,
                output_tokens=output_tokens,
                prompt_tokens=prompt_tokens,
                token_times=tuple(token_times),
                finish_reason=finish_reason,
                backend_request_id=backend_request_id,
                error=error,
            )
            self._pending_records.append(record)
            raise
        except BackendError as exc:
            status = RequestStatus.error
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # defensive: an adapter bug must not kill the run
            status = RequestStatus.error
            error = f"unexpected {type(exc).__name__}: {exc}"

        e2e = max(last_t, 0.0)
        # A success with no tokens is contradictory; treat it as interrupted so
        # throughput is never credited for an empty response.
        if status.is_success and output_tokens == 0:
            status = RequestStatus.interrupted
            error = error or "stream completed with zero content tokens"

        return self._build_record(
            spec,
            started_at=started_at,
            status=status,
            ttft=ttft,
            e2e=e2e,
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            token_times=tuple(token_times),
            finish_reason=finish_reason,
            backend_request_id=backend_request_id,
            error=error,
        )

    @staticmethod
    def _build_record(
        spec: RequestSpec,
        *,
        started_at: float,
        status: RequestStatus,
        ttft: float | None,
        e2e: float,
        output_tokens: int,
        prompt_tokens: int | None,
        token_times: tuple[float, ...],
        finish_reason: str | None,
        backend_request_id: str | None,
        error: str | None,
    ) -> RequestRecord:
        return RequestRecord(
            request_id=spec.request_id,
            status=status,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            requested_output_tokens=spec.max_output_tokens,
            ttft_s=ttft,
            e2e_s=e2e,
            token_times_s=token_times,
            class_label=spec.class_label,
            repetition=spec.repetition,
            is_warmup=spec.is_warmup,
            started_at=started_at,
            error=error,
            finish_reason=finish_reason,
            backend_request_id=backend_request_id,
        )

    def _data_source(self) -> str:
        return "measured" if self.backend.capabilities.is_real else "synthetic"


def _status_from_signal(status: str) -> RequestStatus:
    """Map a backend failure label onto a terminal status, defaulting to error."""
    try:
        return RequestStatus(status)
    except ValueError:
        return RequestStatus.error
