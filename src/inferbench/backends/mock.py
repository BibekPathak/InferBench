"""Deterministic mock backend.

The mock backend is not a toy. It is the reference implementation of the
:class:`~inferbench.backends.base.BackendAdapter` contract, and it exists so the
entire benchmark pipeline — runner, metrics, storage, reporting — can be
developed and *tested against known ground truth* without a GPU.

Because its timing model is explicit, tests can assert that recomputed metrics
match analytically expected values, which is how InferBench validates its
latency math (a Definition-of-Done requirement).

Timing model
------------
- **TTFT** = ``ttft_base_s`` (prefill), optionally scaled by prompt length via
  ``prefill_tokens_per_s``.
- **Inter-token latency** = ``tpot_base_s`` (decode), optionally scaled by
  current in-flight concurrency via ``decode_contention`` to emulate a
  saturating server.

The mock never claims to be real: ``capabilities.is_real`` is ``False`` and run
metadata records ``data_source = "synthetic"``.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from typing import Any

from inferbench.backends.base import (
    BackendAdapter,
    BackendCapabilities,
    CompletionSignal,
    FailureSignal,
    StreamEvent,
    TokenChunk,
)
from inferbench.config import BackendConfig, BackendKind
from inferbench.errors import BackendNotReadyError
from inferbench.logging_utils import get_logger, log_event
from inferbench.types import RequestSpec

__all__ = ["MockBackend", "MockBackendOptions"]

_log = get_logger("backends.mock")


class MockBackendOptions:
    """Tunable timing/failure model for the mock backend.

    These are deliberately *not* part of :class:`BackendConfig`: they exist only
    for tests and smoke runs, and must never be mistaken for real backend
    configuration.
    """

    def __init__(
        self,
        *,
        ttft_base_s: float = 0.02,
        tpot_base_s: float = 0.005,
        prefill_tokens_per_s: float | None = 20_000.0,
        decode_contention: float = 0.0,
        jitter: float = 0.0,
        seed: int = 0,
        fail_every: int = 0,
        fail_status: str = "error",
        interrupt_every: int = 0,
        report_prompt_tokens: bool = True,
        ready: bool = True,
        setup_delay_s: float = 0.0,
        real_time: bool = True,
    ) -> None:
        self.ttft_base_s = ttft_base_s
        self.tpot_base_s = tpot_base_s
        self.prefill_tokens_per_s = prefill_tokens_per_s
        self.decode_contention = decode_contention
        self.jitter = jitter
        self.seed = seed
        self.fail_every = fail_every
        self.fail_status = fail_status
        self.interrupt_every = interrupt_every
        self.report_prompt_tokens = report_prompt_tokens
        self.ready = ready
        self.setup_delay_s = setup_delay_s
        #: When ``False`` the mock emits all events with zero elapsed time and
        #: does not sleep. Used by tests that only check plumbing, keeping the
        #: suite fast. Metrics tests set ``real_time=True`` and use
        #: deterministic *synthetic* timing via the analytic model.
        self.real_time = real_time

    def expected_ttft(self, prompt_tokens: int) -> float:
        """Analytic TTFT for a request, matching what :meth:`stream` produces."""
        if self.prefill_tokens_per_s and self.prefill_tokens_per_s > 0:
            return self.ttft_base_s + prompt_tokens / self.prefill_tokens_per_s
        return self.ttft_base_s

    def expected_tpot(self, concurrency: int = 1) -> float:
        """Analytic inter-token latency at a given in-flight concurrency."""
        return self.tpot_base_s * (1.0 + self.decode_contention * max(0, concurrency - 1))


class MockBackend(BackendAdapter):
    """In-process synthetic backend implementing the full adapter contract."""

    def __init__(
        self,
        config: BackendConfig | None = None,
        *,
        options: MockBackendOptions | None = None,
    ) -> None:
        super().__init__(config or BackendConfig(kind=BackendKind.mock))
        self.options = options or MockBackendOptions()
        self._inflight = 0
        self._counter = 0
        self._setup_done = False
        # A dedicated RNG, seeded for reproducibility, used only for jitter.
        self._rng = random.Random(self.options.seed)

    @property
    def name(self) -> str:
        return "mock"

    @property
    def version(self) -> str | None:
        # Honest: a synthetic backend has no engine version to report.
        return None

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=True,
            reports_prompt_tokens=self.options.report_prompt_tokens,
            reports_request_id=True,
            reports_logprobs=False,
            is_real=False,
        )

    async def setup(self) -> None:
        if not self.options.ready:
            raise BackendNotReadyError("Mock backend configured as not ready")
        if self.options.setup_delay_s > 0:
            await asyncio.sleep(self.options.setup_delay_s)
        self._setup_done = True
        log_event(_log, logging.INFO, "mock backend ready", model=self.config.model)

    async def teardown(self) -> None:
        self._setup_done = False

    async def describe(self) -> dict[str, Any]:
        return {
            "adapter": "mock",
            "model": self.config.model,
            "synthetic": True,
            "timing_model": {
                "ttft_base_s": self.options.ttft_base_s,
                "tpot_base_s": self.options.tpot_base_s,
                "prefill_tokens_per_s": self.options.prefill_tokens_per_s,
                "decode_contention": self.options.decode_contention,
            },
        }

    def _next_index(self) -> int:
        self._counter += 1
        return self._counter

    async def stream(self, spec: RequestSpec) -> AsyncIterator[StreamEvent]:
        index = self._next_index()
        opts = self.options
        self._inflight += 1
        try:
            # --- Prefill / TTFT ------------------------------------------
            ttft = opts.expected_ttft(spec.target_prompt_tokens)
            # Failure and interruption injection key off the request index so
            # that behaviour is deterministic and reproducible.
            if opts.fail_every and index % opts.fail_every == 0:
                await _sleep(ttft, opts.real_time)
                yield FailureSignal(
                    t=ttft,
                    status=opts.fail_status,
                    message=f"injected failure on request {index}",
                    backend_request_id=f"mock-{index}",
                )
                return

            await _sleep(ttft, opts.real_time)
            yield TokenChunk(t=ttft, text=_token_text(0))

            # --- Decode ---------------------------------------------------
            tpot = opts.expected_tpot(self._inflight)
            target = spec.max_output_tokens
            emitted = 1

            while emitted < target:
                step = tpot
                if opts.jitter:
                    step = max(0.0, tpot + self._rng.uniform(-opts.jitter, opts.jitter))

                if (
                    opts.interrupt_every
                    and index % opts.interrupt_every == 0
                    and emitted >= target // 2
                ):
                    # Emit a partial generation then report interruption. This
                    # exercises the "partial generation" path end-to-end.
                    yield FailureSignal(
                        t=ttft + emitted * tpot,
                        status="interrupted",
                        message=f"injected interruption on request {index}",
                        output_tokens=emitted,
                        backend_request_id=f"mock-{index}",
                    )
                    return

                await _sleep(step, opts.real_time)
                emitted += 1
                yield TokenChunk(t=ttft + (emitted - 1) * tpot, text=_token_text(emitted - 1))

            elapsed = ttft + (emitted - 1) * tpot if emitted > 1 else ttft
            yield CompletionSignal(
                t=elapsed,
                prompt_tokens=spec.target_prompt_tokens if opts.report_prompt_tokens else None,
                output_tokens=emitted,
                finish_reason="length",
                backend_request_id=f"mock-{index}",
            )
        finally:
            self._inflight -= 1

    # --- Introspection for tests -----------------------------------------

    @property
    def inflight(self) -> int:
        """Current number of concurrent streams (test observability)."""
        return self._inflight


def _token_text(index: int) -> str:
    """Deterministic placeholder token text.

    Content is irrelevant to timing metrics, but a stable, non-empty string
    keeps streaming code paths (and any future tokenizer work) realistic.
    """
    return f"tok{index % 100}"


async def _sleep(seconds: float, real_time: bool) -> None:
    if real_time and seconds > 0:
        await asyncio.sleep(seconds)
