"""Core data types shared across InferBench.

These are plain, immutable-ish dataclasses rather than pydantic models: they are
created in hot paths (once per streamed token, potentially millions of times
per run) and should be cheap. Validation happens at the configuration boundary
(see :mod:`inferbench.config`); these types describe *observed* facts.

The central type is :class:`RequestRecord`, the raw per-request observation.
Everything in the metrics engine derives from a collection of these, and they
are what gets written to the raw JSONL — so completeness here is what makes a
run auditable.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "RequestStatus",
    "TokenEvent",
    "RequestSpec",
    "RequestRecord",
    "RunMetadata",
    "new_run_id",
]


class RequestStatus(StrEnum):
    """Terminal state of a request.

    Failures are first-class measured outcomes, not exceptions that vanish.
    Every request that is dispatched ends in exactly one of these states.
    """

    success = "success"
    timeout = "timeout"
    cancelled = "cancelled"
    error = "error"
    #: The stream ended before the requested number of tokens arrived. The
    #: partial token count is preserved so throughput is not overstated.
    interrupted = "interrupted"

    @property
    def is_success(self) -> bool:
        return self is RequestStatus.success


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """A single streamed token and the moment it arrived.

    ``t`` is a monotonic timestamp in seconds, taken from
    :func:`time.perf_counter`, so it is immune to wall-clock adjustments.
    """

    t: float
    text: str = ""


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """A request to be issued, before it is executed.

    This is the unit produced by the workload generator. ``prompt_tokens`` and
    ``max_output_tokens`` are *targets*: the backend may produce a different
    actual prompt length depending on its tokenizer, which is recorded on the
    resulting :class:`RequestRecord`.
    """

    request_id: str
    prompt: str
    target_prompt_tokens: int
    max_output_tokens: int
    #: Logical class label, e.g. "short"/"medium"/"long", for mixed workloads.
    #: Used to verify that compared runs used the same composition.
    class_label: str = "fixed"
    #: Index of the repetition this request belongs to (0-based).
    repetition: int = 0
    #: Whether this request is warm-up (excluded from measured statistics).
    is_warmup: bool = False


@dataclass(slots=True)
class RequestRecord:
    """Raw, per-request observation. The atomic unit of reproducibility.

    Timestamps are monotonic seconds relative to the start of the request, so a
    record is self-contained: every metric in :mod:`inferbench.metrics` can be
    recomputed from the record alone, without replaying the run.
    """

    request_id: str
    status: RequestStatus

    # --- Token accounting -------------------------------------------------
    #: Actual prompt token count as reported by the backend, when available.
    #: ``None`` means the backend did not report it (never guessed).
    prompt_tokens: int | None
    #: Number of content tokens actually received on the stream.
    output_tokens: int
    #: The ``max_tokens`` value requested.
    requested_output_tokens: int

    # --- Timing (monotonic seconds, relative to dispatch) -----------------
    #: Time from dispatch to the first content token. ``None`` if none arrived.
    ttft_s: float | None
    #: Time from dispatch to stream end (or timeout/cancel).
    e2e_s: float
    #: Per-token arrival offsets, used to compute inter-token latency and to
    #: make TPOT recomputable from raw data.
    token_times_s: tuple[float, ...]

    # --- Context ----------------------------------------------------------
    class_label: str
    repetition: int
    is_warmup: bool
    #: Wall-clock start time (Unix epoch seconds) for correlation with logs and
    #: GPU telemetry.
    started_at: float
    #: Populated for non-success statuses; never invented for successes.
    error: str | None = None
    #: Backend-reported finish reason, when available.
    finish_reason: str | None = None
    #: Optional backend request id, for cross-referencing server logs.
    backend_request_id: str | None = None

    @property
    def is_success(self) -> bool:
        return self.status.is_success

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for the raw JSONL."""
        data = asdict(self)
        data["status"] = self.status.value
        # Tuples serialize to JSON arrays, but be explicit for readers.
        data["token_times_s"] = list(self.token_times_s)
        return data

    @property
    def duration_s(self) -> float:
        """End-to-end duration, regardless of terminal status."""
        return self.e2e_s


@dataclass(slots=True)
class RunMetadata:
    """Everything needed to reproduce and audit a run.

    A result without this metadata is not considered valid by
    ``inferbench validate``.
    """

    run_id: str
    run_name: str
    experiment: str
    created_at: float
    started_at: float
    finished_at: float | None
    #: Full run configuration, serialized.
    config: dict[str, Any]
    #: Environment fingerprint from :mod:`inferbench.env`.
    environment: dict[str, Any]
    #: Resolved backend identity: kind, version string, and model identifier.
    backend_kind: str
    backend_version: str | None
    model: str
    #: ``"measured"`` when a real backend produced the data, ``"synthetic"``
    #: when the mock backend did. This flag is what keeps the project honest:
    #: reports surface it prominently.
    data_source: str
    tool_version: str
    notes: str = ""
    #: Backend-reported model revision, when available.
    model_revision: str | None = None
    #: Free-form extra facts (dtype, quantization, served config).
    backend_details: dict[str, Any] = field(default_factory=dict)
    #: Populated if the run was aborted partway.
    incomplete: bool = False
    incomplete_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_run_id(run_name: str, *, when: float | None = None) -> str:
    """Generate a sortable, unique run id.

    Format: ``<sanitized-name>-<unix-seconds>-<6 hex chars>``. The suffix keeps
    runs distinguishable when they share a name and start in the same second,
    which happens constantly in tests and CI.
    """
    when = time.time() if when is None else when
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in run_name)
    safe = safe.strip("-") or "run"
    # Deterministic, collision-resistant suffix derived from nanosecond time.
    digest = hashlib.sha256(f"{safe}:{when!r}".encode()).hexdigest()[:6]
    return f"{safe}-{int(when)}-{digest}"
