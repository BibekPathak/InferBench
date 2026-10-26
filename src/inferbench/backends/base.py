"""Backend adapter interface and shared helpers.

Every inference backend — the mock, vLLM, and any future SGLang adapter —
implements :class:`BackendAdapter`. The interface is deliberately narrow and
streaming-first, because the metrics that matter most (TTFT, inter-token
latency) only exist if the adapter exposes per-token arrival times.

Important design rules, enforced by the contract rather than convention:

- An adapter must never fabricate a measurement. If the backend does not report
  prompt token counts, the adapter yields ``None``; the metrics engine handles
  that rather than the adapter guessing.
- An adapter must never raise for an expected per-request outcome (timeout,
  cancellation, disconnect, server error). It yields a terminal event carrying
  the status. Exceptions are reserved for programming errors and for failures
  that make the whole run meaningless (e.g. the model does not exist).
- An adapter must be reentrant across concurrent requests when the backend is.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from inferbench.config import BackendConfig
from inferbench.types import RequestSpec

__all__ = [
    "BackendCapabilities",
    "StreamEvent",
    "TokenChunk",
    "CompletionSignal",
    "BackendAdapter",
]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """What a backend can do, so the runner can adapt instead of guessing.

    The runner records this in run metadata and refuses to run experiments that
    require a capability the backend lacks, rather than silently producing a
    misleading result.
    """

    #: Whether the backend streams token-by-token (required for TTFT/ITL).
    streaming: bool = True
    #: Whether the backend reports the actual prompt token count.
    reports_prompt_tokens: bool = False
    #: Whether the backend reports a server-side request identifier.
    reports_request_id: bool = False
    #: Whether the backend reports token-level logprobs (needed for quality
    #: comparisons in Experiment E).
    reports_logprobs: bool = False
    #: Whether the backend is a real serving engine (vs a synthetic mock).
    is_real: bool = False


@dataclass(frozen=True, slots=True)
class TokenChunk:
    """One streamed content token."""

    #: Monotonic seconds since request dispatch.
    t: float
    text: str = ""


@dataclass(frozen=True, slots=True)
class CompletionSignal:
    """Terminal success event for a request.

    Carries the authoritative token counts. ``prompt_tokens`` is ``None`` when
    the backend does not report it.
    """

    t: float
    prompt_tokens: int | None
    output_tokens: int
    finish_reason: str | None = None
    backend_request_id: str | None = None


@dataclass(frozen=True, slots=True)
class FailureSignal:
    """Terminal failure event for a request.

    ``status`` distinguishes timeout / cancelled / interrupted / error so the
    metrics engine can report them separately rather than lumping all
    non-successes together.
    """

    t: float
    status: str
    message: str
    output_tokens: int = 0
    backend_request_id: str | None = None


#: Union of everything an adapter may yield during a stream.
StreamEvent = TokenChunk | CompletionSignal | FailureSignal


class BackendAdapter(abc.ABC):
    """Abstract streaming inference backend.

    Lifecycle: construct, then ``await setup()``, then use ``stream()`` any
    number of times, then ``await teardown()``. ``setup()`` is where readiness
    and model availability are validated — a failure there means the run must
    not start, whereas a failure inside ``stream()`` is a recorded outcome.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short adapter name, e.g. ``"vllm"``."""

    @property
    @abc.abstractmethod
    def version(self) -> str | None:
        """Backend version string, or ``None`` if unavailable.

        Recorded in run metadata; ``None`` is honest, an invented version is not.
        """

    @property
    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Static capabilities of this adapter/backend."""

    @abc.abstractmethod
    async def setup(self) -> None:
        """Validate readiness and model availability; acquire resources.

        Raises:
            BackendNotReadyError: Server unreachable or not ready in time.
            ModelNotFoundError: Requested model is not served.
            BackendError: Any other setup failure.
        """

    @abc.abstractmethod
    async def teardown(self) -> None:
        """Release resources. Must be safe to call more than once."""

    @abc.abstractmethod
    def stream(self, spec: RequestSpec) -> AsyncIterator[StreamEvent]:
        """Issue ``spec`` and yield stream events.

        Implementations must yield exactly one terminal event
        (:class:`CompletionSignal` or :class:`FailureSignal`) as the last item,
        and must not raise for expected per-request failures.
        """

    @abc.abstractmethod
    async def describe(self) -> dict[str, Any]:
        """Return backend-specific facts for run metadata.

        Examples: served model list, engine flags, dtype, quantization. Used to
        populate ``RunMetadata.backend_details``.
        """

    async def __aenter__(self) -> BackendAdapter:
        await self.setup()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.teardown()
