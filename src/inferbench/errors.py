"""Typed exception hierarchy for InferBench.

Every error raised deliberately by InferBench derives from
:class:`InferBenchError` so callers (and the CLI) can distinguish expected
failure modes from programming bugs. Messages are written to be actionable:
they state what failed, why, and — where known — how to fix it.
"""

from __future__ import annotations

__all__ = [
    "InferBenchError",
    "ConfigurationError",
    "BackendError",
    "BackendNotReadyError",
    "ModelNotFoundError",
    "RequestTimeoutError",
    "StreamInterruptedError",
    "MetricsError",
    "StorageError",
    "RunNotFoundError",
    "IncomparableRunsError",
    "EnvironmentError_",
]


class InferBenchError(Exception):
    """Base class for all expected InferBench failures."""


class ConfigurationError(InferBenchError):
    """A configuration file or value is missing, malformed, or inconsistent."""


class BackendError(InferBenchError):
    """Base class for backend adapter failures."""


class BackendNotReadyError(BackendError):
    """The backend server is reachable but not ready to serve requests."""


class ModelNotFoundError(BackendError):
    """The requested model is not available on the configured backend."""


class RequestTimeoutError(BackendError):
    """A request exceeded its configured deadline.

    This is a *measured outcome*, not a crash: the runner records it as a
    failed request and continues.
    """


class StreamInterruptedError(BackendError):
    """A streaming response ended before completion (disconnect or server bug).

    Recorded as a failed request with a partial generation.
    """


class MetricsError(InferBenchError):
    """Metric computation was requested on data that cannot support it."""


class StorageError(InferBenchError):
    """Persisting or loading experiment data failed."""


class RunNotFoundError(StorageError):
    """A referenced run id does not exist in the store."""


class IncomparableRunsError(InferBenchError):
    """Two runs cannot be compared fairly (e.g. different token distributions)."""


class EnvironmentError_(InferBenchError):
    """Environment inspection failed in a way the user must address.

    Named with a trailing underscore to avoid shadowing the builtin
    ``EnvironmentError``.
    """
