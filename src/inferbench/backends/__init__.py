"""Backend adapters.

Import concrete adapters from here to avoid reaching into submodules. The mock
backend is always importable; the vLLM adapter imports ``httpx`` only, so it is
also always available (``httpx`` is a core dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inferbench.backends.base import (
    BackendAdapter,
    BackendCapabilities,
    CompletionSignal,
    FailureSignal,
    StreamEvent,
    TokenChunk,
)
from inferbench.backends.mock import MockBackend, MockBackendOptions
from inferbench.config import BackendKind
from inferbench.errors import ConfigurationError

if TYPE_CHECKING:
    from inferbench.config import BackendConfig

__all__ = [
    "BackendAdapter",
    "BackendCapabilities",
    "CompletionSignal",
    "FailureSignal",
    "StreamEvent",
    "TokenChunk",
    "MockBackend",
    "MockBackendOptions",
    "create_backend",
]


def create_backend(config: BackendConfig, **kwargs: object) -> BackendAdapter:
    """Construct the adapter named by ``config.kind``.

    Args:
        config: Backend configuration.
        **kwargs: Adapter-specific overrides, e.g. ``options`` for the mock
            backend. These are test hooks and are intentionally not part of
            :class:`BackendConfig`.

    Raises:
        ConfigurationError: The requested backend kind is not supported.
    """
    if config.kind is BackendKind.mock:
        from inferbench.backends.mock import MockBackendOptions

        options = kwargs.get("options")
        if options is None and config.synthetic is not None:
            # A synthetic study may declare its timing model in config; it is
            # still synthetic, and the adapter records that fact.
            s = config.synthetic
            options = MockBackendOptions(
                ttft_base_s=s.ttft_base_s,
                tpot_base_s=s.tpot_base_s,
                prefill_tokens_per_s=s.prefill_tokens_per_s,
                decode_contention=s.decode_contention,
                jitter=s.jitter,
                seed=s.seed,
            )
        if options is not None and not isinstance(options, MockBackendOptions):
            raise ConfigurationError("mock backend 'options' must be MockBackendOptions")
        return MockBackend(config, options=options)

    if config.kind is BackendKind.vllm:
        # Imported lazily so the mock path never pays for it.
        from inferbench.backends.vllm import VLLMBackend

        return VLLMBackend(config)

    raise ConfigurationError(f"Unsupported backend kind: {config.kind!r}")
