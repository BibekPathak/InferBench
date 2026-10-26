"""InferBench — real-world LLM inference benchmarking and optimization platform.

The package is organized around a strict separation of concerns:

- ``inferbench.backends``  — inference engine adapters (vLLM, mock).
- ``inferbench.workload``  — deterministic workload generation.
- ``inferbench.metrics``   — latency/throughput metric computation.
- ``inferbench.runner``    — experiment orchestration.
- ``inferbench.storage``   — raw + aggregated result persistence.
- ``inferbench.analysis``  — comparison and reporting.

Nothing in this package requires a GPU to import. GPU-specific behavior is
detected at runtime and never assumed.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
