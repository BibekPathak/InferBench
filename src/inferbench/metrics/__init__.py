"""Metric computation.

Public entry points:

- :func:`summarize` — a complete :class:`MetricsSummary` for one window.
- :func:`filter_measured` — remove warm-up records before summarizing.
- :func:`percentile` — the documented nearest-rank percentile.
"""

from __future__ import annotations

from inferbench.metrics.engine import (
    FailureStats,
    LatencyStats,
    MetricsSummary,
    Percentiles,
    ThroughputStats,
    failure_stats,
    filter_measured,
    latency_stats,
    percentile,
    summarize,
    throughput_stats,
)

__all__ = [
    "FailureStats",
    "LatencyStats",
    "MetricsSummary",
    "Percentiles",
    "ThroughputStats",
    "failure_stats",
    "filter_measured",
    "latency_stats",
    "percentile",
    "summarize",
    "throughput_stats",
]
