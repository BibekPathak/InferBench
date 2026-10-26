"""Analysis and reporting."""

from __future__ import annotations

from inferbench.analysis.charts import charts_available, try_make_charts
from inferbench.analysis.compare import (
    ComparisonResult,
    CompatibilityReport,
    MetricDelta,
    bootstrap_ci,
    check_compatibility,
    compare_runs,
    format_comparison,
)
from inferbench.analysis.report import generate_report, render_markdown
from inferbench.analysis.validate import (
    Finding,
    Severity,
    ValidationReport,
    format_report,
    validate_run,
)

__all__ = [
    "ComparisonResult",
    "CompatibilityReport",
    "Finding",
    "MetricDelta",
    "Severity",
    "ValidationReport",
    "bootstrap_ci",
    "charts_available",
    "check_compatibility",
    "compare_runs",
    "format_comparison",
    "format_report",
    "generate_report",
    "render_markdown",
    "try_make_charts",
    "validate_run",
]
