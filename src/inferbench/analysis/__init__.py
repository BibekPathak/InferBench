"""Analysis and reporting."""

from __future__ import annotations

from inferbench.analysis.charts import charts_available, try_make_charts
from inferbench.analysis.report import generate_report, render_markdown

__all__ = [
    "charts_available",
    "generate_report",
    "render_markdown",
    "try_make_charts",
]
