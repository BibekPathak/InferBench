"""Multi-run experiments: experiments A (baseline) and B (workload sensitivity).

An experiment expands into several runs that differ along one declared variable,
executes them against a single backend, and reports the trend. See
:mod:`inferbench.experiment.config` for the definition format.
"""

from __future__ import annotations

from inferbench.experiment.config import (
    ExperimentConfig,
    Variant,
    expand_experiment,
    load_experiment,
)
from inferbench.experiment.report import (
    VariantSummary,
    collect_variant_summaries,
    generate_study_report,
    render_study_markdown,
)
from inferbench.experiment.runner import StudyResult, StudyRunner

__all__ = [
    "ExperimentConfig",
    "StudyResult",
    "StudyRunner",
    "Variant",
    "VariantSummary",
    "collect_variant_summaries",
    "expand_experiment",
    "generate_study_report",
    "load_experiment",
    "render_study_markdown",
]
