"""Charts for experiment reports.

The two charts that matter for a sweep are:

1. **Throughput vs the varied axis** — shows where scaling stops paying off.
2. **Latency vs the varied axis** — shows the tail rising as load increases.

Both are plotted only for numeric axes. A categorical sweep (short vs long
prompts, for example) is better read from the table, so charts are skipped with
a note rather than drawn as misleading bars.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from inferbench.logging_utils import get_logger, log_event

if TYPE_CHECKING:
    from inferbench.experiment.report import VariantSummary
    from inferbench.experiment.runner import StudyResult

__all__ = ["try_make_study_charts"]

_log = get_logger("experiment.charts")


def _numeric_series(
    summaries: list[VariantSummary],
) -> tuple[list[float], list[VariantSummary]] | None:
    """Return (x values, summaries) when the varied axis is numeric."""
    from inferbench.experiment.report import _numeric

    keys = [_numeric(s.varies_value) for s in summaries]
    if not summaries or any(key is None for key in keys):
        return None
    return [float(key) for key in keys if key is not None], summaries


def try_make_study_charts(
    study: StudyResult, summaries: list[VariantSummary], output_dir: Path
) -> tuple[list[str], list[str]]:
    """Attempt to render study charts.

    Returns ``(chart_paths, notes)``; failure to render is reported, never fatal.
    """
    from inferbench.analysis.charts import charts_available

    if not charts_available():
        return [], [
            "Charts skipped: matplotlib is not installed. "
            "Install with `pip install -e '.[analysis]'` to enable charts."
        ]

    series = _numeric_series(summaries)
    if series is None:
        return [], [
            "Charts skipped: the varied axis is not numeric, so a trend line would "
            "be misleading. Read the table instead."
        ]
    xs, ordered = series

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-specific
        return [], [f"Charts skipped: matplotlib import failed ({exc})."]

    charts: list[str] = []
    notes: list[str] = []
    axis_label = study.varies or "variant"

    # --- Throughput -------------------------------------------------------
    try:
        rps = [s.requests_per_s for s in ordered]
        if any(value is not None for value in rps):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            # Drop missing points so the line never crosses a gap.
            pairs = [(x, y) for x, y in zip(xs, rps, strict=False) if y is not None]
            if pairs:
                px, py = zip(*pairs, strict=False)
                ax.plot(px, py, marker="o", color="#3b6ea5", label="completed req/s")
            ax.set_xlabel(axis_label)
            ax.set_ylabel("requests / second")
            ax.set_title(f"Throughput vs {axis_label} — {study.experiment}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = output_dir / f"experiment-{study.experiment}-throughput.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            charts.append(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"Throughput chart skipped: {exc}")

    # --- Latency ----------------------------------------------------------
    try:
        ttft = [s.ttft_p50 for s in ordered]
        e2e_p95 = [s.e2e_p95 for s in ordered]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ttft_pairs = [(x, y) for x, y in zip(xs, ttft, strict=False) if y is not None]
        if ttft_pairs:
            tx, ty = zip(*ttft_pairs, strict=False)
            ax.plot(tx, ty, marker="o", color="#3b6ea5", label="TTFT p50")
        e2e_pairs = [(x, y) for x, y in zip(xs, e2e_p95, strict=False) if y is not None]
        if e2e_pairs:
            ex, ey = zip(*e2e_pairs, strict=False)
            ax.plot(ex, ey, marker="s", color="#a5573b", label="E2E p95")
        ax.set_xlabel(axis_label)
        ax.set_ylabel("seconds")
        ax.set_title(f"Latency vs {axis_label} — {study.experiment}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"experiment-{study.experiment}-latency.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        charts.append(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"Latency chart skipped: {exc}")

    log_event(_log, logging.INFO, "study charts generated", count=len(charts))
    if not charts:
        notes.append("Charts skipped: rendering produced no output.")
    return charts, notes
