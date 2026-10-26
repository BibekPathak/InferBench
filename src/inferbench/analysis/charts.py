"""Optional chart generation for reports.

Charts are strictly optional. If matplotlib is not installed, or rendering
fails for any reason, the report is still produced with text tables plus a note
explaining that charts were skipped. A benchmark tool that refuses to emit a
report because a plotting library is missing is a worse tool.
"""

from __future__ import annotations

import logging
from pathlib import Path

from inferbench.logging_utils import get_logger, log_event
from inferbench.storage import ExperimentStore

__all__ = ["try_make_charts", "charts_available"]

_log = get_logger("analysis.charts")


def charts_available() -> bool:
    """Whether matplotlib can be imported."""
    try:
        import matplotlib  # noqa: F401
    except Exception:
        return False
    return True


def try_make_charts(
    store: ExperimentStore, run_id: str, run_dir: Path
) -> tuple[list[str], list[str]]:
    """Attempt to render charts for a run.

    Returns:
        ``(chart_paths, notes)``. ``chart_paths`` are absolute-ish paths for the
        Markdown image links; ``notes`` explains any degradation.
    """
    if not charts_available():
        return [], [
            "Charts skipped: matplotlib is not installed. "
            "Install with `pip install -e '.[analysis]'` to enable charts."
        ]

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless; never try to open a window
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-specific
        return [], [f"Charts skipped: matplotlib import failed ({exc})."]

    try:
        records = store.load_raw_records(run_id)
    except Exception as exc:
        return [], [f"Charts skipped: could not load raw records ({exc})."]

    measured = [r for r in records if not r.is_warmup and r.is_success]
    if not measured:
        return [], ["Charts skipped: no successful measured requests to plot."]

    charts: list[str] = []
    notes: list[str] = []

    # --- Latency distribution --------------------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ttft = [r.ttft_s for r in measured if r.ttft_s is not None]
        e2e = [r.e2e_s for r in measured]
        if ttft:
            axes[0].hist(ttft, bins=min(20, max(5, len(ttft))), color="#3b6ea5")
            axes[0].set_title("TTFT (s)")
        else:
            axes[0].set_title("TTFT (s) — no samples")
        axes[0].set_ylabel("requests")

        if e2e:
            axes[1].hist(e2e, bins=min(20, max(5, len(e2e))), color="#a5573b")
            axes[1].set_title("End-to-end latency (s)")
        axes[1].set_xlabel("seconds")

        fig.suptitle(f"Latency distribution — {run_id}", fontsize=10)
        fig.tight_layout()
        path = run_dir / "chart_latency.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        charts.append(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"Latency chart skipped: {exc}")

    # --- Throughput vs concurrency proxy ---------------------------------
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        by_class: dict[str, list[float]] = {}
        for record in measured:
            by_class.setdefault(record.class_label, []).append(record.e2e_s)
        labels = sorted(by_class)
        if labels:
            data = [by_class[label] for label in labels]
            ax.boxplot(data, tick_labels=labels)
            ax.set_ylabel("end-to-end latency (s)")
            ax.set_title("Latency by request class")
        fig.tight_layout()
        path = run_dir / "chart_classes.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        charts.append(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"Class chart skipped: {exc}")

    log_event(_log, logging.INFO, "charts generated", run_id=run_id, count=len(charts))
    if not charts:
        notes.append("Charts skipped: rendering produced no output.")
    return charts, notes
