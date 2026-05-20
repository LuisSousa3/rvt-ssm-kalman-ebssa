#!/usr/bin/env python3
"""Create report-ready plots for the modelFinal evaluation artifacts.

The checkpoint folder already stores the raw CSV/JSON outputs.  This script
repackages the highest-signal findings into figures that are easier to cite in a
short report:

1. why the early validation checkpoint was selected;
2. how usable temporal context changes detection quality and error counts;
3. how much the hybrid RVT+Kalman tracker recovers on the showcased sequence.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CHECKPOINT_DIR = Path("ebssa_rvt/modelFinal/checkpoints")
DEFAULT_METRICS_CSV = Path("data/rvt_csv_logs/ebssa_gen1_s5_transfer/version_5/metrics.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Folder containing modelFinal checkpoint evaluation artifacts.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help="Lightning CSV log for the modelFinal training run.",
    )
    parser.add_argument(
        "--step-duration-ms",
        type=float,
        default=50.0,
        help="Duration represented by one temporal step; default matches stacked_histogram_dt=50.",
    )
    return parser.parse_args()


def save_training_selection_plot(metrics_csv: Path, output_dir: Path) -> Path:
    metrics = pd.read_csv(metrics_csv)
    val = metrics.dropna(subset=["val/AP"]).copy()
    best = val.loc[val["val/AP"].idxmax()]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(val["step"], val["val/AP"], label="AP", linewidth=2.0)
    ax.plot(val["step"], val["val/AP_50"], label="AP@0.50", linewidth=1.8)
    ax.plot(val["step"], val["val/AP_75"], label="AP@0.75", linewidth=1.8)
    ax.axvline(best["step"], color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.scatter(
        [best["step"]],
        [best["val/AP"]],
        color="black",
        zorder=4,
        label=f"selected checkpoint: step {int(best['step'])}",
    )
    ax.annotate(
        f"best AP = {best['val/AP']:.3f}",
        xy=(best["step"], best["val/AP"]),
        xytext=(best["step"] + 9000, best["val/AP"] + 0.035),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=9,
    )
    ax.set_title("Validation quality peaks early despite continued loss reduction")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation AP")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()

    output_path = output_dir / "report_training_selection.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_temporal_context_plot(
    sequence_study_csv: Path,
    output_dir: Path,
    step_duration_ms: float,
) -> Path:
    study = pd.read_csv(sequence_study_csv).sort_values("sequence_length").copy()
    study["history_s"] = study["sequence_length"] * step_duration_ms / 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    metric_ax = axes[0]
    for metric, label in [("precision", "precision"), ("recall", "recall"), ("f1", "F1")]:
        metric_ax.plot(
            study["history_s"],
            study[metric],
            marker="o",
            linewidth=2.0,
            label=label,
        )
    metric_ax.set_title("Detection quality vs usable event history")
    metric_ax.set_xlabel("usable history (s)")
    metric_ax.set_ylabel("test metric")
    metric_ax.set_ylim(0.0, 0.7)
    metric_ax.set_xticks(study["history_s"])
    metric_ax.grid(alpha=0.25)
    metric_ax.legend(frameon=True)

    count_ax = axes[1]
    for metric, label in [
        ("true_positives", "true positives"),
        ("false_positives", "false positives"),
        ("false_negatives", "false negatives"),
    ]:
        count_ax.plot(
            study["history_s"],
            study[metric],
            marker="o",
            linewidth=2.0,
            label=label,
        )
    count_ax.set_title("More history both recovers objects and suppresses noise")
    count_ax.set_xlabel("usable history (s)")
    count_ax.set_ylabel("count over test split")
    count_ax.set_xticks(study["history_s"])
    count_ax.grid(alpha=0.25)
    count_ax.legend(frameon=True)

    fig.tight_layout()
    output_path = output_dir / "report_temporal_context.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    study.to_csv(output_dir / "report_temporal_context_summary.csv", index=False)
    return output_path


def save_tracking_recovery_plot(completeness_csv: Path, output_dir: Path) -> Path:
    rows = pd.read_csv(completeness_csv)
    active = rows[rows["gt_count"] > 0].copy()

    def outcome_counts(prefix: str) -> list[int]:
        frame_completeness = active[f"{prefix}_frame_completeness"]
        return [
            int((frame_completeness == 0.0).sum()),
            int(((frame_completeness > 0.0) & (frame_completeness < 1.0)).sum()),
            int((frame_completeness == 1.0).sum()),
        ]

    labels = ["missed", "partial", "full"]
    colors = ["#d95f02", "#7570b3", "#1b9e77"]
    systems = ["RVT bbox only", "RVT + Kalman"]
    counts = np.asarray([outcome_counts("rvt"), outcome_counts("hybrid")])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    bar_ax = axes[0]
    bottom = np.zeros(len(systems))
    for idx, (label, color) in enumerate(zip(labels, colors)):
        bar_ax.bar(systems, counts[:, idx], bottom=bottom, color=color, label=label)
        bottom += counts[:, idx]
    bar_ax.set_title("Active-frame outcomes on showcased sequence")
    bar_ax.set_ylabel("frames with at least one ground-truth box")
    bar_ax.legend(frameon=True)
    bar_ax.grid(axis="y", alpha=0.25)

    line_ax = axes[1]
    line_ax.plot(
        rows["time_s"],
        rows["rvt_cumulative_completeness"],
        label="RVT bbox only",
        linewidth=2.0,
    )
    line_ax.plot(
        rows["time_s"],
        rows["hybrid_cumulative_completeness"],
        label="RVT + Kalman",
        linewidth=2.0,
    )
    line_ax.set_title("Hybrid tracking gives a small but targeted recovery")
    line_ax.set_xlabel("time (s)")
    line_ax.set_ylabel("cumulative completeness")
    line_ax.set_ylim(0.86, 1.01)
    line_ax.grid(alpha=0.25)
    line_ax.legend(frameon=True)

    fig.tight_layout()
    output_path = output_dir / "report_tracking_recovery.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    output_dir = checkpoint_dir / "report_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_plot = save_training_selection_plot(args.metrics_csv, output_dir)
    temporal_plot = save_temporal_context_plot(
        checkpoint_dir / "sequence_length_study" / "sequence_length_study.csv",
        output_dir,
        step_duration_ms=args.step_duration_ms,
    )
    tracking_plot = save_tracking_recovery_plot(
        checkpoint_dir / "20170214-21-15_SL8RB_21938_labelled_tracking_completeness.csv",
        output_dir,
    )

    print("Wrote:")
    for path in [training_plot, temporal_plot, tracking_plot]:
        print(path)


if __name__ == "__main__":
    main()
