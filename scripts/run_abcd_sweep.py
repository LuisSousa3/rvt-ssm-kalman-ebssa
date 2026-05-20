#!/usr/bin/env python3
"""Run the A/B/C learning-rate sweep, then train D with the best A-C LR."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DataConfig
from methods.pipeline import run_pipeline
from methods.rvt_training import (
    checkpoint_metric_value,
    default_rvt_checkpoint_dir,
    find_rvt_checkpoint_in_dir,
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    test_name: str
    learning_rate: float
    weight_decay: float
    max_steps: int = 30_000


FIXED_RUNS = {
    "A": RunSpec("A", "modelA", 5e-5, 0.0),
    "B": RunSpec("B", "modelB", 1e-4, 0.0),
    "C": RunSpec("C", "modelC", 2.5e-5, 0.0),
}
RUN_ORDER = ("A", "B", "C", "D")


def config_for(spec: RunSpec) -> DataConfig:
    """Build the project config for one sweep run."""
    return replace(
        DataConfig(),
        pipeline_stages=("train",),
        rvt_test_name=spec.test_name,
        rvt_learning_rate=spec.learning_rate,
        rvt_weight_decay=spec.weight_decay,
        rvt_max_steps=spec.max_steps,
        rvt_use_lr_scheduler=False,
    )


def best_stage1_run() -> tuple[RunSpec, float]:
    """Return the best A-C run using the configured checkpoint metric."""
    scored: list[tuple[float, RunSpec]] = []
    missing: list[str] = []

    for label in ("A", "B", "C"):
        spec = FIXED_RUNS[label]
        cfg = config_for(spec)
        checkpoint_dir = PROJECT_ROOT / default_rvt_checkpoint_dir(cfg)
        if not checkpoint_dir.exists():
            missing.append(f"{label}: missing {checkpoint_dir}")
            continue

        checkpoint = find_rvt_checkpoint_in_dir(checkpoint_dir, cfg, prefer_best=True)
        metric_name = str(cfg.rvt_checkpoint_selection_metric)
        metric_value = checkpoint_metric_value(checkpoint, metric_name)
        if metric_value is None:
            missing.append(f"{label}: no {metric_name} in {checkpoint.name}")
            continue
        scored.append((metric_value, spec))

    if missing:
        details = "\n  ".join(missing)
        raise RuntimeError(
            "Cannot choose run D's learning rate until A-C have scored checkpoints:\n"
            f"  {details}"
        )

    mode = str(DataConfig().rvt_checkpoint_selection_mode).lower()
    if mode == "max":
        metric_value, spec = max(scored, key=lambda item: item[0])
    elif mode == "min":
        metric_value, spec = min(scored, key=lambda item: item[0])
    else:
        raise ValueError("rvt_checkpoint_selection_mode must be 'max' or 'min'")
    return spec, metric_value


def resolve_spec(label: str) -> RunSpec:
    """Return the static spec for A-C or the dynamic spec for D."""
    if label in FIXED_RUNS:
        return FIXED_RUNS[label]
    if label != "D":
        raise ValueError(f"Unknown sweep run: {label}")

    best_spec, best_metric = best_stage1_run()
    print(
        "Resolved D from the best A-C run: "
        f"{best_spec.label} (lr={best_spec.learning_rate:g}, "
        f"{DataConfig().rvt_checkpoint_selection_metric}={best_metric:.4f})",
        flush=True,
    )
    return RunSpec("D", "modelD", best_spec.learning_rate, 1e-4)


def labels_from(start_at: str) -> tuple[str, ...]:
    """Return the ordered subset beginning at the requested run label."""
    start_index = RUN_ORDER.index(start_at)
    return RUN_ORDER[start_index:]


def run_single(label: str) -> None:
    """Execute one run in-process; used by child processes."""
    spec = resolve_spec(label)
    cfg = config_for(spec)
    print(
        f"Starting model{label}: lr={spec.learning_rate:g}, "
        f"weight_decay={spec.weight_decay:g}, steps={spec.max_steps}",
        flush=True,
    )
    run_pipeline(cfg)


def launch_sweep(start_at: str, dry_run: bool) -> None:
    """Launch each run in its own Python process and keep separate log files."""
    base_cfg = DataConfig()
    python_executable = base_cfg.rvt_python_executable or sys.executable
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    for label in labels_from(start_at):
        spec_text = (
            "best A-C lr + wd=1e-4"
            if label == "D"
            else f"lr={FIXED_RUNS[label].learning_rate:g}, wd={FIXED_RUNS[label].weight_decay:g}"
        )
        log_path = logs_dir / f"model{label}.log"
        cmd = [
            str(python_executable),
            str(Path(__file__).resolve()),
            "--single-run",
            label,
        ]
        print(f"\n[{label}] {spec_text}", flush=True)
        print(f"    log: {log_path.relative_to(PROJECT_ROOT)}", flush=True)
        print(f"    cmd: {' '.join(cmd)}", flush=True)

        if dry_run:
            continue

        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise SystemExit(
                f"Run {label} failed with exit code {completed.returncode}. "
                f"See {log_path.relative_to(PROJECT_ROOT)}."
            )
        print(f"    finished model{label}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially run models A-C, then D using the best A-C learning rate."
    )
    parser.add_argument(
        "--start-at",
        choices=RUN_ORDER,
        default="A",
        help="Begin at this run label. Useful if modelA is already complete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without launching training.",
    )
    parser.add_argument(
        "--single-run",
        choices=RUN_ORDER,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single_run:
        run_single(args.single_run)
    else:
        launch_sweep(args.start_at, args.dry_run)


if __name__ == "__main__":
    main()
