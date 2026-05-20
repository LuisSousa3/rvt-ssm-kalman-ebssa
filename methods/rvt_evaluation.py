"""Post-training RVT checkpoint evaluation helpers."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import csv
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from methods.rvt_training import (
    _append_if_not_none,
    _hydra_list,
    _hydra_string,
    _hydra_string_list,
    _hydra_value,
    _python_executable,
    _rvt_environment,
    find_latest_rvt_checkpoint,
    summarize_rvt_preprocessed_dataset,
)


USE_CONFIG_MAX_TEST_SEQUENCES = object()


def run_rvt_post_training_evaluation(
    cfg: Any,
    checkpoint: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate a trained RVT checkpoint on test data and run tracking analysis."""
    processed_dir = Path(cfg.rvt_preprocessed_data_dir)
    processed = summarize_rvt_preprocessed_dataset(processed_dir)
    checkpoint = Path(checkpoint) if checkpoint is not None else find_latest_rvt_checkpoint(cfg)
    if checkpoint is None:
        raise FileNotFoundError("No RVT checkpoint found.")

    output_dir = Path(cfg.rvt_eval_output_dir) if cfg.rvt_eval_output_dir else checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "test_tracking_summary.json"

    cmd = build_rvt_post_training_eval_command(
        cfg=cfg,
        checkpoint=checkpoint,
        output_dir=output_dir,
        summary_path=summary_path,
    )
    print(f"RVT evaluation: {shlex.join(cmd)}")
    if bool(cfg.rvt_dry_run):
        print("Dry run: command not executed.")
        return {
            "checkpoint": str(checkpoint),
            "output_dir": str(output_dir),
            "test_sequences": processed["test_sequences"],
        }

    subprocess.run(cmd, check=True, cwd=Path.cwd(), env=_rvt_environment(cfg))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({"checkpoint": str(checkpoint), "output_dir": str(output_dir), "summary_json": str(summary_path)})

    if bool(getattr(cfg, "run_rvt_s5_memory_study", False)):
        study_summary = run_rvt_s5_memory_study(cfg, checkpoint=checkpoint, output_dir=output_dir)
        summary["s5_memory_study_csv"] = study_summary["csv_path"]
        summary["s5_memory_study_plot"] = study_summary["plot_path"]

    if bool(getattr(cfg, "run_rvt_sequence_length_study", False)):
        study_summary = run_rvt_sequence_length_study(cfg, checkpoint=checkpoint, output_dir=output_dir)
        summary["sequence_length_study_csv"] = study_summary["csv_path"]
        summary["sequence_length_study_plot"] = study_summary["plot_path"]
    return summary


def build_rvt_post_training_eval_command(
    cfg: Any,
    checkpoint: Path,
    output_dir: Path,
    summary_path: Path,
    state_reset_interval: int | None = None,
    sequence_length: int | None = None,
    max_test_sequences: int | None | object = USE_CONFIG_MAX_TEST_SEQUENCES,
    no_video: bool = False,
    artifact_prefix: str = "",
) -> list[str]:
    """Build the command for the standalone RVT test and tracking script."""
    if max_test_sequences is USE_CONFIG_MAX_TEST_SEQUENCES:
        max_test_sequences = cfg.rvt_eval_max_test_sequences
    return [
        _python_executable(cfg),
        "RVT/evaluate_tracking.py",
        "--output-dir",
        str(output_dir),
        "--summary-path",
        str(summary_path),
        "--iou-threshold",
        str(float(cfg.rvt_eval_iou_threshold)),
        "--video-fps",
        str(int(cfg.rvt_eval_video_fps)),
        "--video-scale",
        str(int(cfg.rvt_eval_video_scale)),
        "--kalman-association-iou",
        str(float(cfg.rvt_kalman_association_iou)),
        "--kalman-association-center-distance-scale",
        str(float(cfg.rvt_kalman_association_center_distance_scale)),
        "--kalman-birth-suppression-center-distance-scale",
        str(float(cfg.rvt_kalman_birth_suppression_center_distance_scale)),
        "--kalman-max-missed-frames",
        str(int(cfg.rvt_kalman_max_missed_frames)),
        "--kalman-min-confirmed-hits",
        str(int(cfg.rvt_kalman_min_confirmed_hits)),
        "--kalman-tentative-max-missed-frames",
        str(int(cfg.rvt_kalman_tentative_max_missed_frames)),
        "--kalman-process-noise",
        str(float(cfg.rvt_kalman_process_noise)),
        "--kalman-measurement-noise",
        str(float(cfg.rvt_kalman_measurement_noise)),
        *(["--no-video"] if no_video else []),
        *(["--artifact-prefix", artifact_prefix] if artifact_prefix else []),
        *([] if state_reset_interval is None else ["--state-reset-interval", str(int(state_reset_interval))]),
        *([] if cfg.rvt_eval_video_max_frames is None else ["--video-max-frames", str(int(cfg.rvt_eval_video_max_frames))]),
        *(["--video-all-test-sequences"] if bool(getattr(cfg, "rvt_eval_video_all_test_sequences", False)) and not no_video else []),
        *([] if cfg.rvt_eval_video_sequence_name is None else ["--video-sequence", str(cfg.rvt_eval_video_sequence_name)]),
        *([] if max_test_sequences is None else ["--max-test-sequences", str(int(max_test_sequences))]),
        *(["--no-ground-truth-video"] if not bool(cfg.rvt_eval_video_draw_ground_truth) else []),
        "--",
        *build_rvt_validation_overrides(cfg, checkpoint, sequence_length=sequence_length),
    ]


def build_rvt_validation_overrides(
    cfg: Any,
    checkpoint: Path,
    sequence_length: int | None = None,
) -> list[str]:
    """Build Hydra overrides matching the training architecture for evaluation."""
    experiment = str(cfg.rvt_experiment_config)
    if not experiment.endswith(".yaml"):
        experiment = f"{experiment}.yaml"

    class_names = getattr(cfg, "rvt_class_names", None)
    num_classes = getattr(cfg, "rvt_num_classes", None)

    gpus = str(cfg.rvt_gpus)
    if gpus.startswith("["):
        gpus = gpus.strip("[]").split(",", maxsplit=1)[0].strip()

    overrides = [
        f"dataset={cfg.rvt_dataset_name}",
        f"dataset.path={cfg.rvt_preprocessed_data_dir}",
        f"+experiment/{cfg.rvt_dataset_name}={experiment}",
        f"checkpoint={_hydra_string(checkpoint)}",
        "use_test_set=true",
        f"hardware.gpus={gpus}",
        f"hardware.num_workers.eval={int(cfg.rvt_num_workers_eval)}",
        f"batch_size.eval={int(cfg.rvt_batch_size_eval)}",
        f"training.precision={int(cfg.rvt_precision)}",
        f"dataset.ev_repr_name={_hydra_string(cfg.rvt_event_representation_name)}",
        f"dataset.eval.sampling={cfg.rvt_eval_sampling}",
        f"dataset.sequence_length={int(sequence_length or cfg.rvt_sequence_length)}",
        f"dataset.resolution_hw={_hydra_list(cfg.atis_size)}",
        f"model.backbone.input_channels={int(cfg.rvt_input_channels)}",
        f"model.postprocess.confidence_threshold={_hydra_value(float(cfg.rvt_eval_confidence_threshold))}",
    ]
    if num_classes is not None:
        overrides.append(f"++dataset.num_classes={int(num_classes)}")
    if class_names is not None:
        overrides.append(f"++dataset.class_names={_hydra_string_list(class_names)}")

    _append_if_not_none(overrides, "model.backbone.embed_dim", cfg.rvt_model_embed_dim)
    _append_if_not_none(
        overrides,
        "model.backbone.stage.attention.dim_head",
        cfg.rvt_model_dim_head,
    )
    _append_if_not_none(overrides, "model.fpn.depth", cfg.rvt_model_fpn_depth)
    _append_if_not_none(
        overrides,
        "model.backbone.partition_split_32",
        cfg.rvt_model_partition_split_32,
    )
    _append_if_not_none(overrides, "model.backbone.stage.s5.state_dim", cfg.rvt_s5_state_dim)
    return overrides


def run_rvt_s5_memory_study(
    cfg: Any,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Run the S5 memory reset ablation on the configured test split."""
    study_dir = output_dir / "s5_memory_study"
    study_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    max_sequences = getattr(cfg, "rvt_study_max_test_sequences", None)

    for interval in getattr(cfg, "rvt_s5_memory_reset_intervals", (None, 21, 10, 5, 1)):
        variant = "full_state" if interval is None else f"reset_{int(interval)}"
        summary_path = study_dir / f"{variant}_summary.json"
        cmd = build_rvt_post_training_eval_command(
            cfg=cfg,
            checkpoint=checkpoint,
            output_dir=study_dir,
            summary_path=summary_path,
            state_reset_interval=interval,
            sequence_length=int(cfg.rvt_sequence_length),
            max_test_sequences=max_sequences,
            no_video=True,
            artifact_prefix=variant,
        )
        summary = run_eval_command_and_load_summary(cfg, cmd, summary_path)
        rows.append(
            {
                "study": "s5_memory",
                "variant": variant,
                "state_reset_interval": "" if interval is None else int(interval),
                "sequence_length": int(cfg.rvt_sequence_length),
                **pick_study_metrics(summary),
                "summary_json": str(summary_path),
            }
        )

    return write_study_outputs(rows, study_dir, basename="s5_memory_study")


def run_rvt_sequence_length_study(
    cfg: Any,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Run a sequence-window ablation with the trained checkpoint."""
    study_dir = output_dir / "sequence_length_study"
    study_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    max_sequences = getattr(cfg, "rvt_study_max_test_sequences", None)
    reset_to_window = bool(getattr(cfg, "rvt_sequence_length_study_reset_to_window", True))

    for seq_len in getattr(cfg, "rvt_sequence_length_study_values", (1, 5, 10, 21)):
        seq_len = int(seq_len)
        reset_interval = seq_len if reset_to_window else None
        variant = f"seq_{seq_len}" if reset_to_window else f"seq_{seq_len}_carry_state"
        summary_path = study_dir / f"{variant}_summary.json"
        cmd = build_rvt_post_training_eval_command(
            cfg=cfg,
            checkpoint=checkpoint,
            output_dir=study_dir,
            summary_path=summary_path,
            state_reset_interval=reset_interval,
            sequence_length=seq_len,
            max_test_sequences=max_sequences,
            no_video=True,
            artifact_prefix=variant,
        )
        summary = run_eval_command_and_load_summary(cfg, cmd, summary_path)
        rows.append(
            {
                "study": "sequence_length",
                "variant": variant,
                "state_reset_interval": "" if reset_interval is None else reset_interval,
                "sequence_length": seq_len,
                **pick_study_metrics(summary),
                "summary_json": str(summary_path),
            }
        )

    return write_study_outputs(rows, study_dir, basename="sequence_length_study")


def run_eval_command_and_load_summary(
    cfg: Any,
    cmd: list[str],
    summary_path: Path,
) -> dict[str, object]:
    print(f"RVT study: {shlex.join(cmd)}")
    if bool(cfg.rvt_dry_run):
        return {}
    subprocess.run(cmd, check=True, cwd=Path.cwd(), env=_rvt_environment(cfg))
    return json.loads(summary_path.read_text(encoding="utf-8"))


def pick_study_metrics(summary: dict[str, object]) -> dict[str, object]:
    keys = (
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "false_negatives",
        "rvt_completeness",
        "hybrid_completeness",
        "test_sequences",
        "metric_frames",
    )
    return {key: summary.get(key, "") for key in keys}


def write_study_outputs(
    rows: list[dict[str, object]],
    study_dir: Path,
    basename: str,
) -> dict[str, str]:
    csv_path = study_dir / f"{basename}.csv"
    plot_path = study_dir / f"{basename}.png"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        plot_study_rows(rows, plot_path)
    return {"csv_path": str(csv_path), "plot_path": str(plot_path)}


def plot_study_rows(rows: list[dict[str, object]], plot_path: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    x = list(range(len(labels)))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), tight_layout=True)

    for key in ("precision", "recall", "f1"):
        axes[0].plot(x, [float_or_nan(row.get(key)) for row in rows], marker="o", label=key)
    axes[0].set_ylabel("test metric")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    for key in ("rvt_completeness", "hybrid_completeness"):
        axes[1].plot(x, [float_or_nan(row.get(key)) for row in rows], marker="o", label=key)
    axes[1].set_ylabel("tracking completeness")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def float_or_nan(value: object) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
