"""Config-driven RVT preprocessing and training helpers."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run_rvt_preprocessing(cfg: Any) -> dict[str, int]:
    """Run RVT's Gen1 preprocessor using values from config.py."""
    raw_dir = Path(cfg.rvt_raw_data_dir)
    target_dir = Path(cfg.rvt_preprocessed_data_dir)
    raw_summary = summarize_rvt_raw_dataset(raw_dir)
    cmd = build_rvt_preprocess_command(cfg)
    _run_command("RVT preprocessing", cmd, cfg)
    processed = summarize_rvt_preprocessed_dataset(target_dir)
    processed["raw_recordings"] = raw_summary["n_bbox_files"]
    processed["raw_boxes"] = raw_summary["n_boxes"]
    return processed


def run_rvt_training(cfg: Any) -> dict[str, int | str | bool]:
    """Launch RVT training using Hydra overrides built from config.py."""
    processed_dir = Path(cfg.rvt_preprocessed_data_dir)
    processed = summarize_rvt_preprocessed_dataset(processed_dir)

    processed.update(describe_rvt_training_setup(cfg))
    cmd = build_rvt_training_command(cfg)
    _run_command("RVT training", cmd, cfg)
    checkpoint = None if cfg.rvt_dry_run else find_latest_rvt_checkpoint(cfg, allow_configured=False)
    if checkpoint is not None:
        processed["checkpoint"] = str(checkpoint)
        processed["checkpoint_dir"] = str(checkpoint.parent)
    metrics_csv = find_latest_rvt_metrics_csv(cfg)
    if metrics_csv is not None:
        processed["metrics_csv"] = str(metrics_csv)
        if bool(getattr(cfg, "rvt_generate_metric_plots", True)):
            output_dir = checkpoint.parent if checkpoint is not None else None
            metrics_plot = plot_rvt_training_metrics(cfg, metrics_csv, output_dir=output_dir)
            if metrics_plot is not None:
                processed["metrics_plot"] = str(metrics_plot)
    return processed


def describe_rvt_training_setup(cfg: Any) -> dict[str, str | bool]:
    """Return the facts that determine whether RVT trains from scratch."""
    pretrained_checkpoint = getattr(cfg, "rvt_pretrained_checkpoint_path", None)
    starts_from_random_weights = pretrained_checkpoint is None
    return {
        **describe_rvt_model_setup(cfg),
        "model_initialization": (
            "random"
            if starts_from_random_weights
            else "pretrained_local_checkpoint"
        ),
        "starts_from_random_weights": starts_from_random_weights,
        "pretrained_checkpoint": (
            None if pretrained_checkpoint is None else str(pretrained_checkpoint)
        ),
    }


def describe_rvt_model_setup(cfg: Any) -> dict[str, str | int | float | None]:
    """Return the config facts that define the RVT/S5 detector."""
    return {
        "dataset_path": str(cfg.rvt_preprocessed_data_dir),
        "dataset_type": str(cfg.rvt_dataset_name),
        "task_num_classes": int(getattr(cfg, "rvt_num_classes", 1)),
        "task_class_names": ",".join(getattr(cfg, "rvt_class_names", ("object",))),
        "model": "RVT rnndet: MaxViTRNN/S5 backbone + PAFPN + YOLOX head",
        "experiment_config": str(cfg.rvt_experiment_config),
        "event_representation": str(cfg.rvt_event_representation_name),
        "input_channels": int(cfg.rvt_input_channels),
        "sequence_length": int(cfg.rvt_sequence_length),
        "resolution_hw": "x".join(str(int(value)) for value in cfg.atis_size),
        "s5_state_dim": cfg.rvt_s5_state_dim,
        "backbone_embed_dim": cfg.rvt_model_embed_dim,
        "attention_dim_head": cfg.rvt_model_dim_head,
        "fpn_depth": cfg.rvt_model_fpn_depth,
    }


def build_rvt_preprocess_command(cfg: Any) -> list[str]:
    """Build the command that converts raw Gen1-style EBSSA data for RVT."""
    return [
        _python_executable(cfg),
        "RVT/scripts/genx/preprocess_dataset.py",
        str(cfg.rvt_raw_data_dir),
        str(cfg.rvt_preprocessed_data_dir),
        str(cfg.rvt_preprocess_representation_config),
        str(cfg.rvt_preprocess_extraction_config),
        str(cfg.rvt_preprocess_filter_config),
        "-ds",
        str(cfg.rvt_dataset_name),
        "-np",
        str(int(cfg.rvt_preprocess_num_processes)),
    ]


def build_rvt_training_command(cfg: Any) -> list[str]:
    """Build the RVT/train.py command from config.py values."""
    experiment = str(cfg.rvt_experiment_config)
    if not experiment.endswith(".yaml"):
        experiment = f"{experiment}.yaml"

    class_names = getattr(cfg, "rvt_class_names", None)
    num_classes = getattr(cfg, "rvt_num_classes", None)

    overrides = [
        "model=rnndet",
        f"dataset={cfg.rvt_dataset_name}",
        f"dataset.path={cfg.rvt_preprocessed_data_dir}",
        f"+experiment/{cfg.rvt_dataset_name}={experiment}",
        f"++dataset.num_classes={int(num_classes)}",
        f"wandb.project_name={cfg.rvt_output_root}",
        f"wandb.run_name={_hydra_string(cfg.rvt_test_name)}",
        f"wandb.group_name={cfg.rvt_log_group}",
        "wandb.log_model=false",
        f"hardware.accelerator={cfg.rvt_accelerator}",
        f"hardware.gpus={cfg.rvt_gpus}",
        f"hardware.num_workers.train={int(cfg.rvt_num_workers_train)}",
        f"hardware.num_workers.eval={int(cfg.rvt_num_workers_eval)}",
        f"batch_size.train={int(cfg.rvt_batch_size_train)}",
        f"batch_size.eval={int(cfg.rvt_batch_size_eval)}",
        f"training.max_epochs={int(cfg.rvt_max_epochs)}",
        f"training.max_steps={int(cfg.rvt_max_steps)}",
        f"training.learning_rate={float(cfg.rvt_learning_rate)}",
        f"training.weight_decay={float(cfg.rvt_weight_decay)}",
        f"training.pretrained_checkpoint={_hydra_value(cfg.rvt_pretrained_checkpoint_path)}",
        (
            "training.pretrained_reset_classification_head="
            f"{_hydra_value(bool(cfg.rvt_pretrained_reset_classification_head))}"
        ),
        f"training.precision={int(cfg.rvt_precision)}",
        f"training.lr_scheduler.use={_hydra_value(bool(cfg.rvt_use_lr_scheduler))}",
        f"training.lr_scheduler.pct_start={float(cfg.rvt_lr_scheduler_pct_start)}",
        f"training.lr_scheduler.div_factor={float(cfg.rvt_lr_scheduler_div_factor)}",
        (
            "training.lr_scheduler.final_div_factor="
            f"{float(cfg.rvt_lr_scheduler_final_div_factor)}"
        ),
        f"training.limit_train_batches={float(cfg.rvt_limit_train_batches)}",
        f"validation.limit_val_batches={float(cfg.rvt_limit_val_batches)}",
        f"validation.val_check_interval={_hydra_value(cfg.rvt_val_check_interval)}",
        f"validation.check_val_every_n_epoch={_hydra_value(cfg.rvt_check_val_every_n_epoch)}",
        f"logging.train.log_every_n_steps={int(cfg.rvt_log_every_n_steps)}",
        f"logging.train.high_dim.enable={_hydra_value(bool(cfg.rvt_enable_visual_logging))}",
        f"logging.validation.high_dim.enable={_hydra_value(bool(cfg.rvt_enable_visual_logging))}",
        f"dataset.ev_repr_name={_hydra_string(cfg.rvt_event_representation_name)}",
        f"dataset.train.sampling={cfg.rvt_train_sampling}",
        f"dataset.eval.sampling={cfg.rvt_eval_sampling}",
        f"dataset.sequence_length={int(cfg.rvt_sequence_length)}",
        f"dataset.resolution_hw={_hydra_list(cfg.atis_size)}",
        f"model.backbone.input_channels={int(cfg.rvt_input_channels)}",
    ]

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
    overrides.extend(str(item) for item in cfg.rvt_extra_train_overrides)

    return [_python_executable(cfg), "RVT/train.py", *overrides]


def summarize_rvt_raw_dataset(raw_dir: str | Path) -> dict[str, int]:
    """Count raw RVT input files and labels."""
    raw_dir = Path(raw_dir)
    counts: dict[str, int] = {
        "n_bbox_files": 0,
        "n_h5_files": 0,
        "n_boxes": 0,
        "train_recordings": 0,
        "val_recordings": 0,
        "test_recordings": 0,
    }
    for split in ("train", "val", "test"):
        split_dir = raw_dir / split
        if not split_dir.exists():
            continue
        bbox_files = sorted(split_dir.glob("*_bbox.npy"))
        h5_files = sorted(split_dir.glob("*_td.dat.h5"))
        counts[f"{split}_recordings"] = len(bbox_files)
        counts["n_bbox_files"] += len(bbox_files)
        counts["n_h5_files"] += len(h5_files)
        for bbox_file in bbox_files:
            counts["n_boxes"] += int(len(np.load(bbox_file, mmap_mode="r")))
    return counts


def summarize_rvt_preprocessed_dataset(processed_dir: str | Path) -> dict[str, int]:
    """Count RVT sequence folders produced by preprocess_dataset.py."""
    processed_dir = Path(processed_dir)
    counts: dict[str, int] = {
        "train_sequences": 0,
        "val_sequences": 0,
        "test_sequences": 0,
        "n_sequences": 0,
        "n_label_files": 0,
        "n_event_repr_files": 0,
    }
    for split in ("train", "val", "test"):
        split_dir = processed_dir / split
        if not split_dir.exists():
            continue
        seq_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
        counts[f"{split}_sequences"] = len(seq_dirs)
        counts["n_sequences"] += len(seq_dirs)
        for seq_dir in seq_dirs:
            if (seq_dir / "labels_v2" / "labels.npz").exists():
                counts["n_label_files"] += 1
            counts["n_event_repr_files"] += len(
                list((seq_dir / "event_representations_v2").glob("*/*.h5"))
            )
    return counts


def _run_command(label: str, cmd: list[str], cfg: Any) -> None:
    print(f"{label}: {shlex.join(cmd)}")
    if bool(cfg.rvt_dry_run):
        print("Dry run: command not executed.")
        return
    subprocess.run(cmd, check=True, cwd=Path.cwd(), env=_rvt_environment(cfg))


def find_latest_rvt_metrics_csv(cfg: Any) -> Path | None:
    """Return the newest Lightning CSV metrics file produced by RVT/train.py."""
    csv_root = Path(getattr(cfg, "rvt_csv_log_dir", "data/rvt_csv_logs"))
    if not csv_root.exists():
        return None
    candidates = sorted(csv_root.glob("**/metrics.csv"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def find_latest_rvt_checkpoint(
    cfg: Any,
    prefer_best: bool = True,
    allow_configured: bool = True,
) -> Path | None:
    """Return the configured or newest local RVT checkpoint.

    A configured checkpoint can be either a direct .ckpt file path or a folder
    containing checkpoints. When selecting from a folder, epoch checkpoints are
    preferred over Lightning's last_epoch checkpoint because they correspond to
    validation-selected saves.
    """
    if allow_configured:
        checkpoint = resolve_configured_rvt_checkpoint(cfg, prefer_best=prefer_best)
        if checkpoint is not None:
            return checkpoint

    root = Path.cwd()
    candidates = [
        path
        for path in root.glob("**/*.ckpt")
        if ".git" not in path.parts and path.is_file()
    ]
    if not candidates:
        return None

    return select_rvt_checkpoint(
        candidates,
        cfg=cfg,
        prefer_best=prefer_best,
        prefer_metric=False,
    )


def resolve_configured_rvt_checkpoint(cfg: Any, prefer_best: bool = True) -> Path | None:
    """Resolve rvt_checkpoint_path/rvt_checkpoint_dir from the project config."""
    configured_file_or_dir = getattr(cfg, "rvt_checkpoint_path", None)
    if configured_file_or_dir:
        checkpoint_path = Path(configured_file_or_dir)
        if checkpoint_path.is_dir():
            return find_rvt_checkpoint_in_dir(checkpoint_path, cfg=cfg, prefer_best=prefer_best)
        if checkpoint_path.suffix == ".ckpt":
            return checkpoint_path

    checkpoint_dir = getattr(cfg, "rvt_checkpoint_dir", None) or default_rvt_checkpoint_dir(cfg)
    if checkpoint_dir:
        return find_rvt_checkpoint_in_dir(Path(checkpoint_dir), cfg=cfg, prefer_best=prefer_best)

    return None


def default_rvt_checkpoint_dir(cfg: Any) -> Path | None:
    """Return the run-name-derived checkpoint folder when a named run is configured."""
    run_name = getattr(cfg, "rvt_test_name", None)
    project_name = getattr(cfg, "rvt_output_root", None)
    if run_name is None or project_name is None:
        return None
    return Path(str(project_name)) / str(run_name) / "checkpoints"


def find_rvt_checkpoint_in_dir(
    checkpoint_dir: Path,
    cfg: Any,
    prefer_best: bool = True,
) -> Path | None:
    """Pick a checkpoint from one checkpoint directory."""
    candidates = [path for path in checkpoint_dir.glob("*.ckpt") if path.is_file()]
    return None if not candidates else select_rvt_checkpoint(
        candidates,
        cfg=cfg,
        prefer_best=prefer_best,
        prefer_metric=True,
    )


def select_rvt_checkpoint(
    candidates: list[Path],
    cfg: Any,
    prefer_best: bool = True,
    prefer_metric: bool = True,
) -> Path | None:
    """Select a checkpoint by validation metric when possible, else by recency."""
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime)
    if prefer_best:
        best_candidates = [path for path in candidates if not path.name.startswith("last_")]
        if best_candidates:
            candidates = best_candidates

    metric_name = getattr(cfg, "rvt_checkpoint_selection_metric", None)
    if prefer_metric and metric_name:
        mode = str(getattr(cfg, "rvt_checkpoint_selection_mode", "max")).lower()
        scored = [
            (metric_value, path)
            for path in candidates
            for metric_value in [checkpoint_metric_value(path, str(metric_name))]
            if metric_value is not None
        ]
        if scored:
            if mode == "max":
                return max(scored, key=lambda item: (item[0], item[1].stat().st_mtime))[1]
            return min(scored, key=lambda item: (item[0], -item[1].stat().st_mtime))[1]

    return candidates[-1]


def checkpoint_metric_value(path: Path, metric_name: str) -> float | None:
    """Extract one metric value from a Lightning checkpoint filename."""
    metrics = checkpoint_filename_metrics(path)
    for name in metric_name_aliases(metric_name):
        if name in metrics:
            return metrics[name]
    return None


def checkpoint_filename_metrics(path: Path) -> dict[str, float]:
    """Parse metric fragments such as ``epoch=003-step=10-val_AP=0.21``."""
    return {
        name: float(value)
        for name, value in re.findall(
            r"([A-Za-z][A-Za-z0-9_./]*)=(-?\d+(?:\.\d+)?)",
            path.stem,
        )
    }


def metric_name_aliases(metric_name: str) -> tuple[str, ...]:
    """Allow config to use either Hydra names or filename-safe names."""
    aliases = [
        metric_name,
        metric_name.replace("/", "_"),
        metric_name.replace("_", "/"),
    ]
    return tuple(dict.fromkeys(aliases))


def plot_rvt_training_metrics(
    cfg: Any,
    metrics_csv: str | Path,
    output_dir: str | Path | None = None,
) -> Path | None:
    """Plot losses and validation AP from a Lightning CSV log."""
    metrics_csv = Path(metrics_csv)
    if not metrics_csv.exists():
        return None

    metrics = pd.read_csv(metrics_csv)
    if metrics.empty:
        return None

    if "step" not in metrics.columns:
        metrics["step"] = np.arange(len(metrics), dtype=np.int64)

    step_loss_columns = _columns_present(
        metrics,
        (
            "train/loss_step",
            "train/iou_loss_step",
            "train/conf_loss_step",
            "train/cls_loss_step",
            "train/l1_loss_step",
        ),
    )
    epoch_loss_columns = _columns_present(
        metrics,
        (
            "train/loss_epoch",
            "train/iou_loss_epoch",
            "train/conf_loss_epoch",
            "train/cls_loss_epoch",
            "train/l1_loss_epoch",
        ),
    )
    val_ap_columns = _columns_present(
        metrics,
        (
            "val/AP",
            "val/AP_50",
            "val/AP_75",
            "val/AP_S",
            "val/AP_M",
            "val/AP_L",
        ),
    )
    lr_columns = _columns_present(metrics, ("learning_rate",))

    plot_groups = [
        ("Training Losses per Step", step_loss_columns, "Optimizer step"),
        ("Training Losses per Epoch", epoch_loss_columns, "Epoch"),
        ("Validation AP Metrics", val_ap_columns, "Epoch"),
        ("Learning Rate", lr_columns, "Optimizer step"),
    ]
    plot_groups = [group for group in plot_groups if group[1]]
    if not plot_groups:
        return None

    plots_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(getattr(cfg, "rvt_metric_plots_dir", "outputs/rvt_training_plots"))
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_name = "training_metrics.png" if output_dir is not None else f"{metrics_csv.parent.name}_metrics.png"
    plot_path = plots_dir / plot_name

    nrows = len(plot_groups)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(11, max(3.6, 3.2 * nrows)),
        tight_layout=True,
    )
    axes = np.atleast_1d(axes)

    for axis, (title, columns, xlabel) in zip(axes, plot_groups):
        prefer_epoch = xlabel == "Epoch" and "epoch" in metrics.columns
        x_column = "epoch" if prefer_epoch else "step"
        for column in columns:
            rows = metrics[[x_column, column]].copy()
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
            rows[x_column] = pd.to_numeric(rows[x_column], errors="coerce")
            rows = rows.dropna()
            if rows.empty:
                continue
            linewidth = 2.2 if column == "val/AP" else 1.4
            axis.plot(rows[x_column], rows[column], linewidth=linewidth, label=column)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def _rvt_environment(cfg: Any) -> dict[str, str]:
    env = os.environ.copy()
    cwd = str(Path.cwd())
    env["PYTHONPATH"] = cwd if not env.get("PYTHONPATH") else f"{cwd}{os.pathsep}{env['PYTHONPATH']}"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env["WANDB_MODE"] = "disabled"
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("RVT_CSV_LOG_DIR", str(Path(getattr(cfg, "rvt_csv_log_dir", "data/rvt_csv_logs"))))
    return env


def _python_executable(cfg: Any) -> str:
    configured = getattr(cfg, "rvt_python_executable", None)
    return str(configured) if configured else sys.executable


def _append_if_not_none(overrides: list[str], key: str, value: Any) -> None:
    if value is not None:
        overrides.append(f"{key}={_hydra_value(value)}")


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _hydra_string(value: Any) -> str:
    text = str(value)
    if any(char in text for char in ("=", " ", ",", "[", "]", "{", "}")):
        return "'" + text.replace("'", "\\'") + "'"
    return text


def _hydra_list(values: Iterable[int]) -> str:
    return "[" + ",".join(str(int(value)) for value in values) + "]"


def _hydra_string_list(values: Iterable[Any]) -> str:
    return "[" + ",".join(_hydra_string(value) for value in values) + "]"


def _columns_present(metrics: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [
        column
        for column in candidates
        if column in metrics.columns
        and pd.to_numeric(metrics[column], errors="coerce").notna().any()
    ]
