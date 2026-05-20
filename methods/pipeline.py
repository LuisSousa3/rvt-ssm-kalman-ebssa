"""Project-level EBSSA/RVT pipeline orchestration.

This module keeps ``main.py`` small and makes the workflow explicit:
conversion, RVT preprocessing, training, and checkpoint evaluation are separate
named stages controlled by ``DataConfig.pipeline_stages``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from methods.conversion import (
    assign_recording_splits,
    build_conversion_metadata,
    convert_recording_labels,
    discover_mat_files,
    generate_frames_for_segments,
    load_ebssa_mat,
    load_mat_recordings,
    point_labels_to_boxes,
    split_recordings_into_segments,
    summarize_labels,
    summarize_recordings,
    summarize_segments,
    summarize_splits,
    write_gen1_npz_dataset,
    write_rvt_raw_gen1_dataset,
)
from methods.plotting import make_recording_video, plot_recording_sample
from methods.rvt_evaluation import run_rvt_post_training_evaluation
from methods.rvt_training import run_rvt_preprocessing, run_rvt_training


def run_pipeline(cfg: Any) -> dict[str, dict[str, object]]:
    """Run the configured EBSSA/RVT workflow and return per-stage summaries."""
    if getattr(cfg, "preview_mat_file", None):
        return {"preview": run_preview_video(cfg)}
    if getattr(cfg, "sample_plot_mat_file", None):
        return {"sample_plot": run_sample_plot(cfg)}

    stages = tuple(cfg.pipeline_stages)
    summaries: dict[str, dict[str, object]] = {}
    print_pipeline_header(cfg, stages)

    if "convert" in stages:
        summaries["convert"] = run_conversion(cfg)
        print_stats("1. Conversion", summaries["convert"])

    if "preprocess" in stages:
        summaries["preprocess"] = run_rvt_preprocessing(cfg)
        print_stats("2. RVT preprocessing", summaries["preprocess"])

    trained_checkpoint: Path | None = None
    if "train" in stages:
        summaries["train"] = run_rvt_training(cfg)
        checkpoint = summaries["train"].get("checkpoint")
        trained_checkpoint = Path(str(checkpoint)) if checkpoint else None
        print_stats("3. RVT training", summaries["train"])

    if "evaluate" in stages:
        summaries["evaluate"] = run_rvt_post_training_evaluation(
            cfg,
            checkpoint=trained_checkpoint,
        )
        print_stats("4. RVT checkpoint evaluation", summaries["evaluate"])

    return summaries


def print_pipeline_header(cfg: Any, stages: tuple[str, ...]) -> None:
    video_scope = (
        "all evaluated test sequences"
        if getattr(cfg, "rvt_eval_video_all_test_sequences", False)
        else cfg.rvt_eval_video_sequence_name
    )
    print(
        f"Running {', '.join(stages) if stages else 'no stages'} "
        f"with data in {cfg.rvt_preprocessed_data_dir}; plots: {video_scope}."
    )


def print_stats(title: str, stats: dict[str, object]) -> None:
    print(f"{title}: {stats}")


def run_conversion(cfg: Any) -> dict[str, object]:
    """Convert EBSSA MAT recordings into the configured dataset outputs."""
    output_kind = cfg.output_kind
    mat_paths = discover_mat_files(cfg.mat_data_dir, limit=cfg.limit)
    recordings = load_mat_recordings(
        mat_paths,
        atis_size=cfg.atis_size,
        davis_size=cfg.davis_size,
        use_mat_sensor_size=cfg.use_mat_sensor_size,
    )
    labeled_recordings = convert_recording_labels(
        recordings,
        bbox_size=cfg.bbox_size,
        class_id=cfg.class_id,
        min_box_diag=cfg.min_box_diag,
        min_box_side=cfg.min_box_side,
    )

    summary: dict[str, object] = {
        "output_kind": output_kind,
        **prefix_keys("recording", summarize_recordings(recordings, cfg.frame_duration_us)),
        **prefix_keys("label", summarize_labels(labeled_recordings, cfg.frame_duration_us)),
    }
    if cfg.stats_only:
        return summary

    planned_recordings = assign_recording_splits(
        labeled_recordings,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.split_seed,
        include_unlabeled=cfg.include_unlabeled,
    )
    summary.update(prefix_keys("split", summarize_splits(planned_recordings)))

    metadata = build_conversion_metadata(
        frame_duration_us=cfg.frame_duration_us,
        n_bins=cfg.n_bins,
        segment_duration_us=cfg.segment_duration_us,
        bbox_size=cfg.bbox_size,
        class_id=cfg.class_id,
        atis_size=cfg.atis_size,
        davis_size=cfg.davis_size,
        use_mat_sensor_size=cfg.use_mat_sensor_size,
    )

    if output_kind in ("rvt-raw", "both"):
        summary.update(
            prefix_keys(
                "rvt_raw",
                write_rvt_raw_gen1_dataset(
                    planned_recordings,
                    output_dir=cfg.rvt_raw_data_dir,
                    metadata=metadata,
                ),
            )
        )

    if output_kind in ("npz", "both"):
        segments = split_recordings_into_segments(
            planned_recordings,
            segment_duration_us=cfg.segment_duration_us,
        )
        summary.update(prefix_keys("segment", summarize_segments(segments)))
        summary.update(
            prefix_keys(
                "npz",
                write_gen1_npz_dataset(
                    generate_frames_for_segments(
                        segments,
                        frame_duration_us=cfg.frame_duration_us,
                        n_bins=cfg.n_bins,
                    ),
                    output_dir=cfg.gen1_data_dir,
                    metadata=metadata,
                ),
            )
        )

    return summary


def run_preview_video(cfg: Any) -> dict[str, object]:
    """Render a label overlay video from one EBSSA MAT recording."""
    recording = load_ebssa_mat(
        cfg.preview_mat_file,
        atis_size=cfg.atis_size,
        davis_size=cfg.davis_size,
        use_mat_sensor_size=cfg.use_mat_sensor_size,
    )
    boxes = point_labels_to_boxes(
        recording.obj_labels,
        height=recording.height,
        width=recording.width,
        bbox_size=preview_bbox_size(cfg),
        class_id=cfg.class_id,
    )
    output_path = preview_output_path(cfg, recording.name)
    make_recording_video(
        recording=recording,
        boxes=boxes,
        output_path=output_path,
        frame_duration_us=cfg.preview_frame_duration_us,
        frame_step_us=cfg.preview_frame_step_us,
        n_bins=cfg.preview_n_bins,
        max_frames=cfg.preview_max_frames,
        start_time_us=cfg.preview_start_time_us,
        end_time_us=cfg.preview_end_time_us,
        fps=cfg.preview_fps,
        draw_labels=cfg.preview_draw_labels,
        label_time_tolerance_us=cfg.preview_label_time_tolerance_us,
        box_line_width=cfg.preview_box_line_width,
        scale=cfg.preview_scale,
    )
    summary = preview_summary(recording, boxes)
    summary["output_path"] = str(output_path)
    summary["playback_speed"] = preview_playback_speed(cfg)
    print_stats("Preview video", summary)
    return summary


def run_sample_plot(cfg: Any) -> dict[str, object]:
    """Write a static sample plot for one EBSSA MAT recording."""
    recording = load_ebssa_mat(
        cfg.sample_plot_mat_file,
        atis_size=cfg.atis_size,
        davis_size=cfg.davis_size,
        use_mat_sensor_size=cfg.use_mat_sensor_size,
    )
    boxes = point_labels_to_boxes(
        recording.obj_labels,
        height=recording.height,
        width=recording.width,
        bbox_size=preview_bbox_size(cfg),
        class_id=cfg.class_id,
    )
    output_path = Path(cfg.plots_dir) / f"{recording.name}_sample.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_recording_sample(
        recording=recording,
        boxes=boxes,
        frame_duration_us=cfg.preview_frame_duration_us,
        n_bins=cfg.preview_n_bins,
        save_path=output_path,
        label_time_tolerance_us=cfg.preview_label_time_tolerance_us,
    )
    summary = preview_summary(recording, boxes)
    summary["output_path"] = str(output_path)
    print_stats("Sample plot", summary)
    return summary


def prefix_keys(prefix: str, values: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def preview_output_path(cfg: Any, recording_name: str) -> Path:
    output_format = cfg.preview_output_format.strip().lower().lstrip(".")
    if cfg.preview_output_path:
        path = Path(cfg.preview_output_path)
        return path if path.suffix.lower() in {".gif", ".mp4"} else path.with_suffix(f".{output_format}")
    return Path(cfg.plots_dir) / f"{recording_name}.{output_format}"


def preview_bbox_size(cfg: Any) -> float:
    return float(cfg.bbox_size if cfg.preview_bbox_size is None else cfg.preview_bbox_size)


def preview_playback_speed(cfg: Any) -> float:
    step_us = cfg.preview_frame_step_us or cfg.preview_frame_duration_us
    return float(step_us * cfg.preview_fps / 1_000_000)


def preview_summary(recording: Any, boxes: Any) -> dict[str, object]:
    summary: dict[str, object] = {
        "recording": recording.name,
        "events": len(recording.events),
        "boxes": len(boxes),
    }
    if len(boxes) > 0:
        tracks = sorted(set(map(int, boxes["track_id"])))
        event_start = int(recording.events["ts"].min())
        first_box = int(boxes["t"].min())
        summary["tracks"] = len(tracks)
        summary["first_track_ids"] = tracks[:10]
        summary["first_label_s"] = (first_box - event_start) / 1_000_000
    return summary
