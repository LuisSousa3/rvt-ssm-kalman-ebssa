"""Evaluate an RVT checkpoint and compare RVT/Kalman tracking completeness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RVT.config.modifier import dynamically_modify_train_config
from RVT.models.detection.yolox.utils.boxes import postprocess
from RVT.modules.detection import Module
from RVT.utils.evaluation.prophesee.io.box_loading import BBOX_DTYPE
from RVT.utils.padding import InputPadderFromShape


POS_COLOR = np.asarray([40, 120, 255], dtype=np.float32)
NEG_COLOR = np.asarray([255, 40, 40], dtype=np.float32)


@dataclass
class SequenceResult:
    name: str
    path: Path
    timestamps_us: np.ndarray
    gt_by_time: dict[int, np.ndarray]
    pred_by_time: dict[int, np.ndarray]
    metric_labels: list[np.ndarray]
    metric_predictions: list[np.ndarray]
    height: int
    width: int


@dataclass
class KalmanTrack:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    last_time_us: int
    class_id: int = 0
    confidence: float = 1.0
    missed: int = 0
    hits: int = 1
    confirmed: bool = False

    @classmethod
    def from_box(
        cls,
        track_id: int,
        box: np.ndarray,
        time_us: int,
        measurement_noise: float,
        min_confirmed_hits: int,
    ) -> "KalmanTrack":
        cx, cy, w, h = box_to_measurement(box)
        covariance = np.diag([100.0, 100.0, 25.0, 25.0, 1000.0, 1000.0])
        covariance[:4, :4] *= max(1.0, measurement_noise)
        return cls(
            track_id=track_id,
            state=np.asarray([cx, cy, w, h, 0.0, 0.0], dtype=np.float64),
            covariance=covariance,
            last_time_us=int(time_us),
            class_id=int(box["class_id"]),
            confidence=float(box["class_confidence"]),
            confirmed=min_confirmed_hits <= 1,
        )

    def predict(self, time_us: int, process_noise: float) -> None:
        dt = max(0.0, (int(time_us) - int(self.last_time_us)) / 1_000_000.0)
        transition = np.eye(6, dtype=np.float64)
        transition[0, 4] = dt
        transition[1, 5] = dt
        q = float(process_noise)
        process = np.diag(
            [
                q * dt * dt,
                q * dt * dt,
                0.05 * q * dt,
                0.05 * q * dt,
                q * max(dt, 1e-3),
                q * max(dt, 1e-3),
            ]
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.state[2] = max(1.0, self.state[2])
        self.state[3] = max(1.0, self.state[3])
        self.last_time_us = int(time_us)

    def update(self, box: np.ndarray, measurement_noise: float, min_confirmed_hits: int) -> None:
        measurement = np.asarray(box_to_measurement(box), dtype=np.float64)
        observation = np.zeros((4, 6), dtype=np.float64)
        observation[0, 0] = 1.0
        observation[1, 1] = 1.0
        observation[2, 2] = 1.0
        observation[3, 3] = 1.0
        noise = np.diag([measurement_noise, measurement_noise, measurement_noise, measurement_noise])
        residual = measurement - observation @ self.state
        innovation = observation @ self.covariance @ observation.T + noise
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation)
        self.state = self.state + gain @ residual
        self.covariance = (np.eye(6) - gain @ observation) @ self.covariance
        self.state[2] = max(1.0, self.state[2])
        self.state[3] = max(1.0, self.state[3])
        self.class_id = int(box["class_id"])
        self.confidence = float(box["class_confidence"])
        self.missed = 0
        self.hits += 1
        self.confirmed = self.confirmed or self.hits >= min_confirmed_hits

    def to_box(self, height: int, width: int) -> np.ndarray:
        box = np.zeros((1,), dtype=BBOX_DTYPE)
        cx, cy, w, h = self.state[:4]
        x0 = float(np.clip(cx - w / 2.0, 0, max(0, width - 1)))
        y0 = float(np.clip(cy - h / 2.0, 0, max(0, height - 1)))
        x1 = float(np.clip(cx + w / 2.0, 0, max(0, width - 1)))
        y1 = float(np.clip(cy + h / 2.0, 0, max(0, height - 1)))
        box["x"] = x0
        box["y"] = y0
        box["w"] = max(1.0, x1 - x0)
        box["h"] = max(1.0, y1 - y0)
        box["class_id"] = self.class_id
        box["track_id"] = self.track_id
        box["class_confidence"] = self.confidence
        return box


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = compose_validation_config(args.overrides)
    print("------ Evaluation Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("--------------------------------------")

    device = resolve_device(config)
    module = Module.load_from_checkpoint(str(config.checkpoint), full_config=config)
    dtype = torch.float16 if str(config.training.precision).startswith("16") and device.type == "cuda" else torch.float32
    module = module.to(device=device, dtype=dtype)
    module.eval()

    test_dir = Path(config.dataset.path) / "test"
    all_sequence_paths = sorted(path for path in test_dir.iterdir() if path.is_dir())
    video_sequence_path = (
        resolve_video_sequence_path(args.video_sequence, all_sequence_paths)
        if args.video_sequence
        else None
    )
    sequence_paths = list(all_sequence_paths)
    if args.max_test_sequences is not None:
        sequence_paths = sequence_paths[: args.max_test_sequences]
    if video_sequence_path is not None and video_sequence_path not in sequence_paths:
        sequence_paths = include_requested_video_sequence(
            sequence_paths=sequence_paths,
            video_sequence_path=video_sequence_path,
            max_test_sequences=args.max_test_sequences,
        )
    if not sequence_paths:
        raise FileNotFoundError(f"No test sequences found in {test_dir}")

    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    video_result: SequenceResult | None = None
    results: list[SequenceResult] = []
    for seq_path in sequence_paths:
        print(f"Evaluating sequence: {seq_path.name}")
        result = infer_sequence(
            module=module,
            config=config,
            seq_path=seq_path,
            device=device,
            dtype=dtype,
            state_reset_interval=args.state_reset_interval,
        )
        results.append(result)
        all_labels.extend(result.metric_labels)
        all_predictions.extend(result.metric_predictions)
        if video_sequence_path is None and video_result is None:
            video_result = result
        if video_sequence_path is not None and seq_path == video_sequence_path:
            video_result = result

    metrics = precision_recall_f1(
        labels=all_labels,
        predictions=all_predictions,
        iou_threshold=float(args.iou_threshold),
    )

    assert video_result is not None
    render_results = selected_render_results(
        results=results,
        selected_result=video_result,
        render_all=bool(args.video_all_test_sequences),
    )
    if args.video_all_test_sequences:
        print(f"Video/plot sequences: all evaluated test sequences ({len(render_results)})")
    else:
        print(f"Video/plot sequence: {video_result.name}")

    tracking_sequence_summaries = []
    for result in render_results:
        tracking_summary = compare_tracking(
            result=result,
            output_dir=output_dir,
            iou_threshold=float(args.iou_threshold),
            association_iou=float(args.kalman_association_iou),
            association_center_distance_scale=float(args.kalman_association_center_distance_scale),
            birth_suppression_center_distance_scale=float(args.kalman_birth_suppression_center_distance_scale),
            max_missed_frames=int(args.kalman_max_missed_frames),
            min_confirmed_hits=int(args.kalman_min_confirmed_hits),
            tentative_max_missed_frames=int(args.kalman_tentative_max_missed_frames),
            process_noise=float(args.kalman_process_noise),
            measurement_noise=float(args.kalman_measurement_noise),
            artifact_prefix=args.artifact_prefix,
        )
        sequence_summary = {
            "sequence": result.name,
            "tracking_completeness_csv": tracking_summary["csv_path"],
            "tracking_completeness_plot": tracking_summary["plot_path"],
            "rvt_completeness": tracking_summary["rvt_completeness"],
            "hybrid_completeness": tracking_summary["hybrid_completeness"],
        }
        if not args.no_video:
            video_path = write_detection_tracking_video(
                result=result,
                output_dir=output_dir,
                hybrid_by_time=tracking_summary["hybrid_by_time"],
                fps=int(args.video_fps),
                scale=int(args.video_scale),
                max_frames=args.video_max_frames,
                draw_ground_truth=not args.no_ground_truth_video,
                artifact_prefix=args.artifact_prefix,
            )
            sequence_summary["detection_video"] = str(video_path)
        tracking_sequence_summaries.append(sequence_summary)

    primary_tracking_summary = tracking_sequence_summaries[0]

    summary = {
        **metrics,
        "checkpoint": str(config.checkpoint),
        "dataset_path": str(config.dataset.path),
        "test_sequences": len(sequence_paths),
        "metric_frames": len(all_labels),
        "video_sequence": video_result.name,
        "video_sequences": [item["sequence"] for item in tracking_sequence_summaries],
        "video_all_test_sequences": bool(args.video_all_test_sequences),
        "state_reset_interval": args.state_reset_interval,
        "sequence_length": int(config.dataset.sequence_length),
        "tracking_completeness_csv": primary_tracking_summary["tracking_completeness_csv"],
        "tracking_completeness_plot": primary_tracking_summary["tracking_completeness_plot"],
        "tracking_completeness_csvs": [
            item["tracking_completeness_csv"] for item in tracking_sequence_summaries
        ],
        "tracking_completeness_plots": [
            item["tracking_completeness_plot"] for item in tracking_sequence_summaries
        ],
        "tracking_sequence_summaries": tracking_sequence_summaries,
        "rvt_completeness": primary_tracking_summary["rvt_completeness"],
        "hybrid_completeness": primary_tracking_summary["hybrid_completeness"],
    }
    detection_videos = [
        str(item["detection_video"])
        for item in tracking_sequence_summaries
        if "detection_video" in item
    ]
    if detection_videos:
        summary["detection_video"] = detection_videos[0]
        summary["detection_videos"] = detection_videos
    metrics_csv_path = output_dir / artifact_filename(args.artifact_prefix, "test_metrics.csv")
    write_summary_csv(summary, metrics_csv_path)
    summary["test_metrics_csv"] = str(metrics_csv_path)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def selected_render_results(
    results: list[SequenceResult],
    selected_result: SequenceResult,
    render_all: bool,
) -> list[SequenceResult]:
    """Return the sequences that should receive plots/videos.

    When rendering all sequences, keep the explicitly selected sequence first so
    legacy summary fields continue to point at the same primary artifact.
    """
    if not render_all:
        return [selected_result]
    return [
        selected_result,
        *(result for result in results if result is not selected_result),
    ]


def resolve_video_sequence_path(requested: str, sequence_paths: list[Path]) -> Path:
    """Resolve a requested video sequence name or folder path inside the test split."""
    requested_path = Path(requested)
    requested_name = requested_path.name
    for seq_path in sequence_paths:
        if seq_path.name == requested_name or str(seq_path) == requested:
            return seq_path

    available = ", ".join(path.name for path in sequence_paths[:20])
    if len(sequence_paths) > 20:
        available += f", ... ({len(sequence_paths)} total)"
    raise FileNotFoundError(
        "Requested video sequence was not found in the preprocessed test split: "
        f"{requested!r}. Available test sequence folders: {available}"
    )


def include_requested_video_sequence(
    sequence_paths: list[Path],
    video_sequence_path: Path,
    max_test_sequences: int | None,
) -> list[Path]:
    """Ensure the requested sequence is evaluated so it can be plotted/rendered."""
    if max_test_sequences is None:
        return [*sequence_paths, video_sequence_path]
    if max_test_sequences <= 0:
        return [video_sequence_path]
    if len(sequence_paths) < max_test_sequences:
        return [*sequence_paths, video_sequence_path]
    return [*sequence_paths[:-1], video_sequence_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-test-sequences", type=int, default=None)
    parser.add_argument("--video-sequence", default=None)
    parser.add_argument("--video-max-frames", type=int, default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-scale", type=int, default=4)
    parser.add_argument("--video-all-test-sequences", action="store_true")
    parser.add_argument("--no-ground-truth-video", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--artifact-prefix", default="")
    parser.add_argument("--state-reset-interval", type=int, default=None)
    parser.add_argument("--kalman-association-iou", type=float, default=0.10)
    parser.add_argument("--kalman-association-center-distance-scale", type=float, default=2.0)
    parser.add_argument("--kalman-birth-suppression-center-distance-scale", type=float, default=2.0)
    parser.add_argument("--kalman-max-missed-frames", type=int, default=20)
    parser.add_argument("--kalman-min-confirmed-hits", type=int, default=2)
    parser.add_argument("--kalman-tentative-max-missed-frames", type=int, default=0)
    parser.add_argument("--kalman-process-noise", type=float, default=20.0)
    parser.add_argument("--kalman-measurement-noise", type=float, default=8.0)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.overrides and args.overrides[0] == "--":
        args.overrides = args.overrides[1:]
    return args


def compose_validation_config(overrides: list[str]):
    config_dir = Path(__file__).resolve().parent / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
        config = compose(config_name="val", overrides=overrides)
    dynamically_modify_train_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return config


def resolve_device(config) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(config.hardware.gpus)}")
    return torch.device("cpu")


def infer_sequence(
    module: Module,
    config,
    seq_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    state_reset_interval: int | None = None,
) -> SequenceResult:
    ev_dir = seq_path / "event_representations_v2" / str(config.dataset.ev_repr_name)
    ev_file = ev_dir / "event_representations.h5"
    timestamps = np.load(ev_dir / "timestamps_us.npy").astype(np.int64)
    gt_by_repr_idx = load_ground_truth_by_repr_idx(seq_path, ev_dir)
    gt_by_time = {
        int(timestamps[repr_idx]): labels
        for repr_idx, labels in gt_by_repr_idx.items()
        if 0 <= repr_idx < len(timestamps)
    }

    with h5py.File(ev_file, "r") as h5f:
        data = h5f["data"]
        num_frames, _, height, width = data.shape
        chunk_len = max(1, int(config.dataset.sequence_length))
        if state_reset_interval is not None and state_reset_interval <= 0:
            raise ValueError("state_reset_interval must be positive when set")
        padder = InputPadderFromShape(desired_hw=tuple(module.mdl_config.backbone.in_res_hw))
        pred_by_time: dict[int, np.ndarray] = {}
        metric_labels: list[np.ndarray] = []
        metric_predictions: list[np.ndarray] = []

        with torch.inference_mode():
            reset_blocks = iter_reset_blocks(
                num_frames=num_frames,
                state_reset_interval=state_reset_interval,
            )
            for block_start, block_stop in reset_blocks:
                states = None
                for start in range(block_start, block_stop, chunk_len):
                    stop = min(start + chunk_len, block_stop)
                    chunk = np.asarray(data[start:stop])
                    ev_tensor = torch.from_numpy(chunk).to(device=device, dtype=dtype)
                    ev_tensor = ev_tensor.unsqueeze(1)
                    ev_tensor = padder.pad_tensor_ev_repr(ev_tensor)
                    backbone_features, states = module.mdl.forward_backbone(
                        x=ev_tensor,
                        previous_states=states,
                        train_step=False,
                    )
                    selected_backbone_features = {
                        key: torch.cat(
                            [value[local_idx] for local_idx in range(stop - start)],
                            dim=0,
                        )
                        for key, value in backbone_features.items()
                    }
                    predictions, _ = module.mdl.forward_detect(
                        backbone_features=selected_backbone_features,
                    )
                    processed = postprocess(
                        prediction=predictions,
                        num_classes=int(module.mdl_config.head.num_classes),
                        conf_thre=float(module.mdl_config.postprocess.confidence_threshold),
                        nms_thre=float(module.mdl_config.postprocess.nms_threshold),
                    )
                    for local_idx, processed_frame in enumerate(processed):
                        repr_idx = start + local_idx
                        time_us = int(timestamps[repr_idx])
                        pred_boxes = prediction_to_boxes(
                            processed_frame,
                            time_us=time_us,
                            height=height,
                            width=width,
                        )
                        pred_by_time[time_us] = pred_boxes
                        if repr_idx in gt_by_repr_idx:
                            metric_labels.append(gt_by_repr_idx[repr_idx])
                            metric_predictions.append(pred_boxes)

    return SequenceResult(
        name=seq_path.name,
        path=seq_path,
        timestamps_us=timestamps,
        gt_by_time=gt_by_time,
        pred_by_time=pred_by_time,
        metric_labels=metric_labels,
        metric_predictions=metric_predictions,
        height=height,
        width=width,
    )


def iter_reset_blocks(
    num_frames: int,
    state_reset_interval: int | None,
) -> Iterable[tuple[int, int]]:
    if state_reset_interval is None:
        yield 0, num_frames
        return
    for start in range(0, num_frames, state_reset_interval):
        yield start, min(start + state_reset_interval, num_frames)


def load_ground_truth_by_repr_idx(seq_path: Path, ev_dir: Path) -> dict[int, np.ndarray]:
    labels_data = np.load(seq_path / "labels_v2" / "labels.npz")
    labels = labels_data["labels"]
    label_starts = labels_data["objframe_idx_2_label_idx"]
    objframe_to_repr = np.load(ev_dir / "objframe_idx_2_repr_idx.npy")

    by_repr_idx: dict[int, np.ndarray] = {}
    for objframe_idx, repr_idx in enumerate(objframe_to_repr):
        start = int(label_starts[objframe_idx])
        stop = int(label_starts[objframe_idx + 1]) if objframe_idx + 1 < len(label_starts) else len(labels)
        by_repr_idx[int(repr_idx)] = labels[start:stop].copy()
    return by_repr_idx


def prediction_to_boxes(
    prediction: torch.Tensor | None,
    time_us: int,
    height: int,
    width: int,
) -> np.ndarray:
    if prediction is None or prediction.numel() == 0:
        return empty_boxes()
    pred = prediction.detach().cpu().float().numpy()
    boxes = np.zeros((len(pred),), dtype=BBOX_DTYPE)
    x0 = np.clip(pred[:, 0], 0, max(0, width - 1))
    y0 = np.clip(pred[:, 1], 0, max(0, height - 1))
    x1 = np.clip(pred[:, 2], 0, max(0, width - 1))
    y1 = np.clip(pred[:, 3], 0, max(0, height - 1))
    boxes["t"] = int(time_us)
    boxes["x"] = x0
    boxes["y"] = y0
    boxes["w"] = np.maximum(1.0, x1 - x0)
    boxes["h"] = np.maximum(1.0, y1 - y0)
    boxes["class_id"] = pred[:, 6].astype(np.uint32)
    boxes["track_id"] = np.arange(len(pred), dtype=np.uint32)
    boxes["class_confidence"] = np.asarray(pred[:, 4] * pred[:, 5], dtype=np.float32)
    order = np.argsort(-boxes["class_confidence"])
    return boxes[order]


def precision_recall_f1(
    labels: list[np.ndarray],
    predictions: list[np.ndarray],
    iou_threshold: float,
) -> dict[str, float | int]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for gt_boxes, pred_boxes in zip(labels, predictions):
        matches = match_boxes(gt_boxes, pred_boxes, iou_threshold=iou_threshold)
        true_positives += len(matches)
        false_positives += max(0, len(pred_boxes) - len(matches))
        false_negatives += max(0, len(gt_boxes) - len(matches))

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
    }


def compare_tracking(
    result: SequenceResult,
    output_dir: Path,
    iou_threshold: float,
    association_iou: float,
    association_center_distance_scale: float,
    birth_suppression_center_distance_scale: float,
    max_missed_frames: int,
    min_confirmed_hits: int,
    tentative_max_missed_frames: int,
    process_noise: float,
    measurement_noise: float,
    artifact_prefix: str = "",
) -> dict[str, object]:
    timestamps = [int(t) for t in result.timestamps_us]
    hybrid_by_time = run_hybrid_kalman(
        timestamps=timestamps,
        pred_by_time=result.pred_by_time,
        height=result.height,
        width=result.width,
        association_iou=association_iou,
        association_center_distance_scale=association_center_distance_scale,
        birth_suppression_center_distance_scale=birth_suppression_center_distance_scale,
        max_missed_frames=max_missed_frames,
        min_confirmed_hits=min_confirmed_hits,
        tentative_max_missed_frames=tentative_max_missed_frames,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )

    rows = completeness_rows(
        timestamps=timestamps,
        gt_by_time=result.gt_by_time,
        rvt_by_time=result.pred_by_time,
        hybrid_by_time=hybrid_by_time,
        iou_threshold=iou_threshold,
    )
    csv_path = output_dir / artifact_filename(
        artifact_prefix,
        f"{result.name}_tracking_completeness.csv",
    )
    write_completeness_csv(rows, csv_path)
    plot_path = output_dir / artifact_filename(
        artifact_prefix,
        f"{result.name}_tracking_completeness.png",
    )
    plot_completeness(rows, plot_path)
    totals = rows[-1] if rows else {}
    return {
        "hybrid_by_time": hybrid_by_time,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "rvt_completeness": float(totals.get("rvt_cumulative_completeness", 0.0)),
        "hybrid_completeness": float(totals.get("hybrid_cumulative_completeness", 0.0)),
    }


def run_hybrid_kalman(
    timestamps: list[int],
    pred_by_time: dict[int, np.ndarray],
    height: int,
    width: int,
    association_iou: float,
    association_center_distance_scale: float,
    birth_suppression_center_distance_scale: float,
    max_missed_frames: int,
    min_confirmed_hits: int,
    tentative_max_missed_frames: int,
    process_noise: float,
    measurement_noise: float,
) -> dict[int, np.ndarray]:
    tracks: list[KalmanTrack] = []
    next_track_id = 0
    by_time: dict[int, np.ndarray] = {}

    for time in timestamps:
        detections = pred_by_time.get(time, empty_boxes())
        for track in tracks:
            track.predict(time, process_noise=process_noise)

        if tracks and len(detections) > 0:
            predicted_boxes = tracks_to_boxes(tracks, time, height=height, width=width)
            matches = match_track_boxes(
                predicted_boxes,
                detections,
                iou_threshold=association_iou,
                center_distance_scale=association_center_distance_scale,
            )
        else:
            matches = []

        matched_track_indices = {track_idx for track_idx, _ in matches}
        matched_det_indices = {det_idx for _, det_idx in matches}
        for track_idx, det_idx in matches:
            tracks[track_idx].update(
                detections[det_idx],
                measurement_noise=measurement_noise,
                min_confirmed_hits=min_confirmed_hits,
            )

        for idx, track in enumerate(tracks):
            if idx not in matched_track_indices:
                track.missed += 1
        tracks = [
            track
            for track in tracks
            if track.missed <= (max_missed_frames if track.confirmed else tentative_max_missed_frames)
        ]

        birth_anchors = tracks_to_boxes(tracks, time, height=height, width=width)
        for det_idx, detection in enumerate(detections):
            if det_idx in matched_det_indices:
                continue
            detection_box = detections[det_idx : det_idx + 1]
            if boxes_near_any(
                detection_box,
                birth_anchors,
                center_distance_scale=birth_suppression_center_distance_scale,
            ):
                # Keep the detection as an anchor as well, so a chain of nearby
                # streak fragments collapses into one birth cluster instead of
                # producing several provisional tracks along the same wake.
                birth_anchors = append_boxes(birth_anchors, detection_box)
                continue
            tracks.append(
                KalmanTrack.from_box(
                    next_track_id,
                    detection,
                    time,
                    measurement_noise=measurement_noise,
                    min_confirmed_hits=min_confirmed_hits,
                )
            )
            birth_anchors = append_boxes(birth_anchors, detection_box)
            next_track_id += 1

        visible_tracks = [track for track in tracks if track.confirmed or track.missed == 0]
        by_time[time] = tracks_to_boxes(visible_tracks, time, height=height, width=width)
    return by_time


def completeness_rows(
    timestamps: list[int],
    gt_by_time: dict[int, np.ndarray],
    rvt_by_time: dict[int, np.ndarray],
    hybrid_by_time: dict[int, np.ndarray],
    iou_threshold: float,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    cumulative_gt = 0
    cumulative = {"rvt": 0, "hybrid": 0}

    for time in timestamps:
        gt = gt_by_time.get(time, empty_boxes())
        gt_count = len(gt)
        rvt_boxes = rvt_by_time.get(time, empty_boxes())
        hybrid_track_boxes = hybrid_by_time.get(time, empty_boxes())
        rvt_count = count_covered(gt, rvt_boxes, iou_threshold)
        hybrid_count = count_covered(gt, append_boxes(rvt_boxes, hybrid_track_boxes), iou_threshold)

        cumulative_gt += gt_count
        cumulative["rvt"] += rvt_count
        cumulative["hybrid"] += hybrid_count
        row = {
            "time_us": int(time),
            "time_s": float((time - timestamps[0]) / 1_000_000.0),
            "gt_count": int(gt_count),
            "rvt_covered": int(rvt_count),
            "hybrid_covered": int(hybrid_count),
            "rvt_frame_completeness": safe_div(rvt_count, gt_count),
            "hybrid_frame_completeness": safe_div(hybrid_count, gt_count),
            "rvt_cumulative_completeness": safe_div(cumulative["rvt"], cumulative_gt),
            "hybrid_cumulative_completeness": safe_div(cumulative["hybrid"], cumulative_gt),
        }
        rows.append(row)
    return rows


def write_completeness_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_completeness(rows: list[dict[str, float | int]], path: Path) -> None:
    if not rows:
        return
    times = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
    ax.plot(times, [row["rvt_cumulative_completeness"] for row in rows], label="RVT bbox only", linewidth=1.8)
    ax.plot(times, [row["hybrid_cumulative_completeness"] for row in rows], label="RVT + Kalman", linewidth=2.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cumulative completeness")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_detection_tracking_video(
    result: SequenceResult,
    output_dir: Path,
    hybrid_by_time: dict[int, np.ndarray],
    fps: int,
    scale: int,
    max_frames: int | None,
    draw_ground_truth: bool,
    artifact_prefix: str = "",
) -> Path:
    import cv2

    ev_dir = result.path / "event_representations_v2" / "stacked_histogram_dt=50_nbins=10"
    if not ev_dir.exists():
        ev_dirs = sorted((result.path / "event_representations_v2").iterdir())
        ev_dir = ev_dirs[0]
    video_path = output_dir / artifact_filename(
        artifact_prefix,
        f"{result.name}_detections_tracks.mp4",
    )
    writer = None
    wrote_any = False
    with h5py.File(ev_dir / "event_representations.h5", "r") as h5f:
        data = h5f["data"]
        frame_count = len(result.timestamps_us) if max_frames is None else min(len(result.timestamps_us), max_frames)
        for idx in range(frame_count):
            time = int(result.timestamps_us[idx])
            image = Image.fromarray(event_repr_to_rgb(np.asarray(data[idx])), mode="RGB")
            if scale > 1:
                image = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(image)
            if draw_ground_truth:
                draw_boxes(draw, scale_boxes(result.gt_by_time.get(time, empty_boxes()), scale), "GT", (0, 255, 80), 2 * scale)
            draw_boxes(draw, scale_boxes(result.pred_by_time.get(time, empty_boxes()), scale), "RVT", (255, 220, 0), 2 * scale)
            draw_boxes(draw, scale_boxes(hybrid_by_time.get(time, empty_boxes()), scale), "H", (255, 80, 220), max(1, scale))
            draw_overlay_text(draw, time, scale=scale)

            arr = np.asarray(image.convert("RGB"))
            if writer is None:
                height, width = arr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not open MP4 writer")
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            wrote_any = True
    if writer is not None:
        writer.release()
    if not wrote_any:
        raise ValueError("No frames were written to the detection/tracking video")
    return video_path


def artifact_filename(prefix: str, filename: str) -> str:
    prefix = sanitize_artifact_prefix(prefix)
    return filename if not prefix else f"{prefix}_{filename}"


def sanitize_artifact_prefix(prefix: str) -> str:
    prefix = str(prefix or "").strip().strip("_")
    allowed = []
    for char in prefix:
        allowed.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(allowed).strip("_")


def write_summary_csv(summary: dict[str, object], path: Path) -> None:
    scalar_keys = [
        key
        for key, value in summary.items()
        if isinstance(value, (str, int, float)) or value is None
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerow({key: summary.get(key) for key in scalar_keys})


def event_repr_to_rgb(event_repr: np.ndarray) -> np.ndarray:
    channels, height, width = event_repr.shape
    if channels % 2 != 0:
        raise ValueError(f"Expected an even number of event channels, got {channels}")
    hist = event_repr.reshape(2, channels // 2, height, width)
    pos = hist[0].sum(axis=0).astype(np.float32)
    neg = hist[1].sum(axis=0).astype(np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb += normalize_nonzero(pos)[..., None] * POS_COLOR
    rgb += normalize_nonzero(neg)[..., None] * NEG_COLOR
    return np.clip(rgb, 0, 255).astype(np.uint8)


def normalize_nonzero(values: np.ndarray) -> np.ndarray:
    if values.size == 0 or values.max() <= 0:
        return np.zeros_like(values, dtype=np.float32)
    nonzero = values[values > 0]
    vmax = float(np.percentile(nonzero, 95)) if nonzero.size else 1.0
    return np.clip(values / max(vmax, 1e-6), 0.0, 1.0)


def draw_boxes(draw: ImageDraw.ImageDraw, boxes: np.ndarray, prefix: str, color: tuple[int, int, int], line_width: int) -> None:
    for row in boxes:
        x0 = float(row["x"])
        y0 = float(row["y"])
        x1 = float(row["x"] + row["w"])
        y1 = float(row["y"] + row["h"])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(1, line_width))
        score = float(row["class_confidence"])
        label = f"{prefix}:{score:.2f}" if prefix == "RVT" else f"{prefix}:{int(row['track_id'])}"
        draw.rectangle([x0, max(0, y0 - 12), x0 + 54, max(12, y0)], fill=(0, 0, 0))
        draw.text((x0 + 2, max(0, y0 - 11)), label, fill=color)


def draw_overlay_text(draw: ImageDraw.ImageDraw, time_us: int, scale: int) -> None:
    text = f"t={time_us / 1_000_000:.2f}s  GT green | RVT yellow | RVT+Kalman magenta"
    width = 360 * max(1, scale // 2)
    draw.rectangle([4, 4, width, 20], fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))


def scale_boxes(boxes: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1 or len(boxes) == 0:
        return boxes
    scaled = boxes.copy()
    scaled["x"] *= scale
    scaled["y"] *= scale
    scaled["w"] *= scale
    scaled["h"] *= scale
    return scaled


def tracks_to_boxes(tracks: Iterable[KalmanTrack], time_us: int, height: int, width: int) -> np.ndarray:
    boxes = [track.to_box(height=height, width=width) for track in tracks]
    if not boxes:
        return empty_boxes()
    out = np.concatenate(boxes)
    out["t"] = int(time_us)
    return out


def match_boxes(gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_threshold: float) -> list[tuple[int, int]]:
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    ious = box_iou_matrix(gt_boxes, pred_boxes)
    candidates: list[tuple[float, int, int]] = []
    for gt_idx in range(len(gt_boxes)):
        for pred_idx in range(len(pred_boxes)):
            if int(gt_boxes[gt_idx]["class_id"]) != int(pred_boxes[pred_idx]["class_id"]):
                continue
            iou = float(ious[gt_idx, pred_idx])
            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        matches.append((gt_idx, pred_idx))
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
    return matches


def match_track_boxes(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    iou_threshold: float,
    center_distance_scale: float,
) -> list[tuple[int, int]]:
    """Associate tracks with detections using IoU plus a fast-motion fallback.

    IoU is still preferred, but point-like RSOs can move farther than their box
    width between frames. In that regime, center distance is a better signal
    than overlap alone and prevents the tracker from spawning a fresh identity
    simply because the old and new boxes no longer touch.
    """
    if len(track_boxes) == 0 or len(detection_boxes) == 0:
        return []

    ious = box_iou_matrix(track_boxes, detection_boxes)
    track_centers = box_centers(track_boxes)
    detection_centers = box_centers(detection_boxes)
    center_dists = np.linalg.norm(track_centers[:, None, :] - detection_centers[None, :, :], axis=2)
    track_diags = box_diagonals(track_boxes)[:, None]
    detection_diags = box_diagonals(detection_boxes)[None, :]
    scale = np.maximum(1.0, np.maximum(track_diags, detection_diags))
    normalized_dists = center_dists / scale

    candidates: list[tuple[float, int, int]] = []
    for track_idx in range(len(track_boxes)):
        for det_idx in range(len(detection_boxes)):
            if int(track_boxes[track_idx]["class_id"]) != int(detection_boxes[det_idx]["class_id"]):
                continue
            iou = float(ious[track_idx, det_idx])
            normalized_dist = float(normalized_dists[track_idx, det_idx])
            if iou < iou_threshold and normalized_dist > center_distance_scale:
                continue
            # Prefer overlap when it exists, then the nearest plausible center.
            distance_bonus = max(0.0, center_distance_scale - normalized_dist)
            score = iou + distance_bonus / max(center_distance_scale, 1e-6)
            candidates.append((score, track_idx, det_idx))

    candidates.sort(reverse=True)
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, track_idx, det_idx in candidates:
        if track_idx in used_tracks or det_idx in used_detections:
            continue
        matches.append((track_idx, det_idx))
        used_tracks.add(track_idx)
        used_detections.add(det_idx)
    return matches


def box_centers(boxes: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            boxes["x"].astype(np.float64) + boxes["w"].astype(np.float64) / 2.0,
            boxes["y"].astype(np.float64) + boxes["h"].astype(np.float64) / 2.0,
        ],
        axis=1,
    )


def box_diagonals(boxes: np.ndarray) -> np.ndarray:
    return np.hypot(boxes["w"].astype(np.float64), boxes["h"].astype(np.float64))


def boxes_near_any(
    candidate_boxes: np.ndarray,
    reference_boxes: np.ndarray,
    center_distance_scale: float,
) -> bool:
    if len(candidate_boxes) == 0 or len(reference_boxes) == 0 or center_distance_scale <= 0:
        return False
    candidate_centers = box_centers(candidate_boxes)
    reference_centers = box_centers(reference_boxes)
    center_dists = np.linalg.norm(candidate_centers[:, None, :] - reference_centers[None, :, :], axis=2)
    candidate_diags = box_diagonals(candidate_boxes)[:, None]
    reference_diags = box_diagonals(reference_boxes)[None, :]
    scale = np.maximum(1.0, np.maximum(candidate_diags, reference_diags))

    class_match = candidate_boxes["class_id"][:, None] == reference_boxes["class_id"][None, :]
    normalized_dists = center_dists / scale
    return bool(np.any(class_match & (normalized_dists <= center_distance_scale)))


def append_boxes(existing: np.ndarray, extra: np.ndarray) -> np.ndarray:
    if len(existing) == 0:
        return extra.copy()
    return np.concatenate([existing, extra])


def count_covered(gt_boxes: np.ndarray, candidate_boxes: np.ndarray, iou_threshold: float) -> int:
    return len(match_boxes(gt_boxes, candidate_boxes, iou_threshold=iou_threshold))


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax0 = a["x"].astype(np.float32)
    ay0 = a["y"].astype(np.float32)
    ax1 = ax0 + a["w"].astype(np.float32)
    ay1 = ay0 + a["h"].astype(np.float32)
    bx0 = b["x"].astype(np.float32)
    by0 = b["y"].astype(np.float32)
    bx1 = bx0 + b["w"].astype(np.float32)
    by1 = by0 + b["h"].astype(np.float32)

    inter_x0 = np.maximum(ax0[:, None], bx0[None, :])
    inter_y0 = np.maximum(ay0[:, None], by0[None, :])
    inter_x1 = np.minimum(ax1[:, None], bx1[None, :])
    inter_y1 = np.minimum(ay1[:, None], by1[None, :])
    inter_w = np.maximum(0.0, inter_x1 - inter_x0)
    inter_h = np.maximum(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    area_a = np.maximum(0.0, ax1 - ax0) * np.maximum(0.0, ay1 - ay0)
    area_b = np.maximum(0.0, bx1 - bx0) * np.maximum(0.0, by1 - by0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def box_to_measurement(box: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(box["x"] + box["w"] / 2.0),
        float(box["y"] + box["h"] / 2.0),
        float(box["w"]),
        float(box["h"]),
    )


def empty_boxes() -> np.ndarray:
    return np.empty((0,), dtype=BBOX_DTYPE)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.nan
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
