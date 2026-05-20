"""Simple full-frame visualisation for EBSSA event data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from methods.conversion import (
    BBOX_DTYPE,
    MATRecording,
    Segment,
    events_to_histogram,
)


# Convention used in every preview:
# positive events -> blue, negative events -> red, background -> black.
POS_COLOR = np.asarray([40, 120, 255], dtype=np.float32)
NEG_COLOR = np.asarray([255, 40, 40], dtype=np.float32)
BACKGROUND = np.asarray([0, 0, 0], dtype=np.float32)


@dataclass
class PreviewFrame:
    frame_index: int
    time_us: int
    event_hist: np.ndarray


def event_histogram_to_rgb(event_hist: np.ndarray) -> np.ndarray:
    """Convert a ``(2, n_bins, H, W)`` event histogram to RGB."""
    pos = event_hist[0].sum(axis=0).astype(np.float32)
    neg = event_hist[1].sum(axis=0).astype(np.float32)

    pos_norm = _normalize_nonzero(pos)
    neg_norm = _normalize_nonzero(neg)

    rgb = np.zeros((*pos.shape, 3), dtype=np.float32)
    rgb += pos_norm[..., None] * POS_COLOR
    rgb += neg_norm[..., None] * NEG_COLOR
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_recording_video(
    recording: MATRecording,
    boxes: np.ndarray,
    output_path: str | Path,
    frame_duration_us: int = 20_000,
    frame_step_us: Optional[int] = None,
    n_bins: int = 10,
    fps: int = 10,
    max_frames: Optional[int] = None,
    start_time_us: Optional[int] = None,
    end_time_us: Optional[int] = None,
    draw_labels: bool = True,
    label_time_tolerance_us: int = 100_000,
    box_line_width: int = 3,
    scale: int = 1,
) -> Path:
    """Write a full-frame MP4 or GIF preview of one recording."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = iter_recording_video_frames(
        recording=recording,
        boxes=boxes,
        frame_duration_us=frame_duration_us,
        frame_step_us=frame_step_us,
        n_bins=n_bins,
        max_frames=max_frames,
        start_time_us=start_time_us,
        end_time_us=end_time_us,
        draw_labels=draw_labels,
        label_time_tolerance_us=label_time_tolerance_us,
        box_line_width=box_line_width,
        scale=scale,
    )

    if output_path.suffix.lower() == ".gif":
        _write_gif(frames, output_path, fps=fps)
    else:
        _write_mp4(frames, output_path, fps=fps)
    return output_path


def iter_recording_video_frames(
    recording: MATRecording,
    boxes: np.ndarray,
    frame_duration_us: int = 20_000,
    frame_step_us: Optional[int] = None,
    n_bins: int = 10,
    max_frames: Optional[int] = None,
    start_time_us: Optional[int] = None,
    end_time_us: Optional[int] = None,
    draw_labels: bool = True,
    label_time_tolerance_us: int = 100_000,
    box_line_width: int = 3,
    scale: int = 1,
):
    """Yield full-frame PIL images with event pixels and bbox labels."""
    from PIL import Image, ImageDraw

    segment = recording_as_segment(
        recording=recording,
        boxes=boxes,
        start_time_us=start_time_us,
        end_time_us=end_time_us,
    )

    for frame_count, frame in enumerate(
        iter_preview_windows(
            segment=segment,
            frame_duration_us=frame_duration_us,
            frame_step_us=frame_step_us,
            n_bins=n_bins,
        )
    ):
        if max_frames is not None and frame_count >= max_frames:
            break

        image = Image.fromarray(event_histogram_to_rgb(frame.event_hist), mode="RGB")
        if scale > 1:
            image = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.NEAREST)
        if draw_labels:
            labels = labels_near_time(
                boxes=boxes,
                time_us=frame.time_us + frame_duration_us // 2,
                tolerance_us=label_time_tolerance_us,
            )
            if scale > 1:
                labels = scale_labels(labels, scale=scale)
            draw = ImageDraw.Draw(image)
            draw_boxes(draw, labels=labels, line_width=max(1, box_line_width * scale))
        draw_timestamp(ImageDraw.Draw(image), frame.time_us + frame_duration_us // 2)
        yield image


def iter_preview_windows(
    segment: Segment,
    frame_duration_us: int,
    frame_step_us: Optional[int],
    n_bins: int,
) -> Iterable[PreviewFrame]:
    """Yield event histograms over fixed full-frame time windows."""
    step_us = max(1, int(frame_step_us or frame_duration_us))

    event_ts = (
        segment.events["ts"].astype(np.int64)
        if len(segment.events) > 0
        else np.empty(0, dtype=np.int64)
    )

    frame_start = int(segment.t_start_us)
    frame_idx = 0
    while frame_start < int(segment.t_end_us):
        frame_end = frame_start + int(frame_duration_us)
        ev_start = int(np.searchsorted(event_ts, frame_start, side="left"))
        ev_end = int(np.searchsorted(event_ts, frame_end, side="left"))
        events = segment.events[ev_start:ev_end]
        yield PreviewFrame(
            frame_index=frame_idx,
            time_us=frame_start,
            event_hist=events_to_histogram(
                events=events,
                height=segment.height,
                width=segment.width,
                n_bins=n_bins,
                t_start_us=frame_start,
                t_end_us=frame_end,
            ),
        )
        frame_start += step_us
        frame_idx += 1


def labels_near_time(boxes: np.ndarray, time_us: int, tolerance_us: int) -> np.ndarray:
    """Return all track boxes with annotations near a timestamp."""
    if boxes is None or len(boxes) == 0:
        return np.empty(0, dtype=BBOX_DTYPE)

    times = boxes["t"].astype(np.int64)
    distance = np.abs(times - int(time_us))
    candidate_indices = np.flatnonzero(distance <= int(tolerance_us))
    if candidate_indices.size == 0:
        return np.empty(0, dtype=boxes.dtype)

    if "track_id" not in boxes.dtype.names:
        return boxes[candidate_indices]

    selected_indices = []
    candidates = boxes[candidate_indices]
    candidate_distance = distance[candidate_indices]
    for track_id in np.unique(candidates["track_id"]):
        track_local = np.flatnonzero(candidates["track_id"] == track_id)
        best_local = track_local[np.argmin(candidate_distance[track_local])]
        selected_indices.append(candidate_indices[best_local])
    return boxes[np.asarray(selected_indices, dtype=np.int64)]


def draw_boxes(draw, labels: np.ndarray, line_width: int = 3) -> None:
    """Draw bbox rectangles and track IDs. No centroid is drawn."""
    for row in labels:
        x0 = float(row["x"])
        y0 = float(row["y"])
        x1 = float(row["x"] + row["w"])
        y1 = float(row["y"] + row["h"])
        text = f"id={int(row['track_id'])}" if "track_id" in labels.dtype.names else ""

        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 64), width=line_width)
        if text:
            tx, ty = x0, max(0, y0 - 13)
            draw.rectangle([tx, ty, tx + 42, ty + 12], fill=(0, 0, 0))
            draw.text((tx + 1, ty), text, fill=(0, 255, 64))


def scale_labels(labels: np.ndarray, scale: int) -> np.ndarray:
    """Scale bbox coordinates for nearest-neighbor upscaled previews."""
    if labels is None or len(labels) == 0 or scale == 1:
        return labels
    scaled = labels.copy()
    scaled["x"] = scaled["x"] * scale
    scaled["y"] = scaled["y"] * scale
    scaled["w"] = scaled["w"] * scale
    scaled["h"] = scaled["h"] * scale
    return scaled


def draw_timestamp(draw, time_us: int) -> None:
    text = f"t={time_us / 1_000_000:.2f}s"
    draw.rectangle([4, 4, 76, 18], fill=(0, 0, 0))
    draw.text((7, 6), text, fill=(255, 255, 255))


def plot_recording_sample(
    recording: MATRecording,
    boxes: np.ndarray,
    frame_duration_us: int = 20_000,
    n_bins: int = 10,
    n_frames: int = 6,
    save_path: Optional[str | Path] = None,
    label_time_tolerance_us: int = 100_000,
):
    """Save a simple grid of full-frame samples."""
    import matplotlib.pyplot as plt

    segment = recording_as_segment(recording=recording, boxes=boxes)
    total_windows = max(1, int(np.ceil((segment.t_end_us - segment.t_start_us) / frame_duration_us)))
    selected = set(np.linspace(0, total_windows - 1, min(n_frames, total_windows), dtype=int))
    frames = [
        frame
        for frame in iter_preview_windows(segment, frame_duration_us, frame_duration_us, n_bins)
        if frame.frame_index in selected
    ]

    ncols = min(3, len(frames))
    nrows = int(np.ceil(len(frames) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, frame in zip(axes, frames):
        labels = labels_near_time(
            boxes=boxes,
            time_us=frame.time_us + frame_duration_us // 2,
            tolerance_us=label_time_tolerance_us,
        )
        ax.imshow(event_histogram_to_rgb(frame.event_hist), interpolation="nearest")
        ax.set_title(f"t={frame.time_us / 1_000_000:.2f}s, labels={len(labels)}")
        ax.axis("off")
        for row in labels:
            rect = plt.Rectangle(
                (row["x"], row["y"]),
                row["w"],
                row["h"],
                linewidth=1.5,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(row["x"], max(0, row["y"] - 3), f"id={int(row['track_id'])}", color="lime", fontsize=7)

    for ax in axes[len(frames) :]:
        ax.set_visible(False)

    fig.suptitle(recording.name)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_event_count_timeline(
    recording: MATRecording,
    frame_duration_us: int = 50_000,
    save_path: Optional[str | Path] = None,
):
    """Plot event counts per frame-duration window."""
    import matplotlib.pyplot as plt

    if len(recording.events) == 0:
        raise ValueError(f"No events in {recording.name}")

    t0 = int(recording.events["ts"].min())
    t1 = int(recording.events["ts"].max())
    edges = np.arange(t0, t1 + frame_duration_us, frame_duration_us)
    counts, _ = np.histogram(recording.events["ts"], bins=edges)
    times_s = (edges[:-1] - t0) / 1_000_000.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(times_s, counts, linewidth=1.1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("events")
    ax.set_title(recording.name)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def recording_as_segment(
    recording: MATRecording,
    boxes: np.ndarray,
    start_time_us: Optional[int] = None,
    end_time_us: Optional[int] = None,
) -> Segment:
    """Represent a recording or time range as one full-frame segment."""
    if len(recording.events) == 0:
        raise ValueError(f"No events in {recording.name}")

    event_start = int(recording.events["ts"].min())
    event_end = int(recording.events["ts"].max()) + 1
    t_start = event_start if start_time_us is None else max(event_start, int(start_time_us))
    t_end = event_end if end_time_us is None else min(event_end, int(end_time_us))
    if t_end <= t_start:
        raise ValueError("Preview time range is empty")

    return Segment(
        recording_name=recording.name,
        split="preview",
        video_index=0,
        segment_index=0,
        height=recording.height,
        width=recording.width,
        t_start_us=t_start,
        t_end_us=t_end,
        events=recording.events,
        boxes=boxes if boxes is not None else np.empty(0, dtype=BBOX_DTYPE),
    )


def _normalize_nonzero(values: np.ndarray) -> np.ndarray:
    if values.size == 0 or values.max() <= 0:
        return np.zeros_like(values, dtype=np.float32)
    nonzero = values[values > 0]
    vmax = float(np.percentile(nonzero, 95)) if nonzero.size else 1.0
    return np.clip(values / max(vmax, 1e-6), 0.0, 1.0)


def _write_mp4(frames: Iterable, output_path: Path, fps: int) -> None:
    import cv2

    writer = None
    wrote_any = False
    try:
        for image in frames:
            arr = np.asarray(image.convert("RGB"))
            if writer is None:
                height, width = arr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not open MP4 writer")
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            wrote_any = True
    finally:
        if writer is not None:
            writer.release()
    if not wrote_any:
        raise ValueError("No frames generated for MP4 preview")


def _write_gif(frames: Iterable, output_path: Path, fps: int) -> None:
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("No frames generated for GIF preview")
    duration_ms = max(1, int(1000 / fps))
    frame_list[0].save(
        output_path,
        save_all=True,
        append_images=frame_list[1:],
        duration=duration_ms,
        loop=0,
    )
