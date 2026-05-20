"""Function-based EBSSA to GEN1/RVT conversion utilities.

The intent of this module is to keep each operation small enough that it can be
called explicitly from ``main.py``:

    discover -> load -> summarize -> labels -> split -> frames -> write

The frame NPZ writer is useful for quick visualization and sanity checks. The
raw RVT writer produces the ``*_td.dat.h5`` + ``*_bbox.npy`` split layout that
RVT's ``scripts/genx/preprocess_dataset.py`` expects before training.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)

ATIS_SIZE = (240, 304)
DAVIS_SIZE = (240, 304)

R_MILLISECOND = 1_000
R_SECOND = 1_000_000

EVENT_DTYPE = np.dtype(
    [
        ("x", np.int32),
        ("y", np.int32),
        ("ts", np.uint64),
        ("p", np.int16),
    ]
)

OBJ_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("id", np.int32),
        ("ts", np.uint64),
    ]
)

# RVT/ObjectLabelFactory uses the named fields t, x, y, w, h, class_id,
# class_confidence. track_id is retained for diagnostics and visualization.
BBOX_DTYPE = np.dtype(
    {
        "names": [
            "t",
            "x",
            "y",
            "w",
            "h",
            "class_id",
            "track_id",
            "class_confidence",
        ],
        "formats": ["<i8", "<f4", "<f4", "<f4", "<f4", "<u4", "<u4", "<f4"],
        "offsets": [0, 8, 12, 16, 20, 24, 28, 32],
        "itemsize": 40,
    }
)


@dataclass
class MATRecording:
    """One parsed EBSSA recording."""

    name: str
    path: Path
    height: int
    width: int
    raw_height: int
    raw_width: int
    n_obj: int
    events: np.ndarray
    obj_labels: np.ndarray

    @property
    def has_labels(self) -> bool:
        return self.n_obj > 0 and len(self.obj_labels) > 0

    @property
    def duration_us(self) -> int:
        if len(self.events) == 0:
            return 0
        return int(self.events["ts"].max() - self.events["ts"].min())


@dataclass
class LabeledRecording:
    """Recording after point annotations have been converted to boxes."""

    recording: MATRecording
    boxes: np.ndarray

    @property
    def name(self) -> str:
        return self.recording.name

    @property
    def has_labels(self) -> bool:
        return len(self.boxes) > 0


@dataclass
class PlannedRecording:
    """Recording with an assigned train/val/test split."""

    labeled: LabeledRecording
    split: str
    video_index: int

    @property
    def name(self) -> str:
        return self.labeled.name


@dataclass
class Segment:
    """Fixed-duration recording slice."""

    recording_name: str
    split: str
    video_index: int
    segment_index: int
    height: int
    width: int
    t_start_us: int
    t_end_us: int
    events: np.ndarray
    boxes: np.ndarray


@dataclass
class Frame:
    """One generated event frame."""

    recording_name: str
    split: str
    video_index: int
    segment_index: int
    frame_index: int
    time_us: int
    event_hist: np.ndarray
    labels: np.ndarray


def discover_mat_files(mat_data_dir: str | Path, limit: Optional[int] = None) -> List[Path]:
    """Return sorted EBSSA ``.mat`` files from a directory."""
    mat_dir = Path(mat_data_dir)
    mat_paths = sorted(mat_dir.glob("*.mat"))
    if limit is not None:
        mat_paths = mat_paths[: int(limit)]
    return mat_paths


def load_mat_recordings(
    mat_paths: Sequence[Path],
    atis_size: Tuple[int, int] = ATIS_SIZE,
    davis_size: Tuple[int, int] = DAVIS_SIZE,
    use_mat_sensor_size: bool = False,
) -> List[MATRecording]:
    """Load all MAT files into structured arrays."""
    recordings: List[MATRecording] = []
    for mat_path in mat_paths:
        recordings.append(
            load_ebssa_mat(
                mat_path=mat_path,
                atis_size=atis_size,
                davis_size=davis_size,
                use_mat_sensor_size=use_mat_sensor_size,
            )
        )
    return recordings


def load_ebssa_mat(
    mat_path: str | Path,
    atis_size: Tuple[int, int] = ATIS_SIZE,
    davis_size: Tuple[int, int] = DAVIS_SIZE,
    use_mat_sensor_size: bool = False,
) -> MATRecording:
    """Load one EBSSA MAT file.

    By default the output canvas is inferred from the filename and set to the
    GEN1-compatible size 240x304. DAVIS files can have a raw 180x240 sensor, but
    keeping the output canvas fixed prevents mixed-shape training batches.
    """
    mat_path = Path(mat_path)
    raw = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    raw_height, raw_width = _raw_sensor_size(raw, mat_path.stem, atis_size, davis_size)
    if use_mat_sensor_size:
        height, width = raw_height, raw_width
    else:
        height, width = _infer_canvas_size(mat_path.stem, atis_size, davis_size)

    td = _load_mat_struct(raw["TD"])
    x = np.asarray(td["x"], dtype=np.int32).reshape(-1)
    y = np.asarray(td["y"], dtype=np.int32).reshape(-1)
    ts = np.asarray(td["ts"], dtype=np.uint64).reshape(-1)
    p = np.asarray(td["p"], dtype=np.int16).reshape(-1)
    x, y = _maybe_zero_base_xy(x, y)

    if not (x.size == y.size == ts.size == p.size):
        raise ValueError(f"TD field length mismatch in {mat_path.name}")

    order = np.argsort(ts, kind="mergesort")
    events = np.empty(x.size, dtype=EVENT_DTYPE)
    events["x"] = x[order]
    events["y"] = y[order]
    events["ts"] = ts[order]
    events["p"] = p[order]

    n_obj = int(np.asarray(raw.get("nObj", 0)).item())
    obj_labels = np.empty(0, dtype=OBJ_DTYPE)
    if "Obj" in raw:
        obj_labels = _load_object_labels(raw["Obj"], mat_path.name)

    return MATRecording(
        name=mat_path.stem,
        path=mat_path,
        height=int(height),
        width=int(width),
        raw_height=int(raw_height),
        raw_width=int(raw_width),
        n_obj=n_obj,
        events=events,
        obj_labels=obj_labels,
    )


def summarize_recordings(
    recordings: Sequence[MATRecording],
    frame_duration_us: int = 50_000,
) -> Dict[str, float | int]:
    """Compute dataset-level recording statistics."""
    durations_s = [rec.duration_us / R_SECOND for rec in recordings if len(rec.events) > 0]
    n_frames = [
        int(math.ceil(max(rec.duration_us, 1) / frame_duration_us))
        for rec in recordings
        if len(rec.events) > 0
    ]
    return {
        "n_recordings": len(recordings),
        "n_labeled_recordings": sum(1 for rec in recordings if rec.has_labels),
        "n_events": int(sum(len(rec.events) for rec in recordings)),
        "n_point_labels": int(sum(len(rec.obj_labels) for rec in recordings)),
        "n_estimated_frames": int(sum(n_frames)),
        "median_duration_s": float(np.median(durations_s)) if durations_s else 0.0,
        "min_duration_s": float(np.min(durations_s)) if durations_s else 0.0,
        "max_duration_s": float(np.max(durations_s)) if durations_s else 0.0,
    }


def convert_recording_labels(
    recordings: Sequence[MATRecording],
    bbox_size: float = 22.0,
    class_id: int = 0,
    min_box_diag: Optional[float] = None,
    min_box_side: Optional[float] = None,
) -> List[LabeledRecording]:
    """Convert EBSSA point labels into RVT/PSEE-style bounding boxes."""
    labeled: List[LabeledRecording] = []
    for rec in recordings:
        boxes = point_labels_to_boxes(
            obj_labels=rec.obj_labels,
            height=rec.height,
            width=rec.width,
            bbox_size=bbox_size,
            class_id=class_id,
        )
        if min_box_diag is not None and min_box_side is not None:
            boxes = filter_boxes_by_size(
                boxes=boxes,
                min_box_diag=float(min_box_diag),
                min_box_side=float(min_box_side),
            )
        labeled.append(LabeledRecording(recording=rec, boxes=boxes))
    return labeled


def point_labels_to_boxes(
    obj_labels: np.ndarray,
    height: int,
    width: int,
    bbox_size: float = 22.0,
    class_id: int = 0,
) -> np.ndarray:
    """Wrap each EBSSA object point in a square bounding box."""
    if len(obj_labels) == 0:
        return np.empty(0, dtype=BBOX_DTYPE)

    half = float(bbox_size) * 0.5
    x0 = np.clip(obj_labels["x"] - half, 0, width - 1).astype(np.float32)
    y0 = np.clip(obj_labels["y"] - half, 0, height - 1).astype(np.float32)
    x1 = np.clip(obj_labels["x"] + half, 0, width - 1).astype(np.float32)
    y1 = np.clip(obj_labels["y"] + half, 0, height - 1).astype(np.float32)

    w = x1 - x0
    h = y1 - y0
    keep = (w > 0) & (h > 0)

    boxes = np.zeros(int(np.count_nonzero(keep)), dtype=BBOX_DTYPE)
    boxes["t"] = obj_labels["ts"][keep].astype(np.int64)
    boxes["x"] = x0[keep]
    boxes["y"] = y0[keep]
    boxes["w"] = w[keep]
    boxes["h"] = h[keep]
    boxes["class_id"] = np.uint32(class_id)
    boxes["track_id"] = obj_labels["id"][keep].astype(np.uint32)
    boxes["class_confidence"] = np.float32(1.0)
    boxes.sort(order="t")
    return boxes


def filter_boxes_by_size(
    boxes: np.ndarray,
    min_box_diag: float,
    min_box_side: float,
) -> np.ndarray:
    """Remove boxes that are too small after clipping to the image canvas."""
    if len(boxes) == 0:
        return boxes
    diag = np.sqrt(boxes["w"] ** 2 + boxes["h"] ** 2)
    keep = (diag >= min_box_diag) & (boxes["w"] >= min_box_side) & (boxes["h"] >= min_box_side)
    return boxes[keep]


def summarize_labels(
    labeled_recordings: Sequence[LabeledRecording],
    frame_duration_us: int = 50_000,
) -> Dict[str, float | int]:
    """Compute label coverage statistics after box conversion."""
    total_frames = 0
    total_labeled_frames = 0
    unique_tracks = set()
    for item in labeled_recordings:
        rec = item.recording
        boxes = item.boxes
        if len(rec.events) == 0:
            continue
        t0 = int(rec.events["ts"].min())
        duration_us = max(int(rec.events["ts"].max()) - t0 + 1, 1)
        n_frames = max(1, int(math.ceil(duration_us / frame_duration_us)))
        total_frames += n_frames

        if len(boxes) > 0:
            frame_ids = ((boxes["t"].astype(np.int64) - t0) // frame_duration_us).astype(np.int64)
            frame_ids = np.clip(frame_ids, 0, n_frames - 1)
            total_labeled_frames += int(np.unique(frame_ids).size)
            unique_tracks.update(int(x) for x in np.unique(boxes["track_id"]))

    return {
        "n_recordings": len(labeled_recordings),
        "n_labeled_recordings": sum(1 for item in labeled_recordings if item.has_labels),
        "n_boxes": int(sum(len(item.boxes) for item in labeled_recordings)),
        "n_unique_tracks": len(unique_tracks),
        "n_estimated_frames": int(total_frames),
        "n_labeled_frames": int(total_labeled_frames),
        "labeled_frame_ratio": total_labeled_frames / max(total_frames, 1),
    }


def assign_recording_splits(
    labeled_recordings: Sequence[LabeledRecording],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    include_unlabeled: bool = False,
) -> List[PlannedRecording]:
    """Assign recordings to train, val, and test without group leakage."""
    candidates = [
        item for item in labeled_recordings if include_unlabeled or item.has_labels
    ]
    if not candidates:
        return []

    groups: Dict[str, List[LabeledRecording]] = {}
    for item in candidates:
        groups.setdefault(recording_group_name(item.name), []).append(item)

    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)

    n_groups = len(group_names)
    n_train = int(round(n_groups * train_ratio))
    n_val = int(round(n_groups * val_ratio))
    n_train = min(max(n_train, 1), n_groups)
    n_val = min(max(n_val, 1 if n_groups > 2 else 0), max(n_groups - n_train, 0))

    train_groups = set(group_names[:n_train])
    val_groups = set(group_names[n_train : n_train + n_val])

    planned: List[PlannedRecording] = []
    video_index = 0
    for item in sorted(candidates, key=lambda x: x.name):
        group = recording_group_name(item.name)
        if group in train_groups:
            split = "train"
        elif group in val_groups:
            split = "val"
        else:
            split = "test"
        planned.append(PlannedRecording(labeled=item, split=split, video_index=video_index))
        video_index += 1
    return planned


def summarize_splits(planned_recordings: Sequence[PlannedRecording]) -> Dict[str, int]:
    """Count recordings per split."""
    return {
        split: sum(1 for item in planned_recordings if item.split == split)
        for split in ("train", "val", "test")
    }


def split_recordings_into_segments(
    planned_recordings: Sequence[PlannedRecording],
    segment_duration_us: int = 60 * R_SECOND,
    keep_empty_segments: bool = False,
) -> List[Segment]:
    """Split each planned recording into fixed-duration windows."""
    segments: List[Segment] = []
    for planned in planned_recordings:
        rec = planned.labeled.recording
        boxes = planned.labeled.boxes
        if len(rec.events) == 0:
            continue

        t_min = int(rec.events["ts"].min())
        t_max = int(rec.events["ts"].max())
        if len(boxes) > 0:
            t_min = min(t_min, int(boxes["t"].min()))
            t_max = max(t_max, int(boxes["t"].max()))

        seg_start = (t_min // segment_duration_us) * segment_duration_us
        seg_idx = 0
        event_ts = rec.events["ts"].astype(np.int64)
        box_ts = boxes["t"].astype(np.int64) if len(boxes) > 0 else np.empty(0, dtype=np.int64)
        while seg_start <= t_max:
            seg_end = seg_start + segment_duration_us
            ev_start = int(np.searchsorted(event_ts, seg_start, side="left"))
            ev_end = int(np.searchsorted(event_ts, seg_end, side="left"))
            seg_events = rec.events[ev_start:ev_end]
            if len(boxes) > 0:
                box_start = int(np.searchsorted(box_ts, seg_start, side="left"))
                box_end = int(np.searchsorted(box_ts, seg_end, side="left"))
                seg_boxes = boxes[box_start:box_end]
            else:
                seg_boxes = np.empty(0, dtype=BBOX_DTYPE)

            if keep_empty_segments or len(seg_events) > 0 or len(seg_boxes) > 0:
                segments.append(
                    Segment(
                        recording_name=rec.name,
                        split=planned.split,
                        video_index=planned.video_index,
                        segment_index=seg_idx,
                        height=rec.height,
                        width=rec.width,
                        t_start_us=seg_start,
                        t_end_us=seg_end,
                        events=seg_events,
                        boxes=seg_boxes,
                    )
                )

            seg_start = seg_end
            seg_idx += 1
    return segments


def summarize_segments(segments: Sequence[Segment]) -> Dict[str, int]:
    """Compute simple segment counts."""
    return {
        "n_segments": len(segments),
        "n_segments_with_labels": sum(1 for seg in segments if len(seg.boxes) > 0),
        "n_events": int(sum(len(seg.events) for seg in segments)),
        "n_boxes": int(sum(len(seg.boxes) for seg in segments)),
    }


def generate_frames_for_segments(
    segments: Iterable[Segment],
    frame_duration_us: int = 50_000,
    n_bins: int = 10,
    keep_empty_frames: bool = False,
) -> Iterator[Frame]:
    """Yield GEN1-style event histogram frames for each segment."""
    for seg in segments:
        frame_start = seg.t_start_us
        frame_idx = 0
        event_ts = seg.events["ts"].astype(np.int64) if len(seg.events) > 0 else np.empty(0, dtype=np.int64)
        box_ts = seg.boxes["t"].astype(np.int64) if len(seg.boxes) > 0 else np.empty(0, dtype=np.int64)
        while frame_start < seg.t_end_us:
            frame_end = frame_start + frame_duration_us
            ev_start = int(np.searchsorted(event_ts, frame_start, side="left"))
            ev_end = int(np.searchsorted(event_ts, frame_end, side="left"))
            events = seg.events[ev_start:ev_end]
            if len(seg.boxes) > 0:
                label_start = int(np.searchsorted(box_ts, frame_start, side="left"))
                label_end = int(np.searchsorted(box_ts, frame_end, side="left"))
                labels = seg.boxes[label_start:label_end]
            else:
                labels = np.empty(0, dtype=BBOX_DTYPE)

            if keep_empty_frames or len(events) > 0 or len(labels) > 0:
                yield Frame(
                    recording_name=seg.recording_name,
                    split=seg.split,
                    video_index=seg.video_index,
                    segment_index=seg.segment_index,
                    frame_index=frame_idx,
                    time_us=frame_start,
                    event_hist=events_to_histogram(
                        events=events,
                        height=seg.height,
                        width=seg.width,
                        n_bins=n_bins,
                        t_start_us=frame_start,
                        t_end_us=frame_end,
                    ),
                    labels=labels,
                )

            frame_start = frame_end
            frame_idx += 1


def events_to_histogram(
    events: np.ndarray,
    height: int,
    width: int,
    n_bins: int,
    t_start_us: int,
    t_end_us: int,
) -> np.ndarray:
    """Convert events into ``(2, n_bins, H, W)`` uint16 histograms."""
    hist = np.zeros((2, n_bins, height, width), dtype=np.uint32)
    if len(events) == 0:
        return hist.astype(np.uint16)

    duration_us = max(int(t_end_us) - int(t_start_us), 1)
    t_rel = events["ts"].astype(np.int64) - int(t_start_us)
    bin_idx = (t_rel * n_bins // duration_us).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.all(valid):
        logger.debug("Dropped %d out-of-bounds events", int(np.count_nonzero(~valid)))

    pol = np.where(events["p"] > 0, 0, 1).astype(np.int64)
    np.add.at(hist, (pol[valid], bin_idx[valid], y[valid], x[valid]), 1)
    return np.minimum(hist, np.iinfo(np.uint16).max).astype(np.uint16)


def labels_to_npz_dict(labels: np.ndarray, image_index: int) -> Optional[Dict[str, np.ndarray]]:
    """Convert BBOX_DTYPE labels to the torchvision-style NPZ label dict."""
    if len(labels) == 0:
        return None

    x0 = labels["x"]
    y0 = labels["y"]
    x1 = x0 + labels["w"]
    y1 = y0 + labels["h"]
    return {
        "boxes": np.stack((x0, y0, x1, y1), axis=1).astype(np.float32),
        "labels": labels["class_id"].astype(np.uint8),
        "image_id": np.full(len(labels), image_index, dtype=np.uint64),
        "area": (labels["w"] * labels["h"]).astype(np.float32),
        "iscrowd": np.zeros(len(labels), dtype=np.uint8),
        "time": labels["t"].astype(np.int64),
        "psee_labels": labels,
    }


def write_gen1_npz_dataset(
    frames: Iterable[Frame],
    output_dir: str | Path,
    metadata: Optional[Dict] = None,
) -> Dict[str, int]:
    """Write visualization-friendly GEN1-style NPZ frame directories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    counts = {
        "n_frames": 0,
        "n_labeled_frames": 0,
        "n_labels": 0,
    }

    for global_frame_idx, frame in enumerate(frames):
        seg_dir = (
            output_dir
            / frame.split
            / f"{frame.recording_name}_seg{frame.segment_index:03d}"
        )
        seg_dir.mkdir(parents=True, exist_ok=True)

        data_path = seg_dir / f"data_{frame.frame_index:05d}.npz"
        np.savez_compressed(
            data_path,
            frame=frame.event_hist,
            time=np.int64(frame.time_us),
        )

        label_dict = labels_to_npz_dict(frame.labels, image_index=global_frame_idx)
        if label_dict is not None:
            label_path = seg_dir / f"labels_{frame.frame_index:05d}.npz"
            np.savez_compressed(label_path, **label_dict)
            counts["n_labeled_frames"] += 1
            counts["n_labels"] += int(len(frame.labels))

        counts["n_frames"] += 1

    if metadata is not None:
        with open(output_dir / "config.json", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    return counts


def write_rvt_raw_gen1_dataset(
    planned_recordings: Sequence[PlannedRecording],
    output_dir: str | Path,
    metadata: Optional[Dict] = None,
) -> Dict[str, int]:
    """Write RVT preprocessor input: split folders with H5 events and BBOX npy.

    After this step, run RVT's Gen1 preprocessing script on ``output_dir`` to
    build ``labels_v2`` and ``event_representations_v2`` for training.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("write_rvt_raw_gen1_dataset requires h5py") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    counts = {"n_recordings": 0, "n_events": 0, "n_boxes": 0}
    for planned in planned_recordings:
        rec = planned.labeled.recording
        events = _filter_events_to_frame(rec.events, height=rec.height, width=rec.width)
        boxes = planned.labeled.boxes
        split_dir = output_dir / planned.split
        split_dir.mkdir(parents=True, exist_ok=True)

        stem = rec.name
        h5_path = split_dir / f"{stem}_td.dat.h5"
        bbox_path = split_dir / f"{stem}_bbox.npy"

        with h5py.File(str(h5_path), "w") as h5f:
            events_group = h5f.create_group("events")
            events_group.create_dataset("x", data=events["x"].astype(np.int16), compression="gzip")
            events_group.create_dataset("y", data=events["y"].astype(np.int16), compression="gzip")
            events_group.create_dataset("p", data=(events["p"] > 0).astype(np.int8), compression="gzip")
            events_group.create_dataset("t", data=events["ts"].astype(np.int64), compression="gzip")
            events_group.create_dataset("height", data=np.asarray(rec.height, dtype=np.int32))
            events_group.create_dataset("width", data=np.asarray(rec.width, dtype=np.int32))

        np.save(bbox_path, boxes)
        counts["n_recordings"] += 1
        counts["n_events"] += int(len(events))
        counts["n_boxes"] += int(len(boxes))

    if metadata is not None:
        with open(output_dir / "config.json", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    return counts


def build_conversion_metadata(
    frame_duration_us: int,
    n_bins: int,
    segment_duration_us: int,
    bbox_size: float,
    class_id: int,
    atis_size: Tuple[int, int],
    davis_size: Tuple[int, int],
    use_mat_sensor_size: bool,
) -> Dict:
    """Collect conversion parameters for config.json."""
    return {
        "frame_duration_us": int(frame_duration_us),
        "n_bins": int(n_bins),
        "segment_duration_us": int(segment_duration_us),
        "bbox_size": float(bbox_size),
        "class_id": int(class_id),
        "atis_size": list(atis_size),
        "davis_size": list(davis_size),
        "use_mat_sensor_size": bool(use_mat_sensor_size),
        "rvt_polarity_convention": "p=1 positive, p=0 negative/non-positive",
        "label_dtype": BBOX_DTYPE.descr,
    }


def recording_group_name(name: str) -> str:
    """Strip sensor/label suffixes so related ATIS/DAVIS passes split together."""
    suffixes = (
        "_atis_td_labelled",
        "_davis_td_labelled",
        "_td_labelled",
        "_labelled",
        "_atis",
        "_davis",
    )
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _load_mat_struct(value) -> Dict:
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if hasattr(value, "_fieldnames"):
        return {name: np.squeeze(getattr(value, name)) for name in value._fieldnames}
    if hasattr(value, "dtype") and getattr(value.dtype, "names", None):
        return {name: np.squeeze(value[name]) for name in value.dtype.names}
    raise TypeError(f"Unsupported MATLAB struct type: {type(value)}")


def _load_object_labels(obj_value, mat_name: str) -> np.ndarray:
    try:
        obj = _load_mat_struct(obj_value)
    except TypeError:
        logger.warning("Could not parse Obj struct in %s", mat_name)
        return np.empty(0, dtype=OBJ_DTYPE)

    obj_x = np.asarray(obj.get("x", []), dtype=np.float32).reshape(-1)
    obj_y = np.asarray(obj.get("y", []), dtype=np.float32).reshape(-1)
    obj_x, obj_y = _maybe_zero_base_xy(obj_x, obj_y)
    obj_id = np.asarray(obj.get("id", []), dtype=np.int32).reshape(-1)
    obj_ts = np.asarray(obj.get("ts", []), dtype=np.uint64).reshape(-1)

    if obj_ts.size == 0:
        return np.empty(0, dtype=OBJ_DTYPE)
    if not (obj_x.size == obj_y.size == obj_id.size == obj_ts.size):
        raise ValueError(f"Obj field length mismatch in {mat_name}")

    order = np.argsort(obj_ts, kind="mergesort")
    labels = np.empty(obj_ts.size, dtype=OBJ_DTYPE)
    labels["x"] = obj_x[order]
    labels["y"] = obj_y[order]
    labels["id"] = obj_id[order]
    labels["ts"] = obj_ts[order]
    return labels


def _raw_sensor_size(
    raw: Dict,
    name: str,
    atis_size: Tuple[int, int],
    davis_size: Tuple[int, int],
) -> Tuple[int, int]:
    if "xMax" in raw and "yMax" in raw:
        return int(raw["yMax"]), int(raw["xMax"])
    return _infer_canvas_size(name, atis_size, davis_size)


def _infer_canvas_size(
    name: str,
    atis_size: Tuple[int, int],
    davis_size: Tuple[int, int],
) -> Tuple[int, int]:
    lower = name.lower()
    if "davis" in lower:
        return davis_size
    if "atis" in lower:
        return atis_size
    return atis_size


def _maybe_zero_base_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert EBSSA/MATLAB 1-based pixel coordinates to Python 0-based pixels."""
    if x.size == 0 or y.size == 0:
        return x, y
    if np.nanmin(x) >= 1 and np.nanmin(y) >= 1:
        return x - 1, y - 1
    return x, y


def _filter_events_to_frame(events: np.ndarray, height: int, width: int) -> np.ndarray:
    """Keep only events that can be indexed inside the configured frame."""
    if len(events) == 0:
        return events
    x = events["x"]
    y = events["y"]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if np.all(valid):
        return events
    logger.warning("Dropped %d out-of-frame events before RVT export", int(np.count_nonzero(~valid)))
    return events[valid]
