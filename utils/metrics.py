from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import time

import motmetrics as mm
import numpy as np

from core.base_tracker import Track


# motmetrics currently calls np.asfarray, removed in NumPy 2.0.
if not hasattr(np, "asfarray"):
    def _asfarray_compat(values, dtype=np.float64):
        return np.asarray(values, dtype=dtype)

    np.asfarray = _asfarray_compat


class MetricsError(RuntimeError):
    pass


@dataclass(slots=True)
class _LatencyFrameRecord:
    frame_index: int
    stage_durations_ms: dict[str, float]
    total_duration_ms: float
    metadata: dict[str, Any]


class LatencyTimer:
    def __init__(self, stage_names: Sequence[str] | None = None) -> None:
        self._default_stage_names = tuple(stage_names or ())
        self._frame_records: list[_LatencyFrameRecord] = []
        self._current_frame_index: int | None = None
        self._current_stage_durations_ms: dict[str, float] = {}
        self._current_frame_started_at: float | None = None

    def reset(self) -> None:
        self._frame_records.clear()
        self._current_frame_index = None
        self._current_stage_durations_ms = {}
        self._current_frame_started_at = None

    def start_frame(self, frame_index: int) -> None:
        if self._current_frame_index is not None:
            raise MetricsError("Cannot start a new frame before ending the previous frame")

        self._current_frame_index = int(frame_index)
        self._current_stage_durations_ms = {name: 0.0 for name in self._default_stage_names}
        self._current_frame_started_at = time.perf_counter()

    def end_frame(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._current_frame_index is None or self._current_frame_started_at is None:
            raise MetricsError("Cannot end frame because no active frame is started")

        total_duration_ms = (time.perf_counter() - self._current_frame_started_at) * 1000.0
        metadata_payload = dict(metadata or {})

        record = _LatencyFrameRecord(
            frame_index=self._current_frame_index,
            stage_durations_ms=dict(self._current_stage_durations_ms),
            total_duration_ms=float(total_duration_ms),
            metadata=metadata_payload,
        )
        self._frame_records.append(record)

        payload = {
            "frame_index": int(record.frame_index),
            "stage_durations_ms": dict(record.stage_durations_ms),
            "total_duration_ms": float(record.total_duration_ms),
            "metadata": dict(record.metadata),
        }

        self._current_frame_index = None
        self._current_stage_durations_ms = {}
        self._current_frame_started_at = None
        return payload

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[None]:
        if self._current_frame_index is None:
            raise MetricsError("Cannot measure stage because no active frame is started")

        started_at = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self.add_stage_duration(stage_name, elapsed_ms)

    def add_stage_duration(self, stage_name: str, duration_ms: float) -> None:
        if self._current_frame_index is None:
            raise MetricsError("Cannot add stage duration because no active frame is started")

        key = str(stage_name)
        value = float(duration_ms)
        self._current_stage_durations_ms[key] = self._current_stage_durations_ms.get(key, 0.0) + value

    def frame_records(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for record in self._frame_records:
            payload.append(
                {
                    "frame_index": int(record.frame_index),
                    "stage_durations_ms": dict(record.stage_durations_ms),
                    "total_duration_ms": float(record.total_duration_ms),
                    "metadata": dict(record.metadata),
                }
            )
        return payload

    def average_stage_durations_ms(self) -> dict[str, float]:
        if not self._frame_records:
            return {}

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for record in self._frame_records:
            for stage_name, duration_ms in record.stage_durations_ms.items():
                totals[stage_name] = totals.get(stage_name, 0.0) + float(duration_ms)
                counts[stage_name] = counts.get(stage_name, 0) + 1

        averages: dict[str, float] = {}
        for stage_name, total_ms in totals.items():
            count = counts.get(stage_name, 0)
            averages[stage_name] = (total_ms / float(count)) if count > 0 else 0.0
        return averages

    def average_total_duration_ms(self) -> float:
        if not self._frame_records:
            return 0.0
        total_ms = sum(record.total_duration_ms for record in self._frame_records)
        return float(total_ms) / float(len(self._frame_records))


def compute_mot_id_metrics(
    ground_truth_by_frame: Mapping[int, Sequence[Track]],
    predictions_by_frame: Mapping[int, Sequence[Track]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    if iou_threshold <= 0.0 or iou_threshold >= 1.0:
        raise MetricsError("iou_threshold must be in the range (0, 1)")

    accumulator = mm.MOTAccumulator(auto_id=False)
    frame_ids = sorted(set(ground_truth_by_frame.keys()) | set(predictions_by_frame.keys()))

    max_iou_distance = 1.0 - float(iou_threshold)

    for frame_id in frame_ids:
        gt_tracks = list(ground_truth_by_frame.get(frame_id, []))
        pred_tracks = list(predictions_by_frame.get(frame_id, []))

        gt_ids = [track.track_id for track in gt_tracks]
        pred_ids = [track.track_id for track in pred_tracks]

        if gt_tracks and pred_tracks:
            gt_boxes_xywh = np.asarray([_xyxy_to_xywh(track.bbox_xyxy) for track in gt_tracks], dtype=np.float64)
            pred_boxes_xywh = np.asarray([_xyxy_to_xywh(track.bbox_xyxy) for track in pred_tracks], dtype=np.float64)
            distance_matrix = mm.distances.iou_matrix(
                gt_boxes_xywh,
                pred_boxes_xywh,
                max_iou=max_iou_distance,
            )
        else:
            distance_matrix = np.empty((len(gt_tracks), len(pred_tracks)), dtype=np.float64)

        accumulator.update(gt_ids, pred_ids, distance_matrix, frameid=frame_id)

    metrics_handler = mm.metrics.create()
    summary = metrics_handler.compute(
        accumulator,
        metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses", "num_frames"],
        name="stage1",
    )

    row = summary.loc["stage1"]
    return {
        "MOTA": float(row["mota"] * 100.0),
        "IDF1": float(row["idf1"] * 100.0),
        "ID Swaps": float(row["num_switches"]),
        "False Positives": float(row["num_false_positives"]),
        "Misses": float(row["num_misses"]),
        "Frames": float(row["num_frames"]),
    }


def _xyxy_to_xywh(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    return float(x1), float(y1), width, height
