from __future__ import annotations

from typing import Mapping, Sequence

import motmetrics as mm
import numpy as np

from core.base_tracker import Track


class MetricsError(RuntimeError):
    pass


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
