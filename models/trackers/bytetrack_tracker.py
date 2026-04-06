from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from core.base_detector import Detection
from core.base_tracker import (
    BBoxArray,
    BaseTracker,
    ClassIdArray,
    ConfidenceArray,
    EmbeddingArray,
    Track,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox_xyxy: NDArray[np.float32]
    confidence: float
    class_id: int
    embedding: EmbeddingArray | None
    missed_frames: int = 0


class ByteTrackTracker(BaseTracker):
    def __init__(self, tracker_name: str, config: Mapping[str, Any]) -> None:
        super().__init__(tracker_name=tracker_name, config=config)
        self.track_threshold = float(config.get("track_threshold", 0.5))
        self.track_low_threshold = float(config.get("track_low_threshold", 0.1))
        self.new_track_threshold = float(config.get("new_track_threshold", 0.6))
        self.match_threshold = float(config.get("match_threshold", 0.8))
        self.track_buffer = int(config.get("track_buffer", 30))
        self.min_box_area = float(config.get("min_box_area", 10.0))
        self.use_embeddings = bool(config.get("use_embeddings", False))

        association_alpha = config.get("association_alpha")
        if association_alpha is None:
            motion_weight = float(config.get("motion_weight", 0.6))
            embedding_weight = float(config.get("embedding_weight", 0.4))
            total_weight = motion_weight + embedding_weight
            if total_weight <= 0.0:
                association_alpha = 0.5
            else:
                association_alpha = motion_weight / total_weight

        self.association_alpha = float(max(0.0, min(1.0, float(association_alpha))))
        self.motion_weight = self.association_alpha
        self.embedding_weight = 1.0 - self.association_alpha

        self._match_cost_threshold = max(0.0, min(1.0, 1.0 - self.match_threshold))
        self._device = str(config.get("device") or self._auto_device())
        self._active_tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1
        self._last_frame_index = -1

    @staticmethod
    def _auto_device() -> str:
        if torch is not None and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ByteTrackTracker":
        tracker_name = str(config.get("type", "bytetrack"))
        return cls(tracker_name=tracker_name, config=config)

    @property
    def device(self) -> str:
        return self._device

    def set_device(self, device: str) -> None:
        normalized = device.strip()
        if not normalized:
            raise ValueError("Device must be a non-empty string")
        self._device = normalized

    def initialize(self) -> None:
        if self._is_initialized:
            return
        self._active_tracks.clear()
        self._next_track_id = 1
        self._last_frame_index = -1
        self._is_initialized = True

    def update(
        self,
        detections: Sequence[Detection],
        embeddings: EmbeddingArray | None = None,
        frame_index: int | None = None,
    ) -> list[Track]:
        if not self._is_initialized:
            self.initialize()

        if not detections:
            empty_bboxes = np.zeros((0, 4), dtype=np.float32)
            empty_confidences = np.zeros((0,), dtype=np.float32)
            empty_classes = np.zeros((0,), dtype=np.int32)
            return self.update_from_arrays(
                bboxes_xyxy=empty_bboxes,
                confidences=empty_confidences,
                class_ids=empty_classes,
                embeddings=embeddings,
                frame_index=frame_index,
            )

        bboxes = np.asarray([detection.bbox_xyxy for detection in detections], dtype=np.float32)
        confidences = np.asarray([detection.confidence for detection in detections], dtype=np.float32)
        class_ids = np.asarray([detection.class_id for detection in detections], dtype=np.int32)
        return self.update_from_arrays(
            bboxes_xyxy=bboxes,
            confidences=confidences,
            class_ids=class_ids,
            embeddings=embeddings,
            frame_index=frame_index,
        )

    def update_from_arrays(
        self,
        bboxes_xyxy: BBoxArray,
        confidences: ConfidenceArray,
        class_ids: ClassIdArray | None = None,
        embeddings: EmbeddingArray | None = None,
        frame_index: int | None = None,
    ) -> list[Track]:
        if not self._is_initialized:
            self.initialize()

        boxes = self._to_numpy(bboxes_xyxy, dtype=np.float32)
        scores = self._to_numpy(confidences, dtype=np.float32).reshape(-1)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("bboxes_xyxy must have shape [N, 4]")
        if scores.shape[0] != boxes.shape[0]:
            raise ValueError("confidences must have the same length as bboxes_xyxy")

        if class_ids is None:
            labels = np.zeros((boxes.shape[0],), dtype=np.int32)
        else:
            labels = self._to_numpy(class_ids, dtype=np.int32).reshape(-1)
            if labels.shape[0] != boxes.shape[0]:
                raise ValueError("class_ids must have the same length as bboxes_xyxy")

        embedding_matrix: EmbeddingArray | None
        if embeddings is None:
            embedding_matrix = None
        else:
            embedding_matrix = self._to_numpy(embeddings, dtype=np.float32)
            if embedding_matrix.ndim == 1:
                embedding_matrix = embedding_matrix.reshape(1, -1)
            if embedding_matrix.shape[0] != boxes.shape[0]:
                raise ValueError("embeddings must have shape [N, D] with N matching bboxes")

        valid_indices = self._filter_by_area(boxes, self.min_box_area)
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        labels = labels[valid_indices]
        if embedding_matrix is not None:
            embedding_matrix = embedding_matrix[valid_indices]

        high_indices = np.where(scores >= self.track_threshold)[0]
        low_indices = np.where((scores >= self.track_low_threshold) & (scores < self.track_threshold))[0]

        matched_track_ids: set[int] = set()
        used_detection_indices: set[int] = set()

        high_matches = self._match_tracks(
            track_ids=list(self._active_tracks.keys()),
            detection_indices=high_indices.tolist(),
            boxes=boxes,
            embeddings=embedding_matrix,
        )
        self._apply_matches(
            matches=high_matches,
            boxes=boxes,
            scores=scores,
            labels=labels,
            embeddings=embedding_matrix,
            matched_track_ids=matched_track_ids,
            used_detection_indices=used_detection_indices,
        )

        remaining_track_ids = [
            track_id
            for track_id in self._active_tracks
            if track_id not in matched_track_ids
        ]
        low_candidate_indices = [index for index in low_indices.tolist() if index not in used_detection_indices]

        low_matches = self._match_tracks(
            track_ids=remaining_track_ids,
            detection_indices=low_candidate_indices,
            boxes=boxes,
            embeddings=embedding_matrix,
        )
        self._apply_matches(
            matches=low_matches,
            boxes=boxes,
            scores=scores,
            labels=labels,
            embeddings=embedding_matrix,
            matched_track_ids=matched_track_ids,
            used_detection_indices=used_detection_indices,
        )

        unmatched_track_ids = [
            track_id
            for track_id in self._active_tracks
            if track_id not in matched_track_ids
        ]
        self._mark_unmatched_tracks(unmatched_track_ids)

        unmatched_high_indices = [
            index
            for index in high_indices.tolist()
            if index not in used_detection_indices and scores[index] >= self.new_track_threshold
        ]
        self._spawn_tracks(
            detection_indices=unmatched_high_indices,
            boxes=boxes,
            scores=scores,
            labels=labels,
            embeddings=embedding_matrix,
        )

        if frame_index is not None:
            self._last_frame_index = int(frame_index)

        return self._export_visible_tracks()

    def reset(self) -> None:
        self._active_tracks.clear()
        self._next_track_id = 1
        self._last_frame_index = -1
        self._is_initialized = True

    def shutdown(self) -> None:
        self._active_tracks.clear()
        self._next_track_id = 1
        self._last_frame_index = -1
        self._is_initialized = False

    @staticmethod
    def _to_numpy(values: Any, dtype: np.dtype[Any]) -> NDArray[Any]:
        if torch is not None and isinstance(values, torch.Tensor):
            return values.detach().to("cpu").numpy().astype(dtype, copy=False)
        if isinstance(values, np.ndarray):
            return values.astype(dtype, copy=False)
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def _filter_by_area(boxes: NDArray[np.float32], min_area: float) -> NDArray[np.int64]:
        widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        areas = widths * heights
        return np.where(areas >= float(min_area))[0]

    def _match_tracks(
        self,
        track_ids: list[int],
        detection_indices: list[int],
        boxes: NDArray[np.float32],
        embeddings: EmbeddingArray | None,
    ) -> list[tuple[int, int]]:
        if not track_ids or not detection_indices:
            return []

        cost_matrix = self._build_cost_matrix(
            track_ids=track_ids,
            detection_indices=detection_indices,
            boxes=boxes,
            embeddings=embeddings,
        )
        if cost_matrix.size == 0:
            return []

        if linear_sum_assignment is not None:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:  # pragma: no cover
            row_indices, col_indices = self._greedy_assignment(cost_matrix)

        matches: list[tuple[int, int]] = []

        for row_index, col_index in zip(row_indices.tolist(), col_indices.tolist()):
            pair_cost = float(cost_matrix[row_index, col_index])
            if pair_cost > self._match_cost_threshold:
                continue
            matches.append((track_ids[row_index], detection_indices[col_index]))

        return matches

    def _build_cost_matrix(
        self,
        track_ids: list[int],
        detection_indices: list[int],
        boxes: NDArray[np.float32],
        embeddings: EmbeddingArray | None,
    ) -> NDArray[np.float32]:
        costs = np.zeros((len(track_ids), len(detection_indices)), dtype=np.float32)
        for row_index, track_id in enumerate(track_ids):
            track_state = self._active_tracks[track_id]
            for column_index, detection_index in enumerate(detection_indices):
                detection_embedding = None if embeddings is None else embeddings[detection_index]
                costs[row_index, column_index] = self._association_distance(
                    track_state=track_state,
                    detection_box=boxes[detection_index],
                    detection_embedding=detection_embedding,
                )
        return costs

    def _association_distance(
        self,
        track_state: _TrackState,
        detection_box: NDArray[np.float32],
        detection_embedding: NDArray[np.float32] | None,
    ) -> float:
        iou_distance = 1.0 - self._iou(track_state.bbox_xyxy, detection_box)

        if (
            not self.use_embeddings
            or detection_embedding is None
            or track_state.embedding is None
        ):
            return float(max(0.0, min(1.0, iou_distance)))

        if track_state.embedding.shape[0] != detection_embedding.shape[0]:
            return float(max(0.0, min(1.0, iou_distance)))

        cosine_similarity = self._cosine_similarity(track_state.embedding, detection_embedding)
        cosine_distance = 1.0 - cosine_similarity

        combined_distance = (self.association_alpha * iou_distance) + (
            (1.0 - self.association_alpha) * cosine_distance
        )
        return float(max(0.0, min(1.0, combined_distance)))

    @staticmethod
    def _greedy_assignment(cost_matrix: NDArray[np.float32]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        flat_indices = np.argsort(cost_matrix, axis=None)
        used_rows: set[int] = set()
        used_columns: set[int] = set()
        matched_rows: list[int] = []
        matched_columns: list[int] = []

        for flat_index in flat_indices.tolist():
            row_index, column_index = np.unravel_index(flat_index, cost_matrix.shape)
            if row_index in used_rows or column_index in used_columns:
                continue
            used_rows.add(int(row_index))
            used_columns.add(int(column_index))
            matched_rows.append(int(row_index))
            matched_columns.append(int(column_index))

        return np.asarray(matched_rows, dtype=np.int64), np.asarray(matched_columns, dtype=np.int64)

    @staticmethod
    def _iou(box_a: NDArray[np.float32], box_b: NDArray[np.float32]) -> float:
        left = max(float(box_a[0]), float(box_b[0]))
        top = max(float(box_a[1]), float(box_b[1]))
        right = min(float(box_a[2]), float(box_b[2]))
        bottom = min(float(box_a[3]), float(box_b[3]))

        inter_w = max(0.0, right - left)
        inter_h = max(0.0, bottom - top)
        intersection = inter_w * inter_h

        area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
        area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
        union = area_a + area_b - intersection

        if union <= 0.0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def _cosine_similarity(vector_a: NDArray[np.float32], vector_b: NDArray[np.float32]) -> float:
        norm_a = float(np.linalg.norm(vector_a))
        norm_b = float(np.linalg.norm(vector_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        value = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
        return max(0.0, min(1.0, (value + 1.0) / 2.0))

    def _apply_matches(
        self,
        matches: list[tuple[int, int]],
        boxes: NDArray[np.float32],
        scores: NDArray[np.float32],
        labels: NDArray[np.int32],
        embeddings: EmbeddingArray | None,
        matched_track_ids: set[int],
        used_detection_indices: set[int],
    ) -> None:
        for track_id, detection_index in matches:
            state = self._active_tracks[track_id]
            state.bbox_xyxy = boxes[detection_index].copy()
            state.confidence = float(scores[detection_index])
            state.class_id = int(labels[detection_index])
            state.embedding = (
                None
                if embeddings is None
                else np.asarray(embeddings[detection_index], dtype=np.float32).copy()
            )
            state.missed_frames = 0
            matched_track_ids.add(track_id)
            used_detection_indices.add(detection_index)

    def _mark_unmatched_tracks(self, unmatched_track_ids: list[int]) -> None:
        for track_id in unmatched_track_ids:
            state = self._active_tracks[track_id]
            state.missed_frames += 1

        stale_track_ids = [
            track_id
            for track_id, state in self._active_tracks.items()
            if state.missed_frames > self.track_buffer
        ]
        for track_id in stale_track_ids:
            del self._active_tracks[track_id]

    def _spawn_tracks(
        self,
        detection_indices: list[int],
        boxes: NDArray[np.float32],
        scores: NDArray[np.float32],
        labels: NDArray[np.int32],
        embeddings: EmbeddingArray | None,
    ) -> None:
        for detection_index in detection_indices:
            track_id = self._next_track_id
            self._next_track_id += 1

            self._active_tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox_xyxy=boxes[detection_index].copy(),
                confidence=float(scores[detection_index]),
                class_id=int(labels[detection_index]),
                embedding=(
                    None
                    if embeddings is None
                    else np.asarray(embeddings[detection_index], dtype=np.float32).copy()
                ),
            )

    def _export_visible_tracks(self) -> list[Track]:
        visible_tracks: list[Track] = []
        for track_id in sorted(self._active_tracks.keys()):
            state = self._active_tracks[track_id]
            if state.missed_frames != 0:
                continue

            visible_tracks.append(
                Track(
                    track_id=state.track_id,
                    bbox_xyxy=(
                        float(state.bbox_xyxy[0]),
                        float(state.bbox_xyxy[1]),
                        float(state.bbox_xyxy[2]),
                        float(state.bbox_xyxy[3]),
                    ),
                    confidence=float(state.confidence),
                    class_id=int(state.class_id),
                    embedding=None if state.embedding is None else state.embedding.copy(),
                )
            )
        return visible_tracks
