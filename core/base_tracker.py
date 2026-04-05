from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from core.base_detector import Detection


EmbeddingArray = NDArray[np.float32]
BBoxArray = NDArray[np.float32]
ConfidenceArray = NDArray[np.float32]
ClassIdArray = NDArray[np.int32]


@dataclass(slots=True, frozen=True)
class Track:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    embedding: EmbeddingArray | None = None


class BaseTracker(ABC):
    def __init__(self, tracker_name: str, config: Mapping[str, Any]) -> None:
        self.tracker_name = tracker_name
        self.config = dict(config)
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BaseTracker":
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        detections: Sequence[Detection],
        embeddings: EmbeddingArray | None = None,
        frame_index: int | None = None,
    ) -> list[Track]:
        raise NotImplementedError

    @abstractmethod
    def update_from_arrays(
        self,
        bboxes_xyxy: BBoxArray,
        confidences: ConfidenceArray,
        class_ids: ClassIdArray | None = None,
        embeddings: EmbeddingArray | None = None,
        frame_index: int | None = None,
    ) -> list[Track]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError
