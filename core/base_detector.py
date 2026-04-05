from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


ImageArray = NDArray[np.uint8]


@dataclass(slots=True, frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def area(self) -> float:
        return self.width * self.height


class BaseDetector(ABC):
    def __init__(self, model_name: str, config: Mapping[str, Any]) -> None:
        self.model_name = model_name
        self.config = dict(config)
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BaseDetector":
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def warmup(self, image_size: tuple[int, int] = (640, 640), runs: int = 1) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        image: ImageArray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        class_ids: Sequence[int] | None = None,
    ) -> list[Detection]:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError
