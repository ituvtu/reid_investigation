from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from ultralytics import YOLO

from core.base_detector import BaseDetector, Detection, ImageArray


class YOLODetector(BaseDetector):
    def __init__(self, model_name: str, config: Mapping[str, Any]) -> None:
        super().__init__(model_name=model_name, config=config)
        self.weights_path = str(config.get("weights_path", model_name))
        self.image_size = int(config.get("img_size", 640))
        self.max_detections = int(config.get("max_detections", 300))
        self.use_half_precision = bool(config.get("half_precision", True))
        self.default_confidence = float(config.get("confidence_threshold", 0.25))
        self.default_iou = float(config.get("iou_threshold", 0.7))
        raw_default_classes = config.get("classes")
        if raw_default_classes is None:
            self.default_class_ids: tuple[int, ...] | None = None
        else:
            self.default_class_ids = tuple(int(value) for value in raw_default_classes)

        requested_device = config.get("device")
        self._device = str(requested_device) if requested_device else self._auto_device()
        self._manual_device_override = requested_device is not None
        self._model: YOLO | None = None

    @staticmethod
    def _auto_device() -> str:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "YOLODetector":
        model_name = str(config.get("model_name", "yolo26m"))
        return cls(model_name=model_name, config=config)

    @property
    def device(self) -> str:
        return self._device

    def set_device(self, device: str) -> None:
        normalized = device.strip()
        if not normalized:
            raise ValueError("Device must be a non-empty string")

        self._device = normalized
        self._manual_device_override = True

        if self._model is not None:
            self._model.to(self._device)

    def load(self) -> None:
        if self._is_loaded:
            return

        source = self.weights_path if self.weights_path else self.model_name
        self._model = YOLO(source)
        self._model.to(self._device)
        self._is_loaded = True

    def warmup(self, image_size: tuple[int, int] = (640, 640), runs: int = 1) -> None:
        if not self._is_loaded:
            self.load()

        warmup_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        total_runs = max(1, runs)
        for _ in range(total_runs):
            self.predict(
                image=warmup_image,
                confidence_threshold=self.default_confidence,
                iou_threshold=self.default_iou,
                class_ids=None,
            )

    def predict(
        self,
        image: ImageArray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        class_ids: Sequence[int] | None = None,
    ) -> list[Detection]:
        if not self._is_loaded:
            self.load()

        if self._model is None:
            raise RuntimeError("Detector model is not available")

        effective_class_ids = class_ids if class_ids is not None else self.default_class_ids
        classes_arg = list(effective_class_ids) if effective_class_ids is not None else None
        predict_kwargs: dict[str, Any] = {
            "source": image,
            "conf": float(confidence_threshold),
            "iou": float(iou_threshold),
            "imgsz": self.image_size,
            "max_det": self.max_detections,
            "classes": classes_arg,
            "device": self._device,
            "half": self.use_half_precision and self._device.startswith("cuda"),
            "verbose": False,
        }

        raw_results = self._model.predict(**predict_kwargs)
        if not raw_results:
            return []

        result = raw_results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy.numel() == 0:
            return []

        xyxy = boxes.xyxy.detach().to("cpu").numpy()
        confidences = boxes.conf.detach().to("cpu").numpy() if boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
        class_ids_array = boxes.cls.detach().to("cpu").numpy().astype(np.int32) if boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=np.int32)
        names = result.names if isinstance(result.names, dict) else {}

        detections: list[Detection] = []
        for index in range(xyxy.shape[0]):
            class_id = int(class_ids_array[index])
            class_name = names.get(class_id)
            bbox = (
                float(xyxy[index, 0]),
                float(xyxy[index, 1]),
                float(xyxy[index, 2]),
                float(xyxy[index, 3]),
            )
            detections.append(
                Detection(
                    bbox_xyxy=bbox,
                    confidence=float(confidences[index]),
                    class_id=class_id,
                    class_name=str(class_name) if class_name is not None else None,
                )
            )

        return detections

    def shutdown(self) -> None:
        self._model = None
        self._is_loaded = False

        if self._device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
