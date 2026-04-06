from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from core.base_reid import BaseReID

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class DinoV2ReID(BaseReID):
    def __init__(self, model_name: str, config: Mapping[str, Any]) -> None:
        super().__init__(model_name=model_name, config=config)
        self.batch_size = max(1, int(config.get("batch_size", 16)))
        self.normalize_embeddings = bool(config.get("normalize_embeddings", True))
        self.pretrained = bool(config.get("pretrained", True))
        self.use_fp16 = bool(config.get("fp16", True))

        input_size = config.get("input_size", (224, 224))
        self.input_height, self.input_width = self._parse_input_size(input_size)

        requested_device = config.get("device")
        self._device = str(requested_device) if requested_device else self._auto_device()
        self._model: Any | None = None
        self._embedding_dim = 0

    @staticmethod
    def _auto_device() -> str:
        if torch is not None and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    @staticmethod
    def _parse_input_size(value: Any) -> tuple[int, int]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            return int(value[0]), int(value[1])
        return 224, 224

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DinoV2ReID":
        model_name = str(config.get("model_name", "dinov2_vits14"))
        return cls(model_name=model_name, config=config)

    @property
    def device(self) -> str:
        return self._device

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def set_device(self, device: str) -> None:
        normalized = device.strip()
        if not normalized:
            raise ValueError("Device must be a non-empty string")
        self._device = normalized

        if self._model is not None:
            runtime_device = self._resolve_runtime_device(self._device)
            self._model.to(runtime_device)

    def load(self) -> None:
        if self._is_loaded:
            return

        if torch is None:
            raise ImportError("torch is required to load DinoV2ReID")

        runtime_device = self._resolve_runtime_device(self._device)
        self._model = self._load_model()
        self._model.eval()
        self._model.to(runtime_device)

        with torch.inference_mode():
            probe = torch.zeros(
                (1, 3, self.input_height, self.input_width),
                dtype=torch.float32,
                device=runtime_device,
            )
            features = self._forward_model(probe)
            self._embedding_dim = int(features.shape[1]) if features.ndim == 2 else 0

        self._device = runtime_device
        self._is_loaded = True

    def extract(self, crops: list[np.ndarray]) -> NDArray[np.float32]:
        if not self._is_loaded:
            self.load()

        if self._model is None or torch is None:
            raise RuntimeError("DinoV2ReID is not initialized")

        if not crops:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        runtime_device = self._resolve_runtime_device(self._device)
        embeddings_batches: list[NDArray[np.float32]] = []

        for start in range(0, len(crops), self.batch_size):
            batch_crops = crops[start : start + self.batch_size]
            batch_tensor = self._preprocess_batch(batch_crops).to(runtime_device)

            with torch.inference_mode():
                if self.use_fp16 and runtime_device.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        features = self._forward_model(batch_tensor)
                else:
                    features = self._forward_model(batch_tensor)

                features = features.float()
                if self.normalize_embeddings:
                    features = torch.nn.functional.normalize(features, p=2, dim=1)

            embeddings_batches.append(features.detach().to("cpu").numpy().astype(np.float32, copy=False))

        if not embeddings_batches:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        return np.vstack(embeddings_batches).astype(np.float32, copy=False)

    def shutdown(self) -> None:
        self._model = None
        self._is_loaded = False

        if self._device.startswith("cuda") and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self) -> Any:
        if torch is None:
            raise RuntimeError("torch is required to load DINOv2")

        try:
            return torch.hub.load(
                repo_or_dir="facebookresearch/dinov2",
                model=self.model_name,
                pretrained=self.pretrained,
                trust_repo=True,
            )
        except TypeError:
            return torch.hub.load(
                repo_or_dir="facebookresearch/dinov2",
                model=self.model_name,
                trust_repo=True,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to load DINOv2 from torch.hub. Ensure internet access and valid model name."
            ) from error

    def _forward_model(self, inputs: Any) -> Any:
        if self._model is None:
            raise RuntimeError("DinoV2ReID model is not loaded")
        if torch is None:
            raise RuntimeError("torch is required for forward pass")

        outputs = self._model(inputs)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if outputs.ndim > 2:
            outputs = torch.flatten(outputs, start_dim=1)
        return outputs

    def _resolve_runtime_device(self, requested_device: str) -> str:
        if requested_device.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
            return "cpu"
        return requested_device

    def _preprocess_batch(self, crops: list[np.ndarray]) -> Any:
        if torch is None:
            raise RuntimeError("torch is required for preprocessing")

        tensors: list[Any] = []
        for crop in crops:
            normalized_crop = self._prepare_crop(crop)
            tensor = torch.from_numpy(normalized_crop).permute(2, 0, 1)
            tensors.append(tensor)

        batch = torch.stack(tensors, dim=0)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        batch = (batch - mean) / std
        return batch

    def _prepare_crop(self, crop: Any) -> NDArray[np.float32]:
        if crop is None:
            working = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        else:
            working = np.asarray(crop)

        if working.ndim == 2:
            working = np.repeat(working[:, :, None], 3, axis=2)
        if working.ndim == 3 and working.shape[2] == 1:
            working = np.repeat(working, 3, axis=2)
        if working.ndim == 3 and working.shape[2] > 3:
            working = working[:, :, :3]

        if working.ndim != 3:
            working = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)

        if working.size == 0 or working.shape[0] == 0 or working.shape[1] == 0:
            working = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)

        resized = cv2.resize(working, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].copy()
        return (rgb.astype(np.float32) / 255.0).astype(np.float32, copy=False)
