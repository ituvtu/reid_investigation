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

try:
    import torchreid
    from torchreid.utils import load_pretrained_weights
except Exception:  # pragma: no cover
    torchreid = None  # type: ignore[assignment]
    load_pretrained_weights = None  # type: ignore[assignment]


class OSNetReID(BaseReID):
    def __init__(self, model_name: str, config: Mapping[str, Any]) -> None:
        super().__init__(model_name=model_name, config=config)
        self.batch_size = max(1, int(config.get("batch_size", 32)))
        self.normalize_embeddings = bool(config.get("normalize_embeddings", True))
        self.pretrained = bool(config.get("pretrained", True))
        self.use_fp16 = bool(config.get("fp16", True))
        self.model_path = str(config.get("model_path")) if config.get("model_path") else None

        input_size = config.get("input_size", (256, 128))
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
        return 256, 128

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "OSNetReID":
        model_name = str(config.get("model_name", "osnet_x1_0"))
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
            raise ImportError("torch is required to load OSNetReID")
        if torchreid is None:
            raise ImportError("torchreid is required to load OSNetReID")

        runtime_device = self._resolve_runtime_device(self._device)
        use_gpu = runtime_device.startswith("cuda")
        self._model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=1000,
            loss="softmax",
            pretrained=self.pretrained,
            use_gpu=use_gpu,
        )
        if self._model is None:
            raise RuntimeError("torchreid returned an empty model")

        if self.model_path:
            if load_pretrained_weights is None:
                raise ImportError("torchreid.utils.load_pretrained_weights is unavailable")
            load_pretrained_weights(self._model, self.model_path)

        self._model.eval()
        self._model.to(runtime_device)

        with torch.inference_mode():
            probe = torch.zeros(
                (1, 3, self.input_height, self.input_width),
                dtype=torch.float32,
                device=runtime_device,
            )
            feature_probe = self._forward_model(probe)
            self._embedding_dim = int(feature_probe.shape[1]) if feature_probe.ndim == 2 else 0

        self._device = runtime_device
        self._is_loaded = True

    def extract(self, crops: list) -> Any:
        if not self._is_loaded:
            self.load()

        if self._model is None or torch is None:
            raise RuntimeError("OSNetReID is not initialized")

        if not crops:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        runtime_device = self._resolve_runtime_device(self._device)
        embeddings_batches: list[NDArray[np.float32]] = []

        for start in range(0, len(crops), self.batch_size):
            batch_crops = crops[start : start + self.batch_size]
            batch_tensor = self._preprocess_batch(batch_crops).to(runtime_device)

            if self.use_fp16 and runtime_device.startswith("cuda"):
                batch_tensor = batch_tensor.half()

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

    def _forward_model(self, inputs: Any) -> Any:
        if self._model is None:
            raise RuntimeError("OSNetReID model is not loaded")
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

    def _preprocess_batch(self, crops: list) -> Any:
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
