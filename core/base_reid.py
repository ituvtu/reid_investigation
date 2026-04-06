from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


EmbeddingArray = NDArray[np.float32]


class BaseReID(ABC):
    def __init__(self, model_name: str, config: Mapping[str, Any]) -> None:
        self.model_name = model_name
        self.config = dict(config)
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BaseReID":
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def extract(self, crops: list[np.ndarray]) -> EmbeddingArray:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError
