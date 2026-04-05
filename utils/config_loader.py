from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigLoaderError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class DetectorConfig:
    type: str
    model_name: str
    weights_path: str
    nms_free: bool
    img_size: int
    max_detections: int
    confidence_threshold: float
    iou_threshold: float
    classes: tuple[int, ...] | None
    half_precision: bool
    device: str | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TrackerConfig:
    type: str
    track_threshold: float
    track_low_threshold: float
    new_track_threshold: float
    match_threshold: float
    track_buffer: int
    min_box_area: float
    frame_rate: int
    mot20: bool
    use_embeddings: bool
    embedding_weight: float
    motion_weight: float
    device: str | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    use_cuda: bool
    cudnn_benchmark: bool
    num_workers: int
    pin_memory: bool
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SoccerNetConfig:
    root_dir: str
    subset: str
    split: tuple[str, ...]
    password: str | None
    auto_download: bool
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    soccernet: SoccerNetConfig | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Stage1BaselineConfig:
    experiment_name: str
    seed: int
    detector: DetectorConfig
    tracker: TrackerConfig
    runtime: RuntimeConfig
    dataset: DatasetConfig | None = None


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigLoaderError(f"Expected mapping at '{field_name}'")
    return value


def _as_optional_mapping(value: Any, field_name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _as_mapping(value, field_name)


def _as_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigLoaderError(f"Expected non-empty string at '{field_name}'")
    return value


def _as_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigLoaderError(f"Expected boolean at '{field_name}'")
    return value


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigLoaderError(f"Expected integer at '{field_name}'")
    return value


def _as_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigLoaderError(f"Expected numeric value at '{field_name}'")
    return float(value)


def _as_optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _as_str(value, field_name)


def _as_optional_class_ids(value: Any, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ConfigLoaderError(f"Expected list or null at '{field_name}'")
    class_ids: list[int] = []
    for index, item in enumerate(value):
        class_ids.append(_as_int(item, f"{field_name}[{index}]"))
    return tuple(class_ids)


def _as_optional_str_tuple(value: Any, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ConfigLoaderError(f"Expected list or null at '{field_name}'")
    items: list[str] = []
    for index, item in enumerate(value):
        items.append(_as_str(item, f"{field_name}[{index}]"))
    return tuple(items)


def _extra_fields(source: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key not in known}


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigLoaderError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigLoaderError("Top-level YAML content must be a mapping")
    return payload


def parse_stage1_baseline_config(data: Mapping[str, Any]) -> Stage1BaselineConfig:
    detector_raw = _as_mapping(data.get("detector"), "detector")
    tracker_raw = _as_mapping(data.get("tracker"), "tracker")
    runtime_raw = _as_mapping(data.get("runtime"), "runtime")
    dataset_raw = _as_optional_mapping(data.get("dataset"), "dataset")

    detector_known = {
        "type",
        "model_name",
        "weights_path",
        "nms_free",
        "img_size",
        "max_detections",
        "confidence_threshold",
        "iou_threshold",
        "classes",
        "half_precision",
        "device",
    }
    tracker_known = {
        "type",
        "track_threshold",
        "track_low_threshold",
        "new_track_threshold",
        "match_threshold",
        "track_buffer",
        "min_box_area",
        "frame_rate",
        "mot20",
        "use_embeddings",
        "embedding_weight",
        "motion_weight",
        "device",
    }
    runtime_known = {"use_cuda", "cudnn_benchmark", "num_workers", "pin_memory"}
    dataset_known = {"soccernet"}
    soccernet_known = {"root_dir", "subset", "split", "password", "auto_download"}

    detector = DetectorConfig(
        type=_as_str(detector_raw.get("type"), "detector.type"),
        model_name=_as_str(detector_raw.get("model_name"), "detector.model_name"),
        weights_path=_as_str(detector_raw.get("weights_path"), "detector.weights_path"),
        nms_free=_as_bool(detector_raw.get("nms_free"), "detector.nms_free"),
        img_size=_as_int(detector_raw.get("img_size"), "detector.img_size"),
        max_detections=_as_int(detector_raw.get("max_detections"), "detector.max_detections"),
        confidence_threshold=_as_float(detector_raw.get("confidence_threshold"), "detector.confidence_threshold"),
        iou_threshold=_as_float(detector_raw.get("iou_threshold"), "detector.iou_threshold"),
        classes=_as_optional_class_ids(detector_raw.get("classes"), "detector.classes"),
        half_precision=_as_bool(detector_raw.get("half_precision"), "detector.half_precision"),
        device=_as_optional_str(detector_raw.get("device"), "detector.device"),
        extra=_extra_fields(detector_raw, detector_known),
    )

    tracker = TrackerConfig(
        type=_as_str(tracker_raw.get("type"), "tracker.type"),
        track_threshold=_as_float(tracker_raw.get("track_threshold"), "tracker.track_threshold"),
        track_low_threshold=_as_float(tracker_raw.get("track_low_threshold"), "tracker.track_low_threshold"),
        new_track_threshold=_as_float(tracker_raw.get("new_track_threshold"), "tracker.new_track_threshold"),
        match_threshold=_as_float(tracker_raw.get("match_threshold"), "tracker.match_threshold"),
        track_buffer=_as_int(tracker_raw.get("track_buffer"), "tracker.track_buffer"),
        min_box_area=_as_float(tracker_raw.get("min_box_area"), "tracker.min_box_area"),
        frame_rate=_as_int(tracker_raw.get("frame_rate"), "tracker.frame_rate"),
        mot20=_as_bool(tracker_raw.get("mot20"), "tracker.mot20"),
        use_embeddings=_as_bool(tracker_raw.get("use_embeddings"), "tracker.use_embeddings"),
        embedding_weight=_as_float(tracker_raw.get("embedding_weight"), "tracker.embedding_weight"),
        motion_weight=_as_float(tracker_raw.get("motion_weight"), "tracker.motion_weight"),
        device=_as_optional_str(tracker_raw.get("device"), "tracker.device"),
        extra=_extra_fields(tracker_raw, tracker_known),
    )

    runtime = RuntimeConfig(
        use_cuda=_as_bool(runtime_raw.get("use_cuda"), "runtime.use_cuda"),
        cudnn_benchmark=_as_bool(runtime_raw.get("cudnn_benchmark"), "runtime.cudnn_benchmark"),
        num_workers=_as_int(runtime_raw.get("num_workers"), "runtime.num_workers"),
        pin_memory=_as_bool(runtime_raw.get("pin_memory"), "runtime.pin_memory"),
        extra=_extra_fields(runtime_raw, runtime_known),
    )

    dataset: DatasetConfig | None = None
    if dataset_raw is not None:
        soccernet_raw = _as_optional_mapping(dataset_raw.get("soccernet"), "dataset.soccernet")
        soccernet: SoccerNetConfig | None = None
        if soccernet_raw is not None:
            split_value = _as_optional_str_tuple(soccernet_raw.get("split"), "dataset.soccernet.split")
            soccernet = SoccerNetConfig(
                root_dir=_as_str(soccernet_raw.get("root_dir"), "dataset.soccernet.root_dir"),
                subset=_as_str(soccernet_raw.get("subset"), "dataset.soccernet.subset"),
                split=split_value if split_value is not None else ("train", "valid", "test"),
                password=_as_optional_str(soccernet_raw.get("password"), "dataset.soccernet.password"),
                auto_download=_as_bool(soccernet_raw.get("auto_download"), "dataset.soccernet.auto_download"),
                extra=_extra_fields(soccernet_raw, soccernet_known),
            )

        dataset = DatasetConfig(
            soccernet=soccernet,
            extra=_extra_fields(dataset_raw, dataset_known),
        )

    return Stage1BaselineConfig(
        experiment_name=_as_str(data.get("experiment_name"), "experiment_name"),
        seed=_as_int(data.get("seed"), "seed"),
        detector=detector,
        tracker=tracker,
        runtime=runtime,
        dataset=dataset,
    )


def load_stage1_baseline_config(path: str | Path) -> Stage1BaselineConfig:
    payload = load_yaml(path)
    return parse_stage1_baseline_config(payload)


def detector_config_to_mapping(config: DetectorConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": config.type,
        "model_name": config.model_name,
        "weights_path": config.weights_path,
        "nms_free": config.nms_free,
        "img_size": config.img_size,
        "max_detections": config.max_detections,
        "confidence_threshold": config.confidence_threshold,
        "iou_threshold": config.iou_threshold,
        "classes": list(config.classes) if config.classes is not None else None,
        "half_precision": config.half_precision,
        "device": config.device,
    }
    payload.update(config.extra)
    return payload


def tracker_config_to_mapping(config: TrackerConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": config.type,
        "track_threshold": config.track_threshold,
        "track_low_threshold": config.track_low_threshold,
        "new_track_threshold": config.new_track_threshold,
        "match_threshold": config.match_threshold,
        "track_buffer": config.track_buffer,
        "min_box_area": config.min_box_area,
        "frame_rate": config.frame_rate,
        "mot20": config.mot20,
        "use_embeddings": config.use_embeddings,
        "embedding_weight": config.embedding_weight,
        "motion_weight": config.motion_weight,
        "device": config.device,
    }
    payload.update(config.extra)
    return payload


def soccernet_config_to_mapping(config: SoccerNetConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "root_dir": config.root_dir,
        "subset": config.subset,
        "split": list(config.split),
        "password": config.password,
        "auto_download": config.auto_download,
    }
    payload.update(config.extra)
    return payload


def stage1_component_mappings(config: Stage1BaselineConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    return detector_config_to_mapping(config.detector), tracker_config_to_mapping(config.tracker)


def stage1_soccernet_mapping(config: Stage1BaselineConfig) -> dict[str, Any] | None:
    if config.dataset is None or config.dataset.soccernet is None:
        return None
    return soccernet_config_to_mapping(config.dataset.soccernet)

