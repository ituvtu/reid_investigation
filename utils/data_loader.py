from __future__ import annotations

import csv
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from core.base_detector import Detection
from core.base_tracker import Track
from utils.config_loader import SoccerNetConfig


class SoccerNetLoaderError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class SoccerNetDownloadConfig:
    root_dir: str
    subset: str = "tracking"
    split: tuple[str, ...] = ("train", "valid", "test")
    password: str | None = None
    auto_download: bool = True


class SoccerNetLoader:
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
    ANNOTATION_EXTENSIONS = (".json", ".csv", ".txt")

    def __init__(self, config: SoccerNetDownloadConfig) -> None:
        self.root_dir = Path(config.root_dir).expanduser()
        self.subset = config.subset
        self.split = config.split
        self.password = config.password
        self.auto_download = config.auto_download

    @classmethod
    def from_config(cls, config: SoccerNetConfig | Mapping[str, Any]) -> "SoccerNetLoader":
        if isinstance(config, SoccerNetConfig):
            return cls(
                SoccerNetDownloadConfig(
                    root_dir=config.root_dir,
                    subset=config.subset,
                    split=config.split,
                    password=config.password,
                    auto_download=config.auto_download,
                )
            )

        split_raw = config.get("split", ["train", "valid", "test"])
        if not isinstance(split_raw, list):
            raise SoccerNetLoaderError("Field 'split' must be a list of strings")

        split: list[str] = []
        for index, value in enumerate(split_raw):
            if not isinstance(value, str):
                raise SoccerNetLoaderError(f"Field 'split[{index}]' must be a string")
            split.append(value)

        password_value = config.get("password")
        if password_value is not None and not isinstance(password_value, str):
            raise SoccerNetLoaderError("Field 'password' must be a string or null")

        return cls(
            SoccerNetDownloadConfig(
                root_dir=str(config.get("root_dir", "")),
                subset=str(config.get("subset", "tracking")),
                split=tuple(split),
                password=password_value,
                auto_download=bool(config.get("auto_download", True)),
            )
        )

    def ensure_dataset(self, force_download: bool = False) -> Path:
        existing_root = self._resolve_existing_dataset_root()
        if existing_root is not None and not force_download:
            self.root_dir = existing_root
            return existing_root

        if not self.auto_download and not force_download:
            raise SoccerNetLoaderError(
                f"SoccerNet data was not found at {self.root_dir}; set a valid local dataset path or enable auto_download"
            )

        download_root = self._resolve_download_root()
        try:
            download_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SoccerNetLoaderError(
                f"Unable to prepare SoccerNet download directory: {download_root}; {error}"
            ) from error

        self.root_dir = download_root

        self.download_tracking_subset()

        resolved_root = self._resolve_existing_dataset_root()
        if resolved_root is None:
            raise SoccerNetLoaderError(
                f"Download completed but dataset files were not found at {self.root_dir}"
            )
        self.root_dir = resolved_root
        return resolved_root

    def dataset_exists(self) -> bool:
        resolved_root = self._resolve_existing_dataset_root()
        if resolved_root is None:
            return False
        self.root_dir = resolved_root
        return True

    def download_tracking_subset(self) -> None:
        try:
            from SoccerNet.Downloader import SoccerNetDownloader
        except Exception as error:
            raise SoccerNetLoaderError(
                "SoccerNet API is not available; install SoccerNet before calling download"
            ) from error

        downloader = self._create_downloader(SoccerNetDownloader)

        method_candidates: list[tuple[str, dict[str, Any]]] = [
            (
                "downloadDataTask",
                {
                    "task": self.subset,
                    "split": list(self.split),
                    "password": self.password,
                },
            ),
            (
                "downloadGames",
                {
                    "split": list(self.split),
                    "task": [self.subset],
                    "password": self.password,
                },
            ),
            (
                "download",
                {
                    "task": self.subset,
                    "split": list(self.split),
                    "password": self.password,
                },
            ),
        ]

        errors: list[str] = []
        for method_name, kwargs in method_candidates:
            method = getattr(downloader, method_name, None)
            if method is None:
                continue

            try:
                self._invoke_with_supported_kwargs(method, kwargs)
                return
            except Exception as error:
                errors.append(f"{method_name}: {error}")

        error_message = "; ".join(errors) if errors else "No supported download method found"
        raise SoccerNetLoaderError(f"Unable to download SoccerNet subset '{self.subset}'; {error_message}")

    def map_tracking_annotations_to_detections(
        self,
        annotation_payload: Mapping[str, Any] | list[Mapping[str, Any]],
        default_class_id: int = 0,
        default_confidence: float = 1.0,
    ) -> dict[int, list[Detection]]:
        records = self._extract_annotation_records(annotation_payload)
        frame_to_detections: dict[int, list[Detection]] = {}

        for record in records:
            frame_index = self._extract_frame_index(record)
            bbox_xyxy = self._extract_bbox_xyxy(record)
            confidence = float(record.get("confidence", record.get("score", default_confidence)))
            class_id = int(record.get("class_id", record.get("label_id", default_class_id)))
            class_name_value = record.get("class_name", record.get("label", None))
            class_name = None if class_name_value is None else str(class_name_value)

            frame_to_detections.setdefault(frame_index, []).append(
                Detection(
                    bbox_xyxy=bbox_xyxy,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return frame_to_detections

    def map_tracking_annotations_to_tracks(
        self,
        annotation_payload: Mapping[str, Any] | list[Mapping[str, Any]],
        default_class_id: int = 0,
        default_confidence: float = 1.0,
    ) -> dict[int, list[Track]]:
        records = self._extract_annotation_records(annotation_payload)
        frame_to_tracks: dict[int, list[Track]] = {}

        for record in records:
            frame_index = self._extract_frame_index(record)
            track_id = self._extract_track_id(record)
            bbox_xyxy = self._extract_bbox_xyxy(record)
            confidence = float(record.get("confidence", record.get("score", default_confidence)))
            class_id = int(record.get("class_id", record.get("label_id", default_class_id)))

            frame_to_tracks.setdefault(frame_index, []).append(
                Track(
                    track_id=track_id,
                    bbox_xyxy=bbox_xyxy,
                    confidence=confidence,
                    class_id=class_id,
                    embedding=None,
                )
            )

        return frame_to_tracks

    def find_video_files(self, max_results: int | None = None) -> list[Path]:
        candidates: list[Path] = []
        for search_root in self._search_roots():
            if not os.path.exists(search_root):
                continue
            for path in search_root.rglob("*"):
                if path.is_file() and path.suffix.lower() in self.VIDEO_EXTENSIONS:
                    candidates.append(path)
                    if max_results is not None and len(candidates) >= max_results:
                        return sorted(candidates)
        return sorted(candidates)

    def find_annotation_files(self, max_results: int | None = None) -> list[Path]:
        candidates: list[Path] = []
        for search_root in self._search_roots():
            if not os.path.exists(search_root):
                continue
            for extension in self.ANNOTATION_EXTENSIONS:
                for path in search_root.rglob(f"*{extension}"):
                    if path.is_file():
                        candidates.append(path)
                        if max_results is not None and len(candidates) >= max_results:
                            return sorted(candidates)
        return sorted(candidates)

    def get_default_video_path(self) -> Path:
        video_files = self.find_video_files(max_results=None)
        if not video_files:
            raise SoccerNetLoaderError(
                f"No video files were found under {self.root_dir}; provide a valid local SoccerNet root path"
            )

        preferred = [path for path in video_files if "tracking" in str(path).lower()]
        if preferred:
            return preferred[0]
        return video_files[0]

    def get_default_annotation_path(self) -> Path:
        annotation_files = self.find_annotation_files(max_results=None)
        if not annotation_files:
            raise SoccerNetLoaderError(
                f"No annotation files were found under {self.root_dir}; provide a valid local SoccerNet root path"
            )

        preferred_keywords = ("tracking", "annotation", "labels")
        for keyword in preferred_keywords:
            for path in annotation_files:
                if keyword in path.name.lower():
                    return path
        return annotation_files[0]

    def load_tracking_annotations(self, annotation_path: str | Path | None = None) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        target_path = self.get_default_annotation_path() if annotation_path is None else Path(annotation_path)
        if not os.path.exists(target_path):
            raise SoccerNetLoaderError(f"Annotation file was not found: {target_path}")

        if target_path.suffix.lower() in {".csv", ".txt"}:
            return self._load_tracking_csv_annotations(target_path)

        with target_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload
        raise SoccerNetLoaderError("Unsupported annotation JSON content")

    def _load_tracking_csv_annotations(self, path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            first_row = next(reader, None)
            if first_row is None:
                return records

            has_header = not self._is_numeric(first_row[0])
            header: list[str] | None = None

            if has_header:
                header = [item.strip().lower() for item in first_row]
            else:
                self._append_mot_row_record(records, first_row)

            for row in reader:
                if not row:
                    continue
                if header is None:
                    self._append_mot_row_record(records, row)
                else:
                    self._append_header_row_record(records, row, header)

        return records

    @staticmethod
    def _append_mot_row_record(records: list[dict[str, Any]], row: list[str]) -> None:
        if len(row) < 6:
            return
        try:
            records.append(
                {
                    "frame": int(float(row[0])),
                    "track_id": int(float(row[1])),
                    "x": float(row[2]),
                    "y": float(row[3]),
                    "w": float(row[4]),
                    "h": float(row[5]),
                    "confidence": float(row[6]) if len(row) > 6 and row[6] != "" else 1.0,
                }
            )
        except (TypeError, ValueError):
            return

    @staticmethod
    def _append_header_row_record(records: list[dict[str, Any]], row: list[str], header: list[str]) -> None:
        mapping = {header[index]: row[index] for index in range(min(len(header), len(row)))}
        if not mapping:
            return

        frame_value = (
            mapping.get("frame")
            or mapping.get("frame_id")
            or mapping.get("image_id")
            or mapping.get("frameindex")
        )
        track_value = (
            mapping.get("track_id")
            or mapping.get("id")
            or mapping.get("object_id")
            or mapping.get("player_id")
        )
        x = mapping.get("x") or mapping.get("left") or mapping.get("xmin")
        y = mapping.get("y") or mapping.get("top") or mapping.get("ymin")
        w = mapping.get("w") or mapping.get("width")
        h = mapping.get("h") or mapping.get("height")

        if frame_value is None or track_value is None or x is None or y is None or w is None or h is None:
            return

        try:
            records.append(
                {
                    "frame": int(float(frame_value)),
                    "track_id": int(float(track_value)),
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "confidence": float(mapping.get("confidence", mapping.get("score", 1.0))),
                }
            )
        except (TypeError, ValueError):
            return

    @staticmethod
    def _is_numeric(value: str) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def iter_video_frames(
        self,
        video_path: str | Path | None = None,
        max_frames: int | None = None,
        stride: int = 1,
    ) -> Iterator[tuple[int, Any]]:
        if stride <= 0:
            raise SoccerNetLoaderError("stride must be a positive integer")

        selected_video_path = self.get_default_video_path() if video_path is None else Path(video_path)
        if not os.path.exists(selected_video_path):
            raise SoccerNetLoaderError(f"Video path was not found: {selected_video_path}")

        try:
            import cv2
        except Exception as error:
            raise SoccerNetLoaderError("OpenCV is required for frame iteration") from error

        capture = cv2.VideoCapture(str(selected_video_path))
        if not capture.isOpened():
            raise SoccerNetLoaderError(f"Unable to open video: {selected_video_path}")

        produced_frames = 0
        frame_index = -1
        try:
            while True:
                is_ok, frame = capture.read()
                if not is_ok:
                    break

                frame_index += 1
                if frame_index % stride != 0:
                    continue

                yield frame_index, frame
                produced_frames += 1

                if max_frames is not None and produced_frames >= max_frames:
                    break
        finally:
            capture.release()

    def _create_downloader(self, downloader_class: type[Any]) -> Any:
        constructor_variants = [
            {"LocalDirectory": str(self.root_dir), "password": self.password},
            {"LocalDirectory": str(self.root_dir)},
            {"local_directory": str(self.root_dir), "password": self.password},
            {"local_directory": str(self.root_dir)},
        ]

        errors: list[str] = []
        for kwargs in constructor_variants:
            if kwargs.get("password") is None:
                kwargs = {key: value for key, value in kwargs.items() if key != "password"}

            try:
                downloader = downloader_class(**kwargs)
                if self.password is not None and hasattr(downloader, "password"):
                    setattr(downloader, "password", self.password)
                return downloader
            except Exception as error:
                errors.append(str(error))

        raise SoccerNetLoaderError(f"Could not initialize SoccerNetDownloader; {'; '.join(errors)}")

    @staticmethod
    def _invoke_with_supported_kwargs(method: Callable[..., Any], kwargs: Mapping[str, Any]) -> None:
        signature = inspect.signature(method)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        if accepts_var_kwargs:
            filtered_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        else:
            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters and value is not None
            }

        method(**filtered_kwargs)

    @staticmethod
    def _extract_annotation_records(
        annotation_payload: Mapping[str, Any] | list[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        if isinstance(annotation_payload, list):
            return annotation_payload

        key_candidates = (
            "annotations",
            "instances",
            "labels",
            "frames",
            "data",
        )
        for key in key_candidates:
            value = annotation_payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                records: list[Mapping[str, Any]] = []
                for frame_key, frame_records in value.items():
                    if not isinstance(frame_records, list):
                        continue
                    for record in frame_records:
                        if not isinstance(record, Mapping):
                            continue
                        record_with_frame = dict(record)
                        if "frame" not in record_with_frame:
                            record_with_frame["frame"] = frame_key
                        records.append(record_with_frame)
                if records:
                    return records

        raise SoccerNetLoaderError("Unsupported SoccerNet annotation payload format")

    @staticmethod
    def _extract_frame_index(record: Mapping[str, Any]) -> int:
        frame_key_candidates = ("frame", "frame_id", "frameIndex", "image_id", "time")
        for key in frame_key_candidates:
            value = record.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue

        raise SoccerNetLoaderError(f"Could not extract frame index from record: {record}")

    @staticmethod
    def _extract_track_id(record: Mapping[str, Any]) -> int:
        track_key_candidates = ("track_id", "id", "player_id", "object_id", "instance_id", "uid")
        for key in track_key_candidates:
            value = record.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue

        raise SoccerNetLoaderError(f"Could not extract track id from record: {record}")

    @staticmethod
    def _extract_bbox_xyxy(record: Mapping[str, Any]) -> tuple[float, float, float, float]:
        if "bbox" in record and isinstance(record["bbox"], (list, tuple)) and len(record["bbox"]) >= 4:
            bbox_values = [float(value) for value in record["bbox"][:4]]
            return SoccerNetLoader._bbox_from_xywh(bbox_values)

        if "box" in record and isinstance(record["box"], (list, tuple)) and len(record["box"]) >= 4:
            box_values = [float(value) for value in record["box"][:4]]
            return SoccerNetLoader._bbox_from_xywh(box_values)

        x = record.get("x", record.get("left", record.get("xmin", None)))
        y = record.get("y", record.get("top", record.get("ymin", None)))
        w = record.get("w", record.get("width", None))
        h = record.get("h", record.get("height", None))
        if x is not None and y is not None and w is not None and h is not None:
            x_value = float(x)
            y_value = float(y)
            w_value = float(w)
            h_value = float(h)
            return (x_value, y_value, x_value + w_value, y_value + h_value)

        xmin = record.get("xmin", None)
        ymin = record.get("ymin", None)
        xmax = record.get("xmax", None)
        ymax = record.get("ymax", None)
        if xmin is not None and ymin is not None and xmax is not None and ymax is not None:
            return (float(xmin), float(ymin), float(xmax), float(ymax))

        raise SoccerNetLoaderError(f"Could not extract bounding box from record: {record}")

    @staticmethod
    def _bbox_from_xywh(values: list[float]) -> tuple[float, float, float, float]:
        x1, y1, width, height = values
        return (x1, y1, x1 + width, y1 + height)

    def _search_roots(self) -> list[Path]:
        roots: list[Path] = []
        for candidate_root in self._candidate_dataset_roots():
            subset_root = candidate_root / self.subset
            self._append_candidate_root(roots, subset_root)
            self._append_candidate_root(roots, candidate_root)
        return roots

    def _resolve_existing_dataset_root(self) -> Path | None:
        payload_extensions = self.VIDEO_EXTENSIONS + self.ANNOTATION_EXTENSIONS + (".zip",)

        for candidate_root in self._candidate_dataset_roots():
            search_targets = [candidate_root / self.subset, candidate_root]
            for search_target in search_targets:
                if not os.path.exists(search_target):
                    continue

                for extension in payload_extensions:
                    if any(search_target.rglob(f"*{extension}")):
                        return candidate_root
        return None

    def _candidate_dataset_roots(self) -> list[Path]:
        roots: list[Path] = []

        env_root = os.environ.get("SOCCERNET_ROOT_DIR", "").strip()
        if env_root:
            self._append_candidate_root(roots, Path(env_root).expanduser())

        self._append_candidate_root(roots, self.root_dir)

        if self._is_kaggle_environment():
            input_root = Path("/kaggle/input")
            self._append_candidate_root(roots, Path("/kaggle/input/SoccerNet"))
            self._append_candidate_root(roots, Path("/kaggle/input/soccernet"))
            self._append_candidate_root(roots, Path("/kaggle/input/soccernet-tracking"))

            if os.path.exists(input_root):
                for child in sorted(input_root.iterdir()):
                    if child.is_dir() and "soccernet" in child.name.lower():
                        self._append_candidate_root(roots, child)

            self._append_candidate_root(roots, Path("/kaggle/working/data/SoccerNet"))
            self._append_candidate_root(roots, Path("/kaggle/temp/SoccerNet"))

        if os.path.exists("/content"):
            self._append_candidate_root(roots, Path("/content/drive/MyDrive/SoccerNet"))
            self._append_candidate_root(roots, Path("/content/SoccerNet"))

        return roots

    def _resolve_download_root(self) -> Path:
        override = os.environ.get("SOCCERNET_DOWNLOAD_ROOT", "").strip()
        if override:
            return Path(override).expanduser()

        if self._is_kaggle_environment():
            return Path("/kaggle/working/data/SoccerNet")

        return self.root_dir

    @staticmethod
    def _append_candidate_root(roots: list[Path], candidate: Path) -> None:
        expanded = candidate.expanduser()
        if expanded not in roots:
            roots.append(expanded)

    @staticmethod
    def _is_kaggle_environment() -> bool:
        return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"))

    @staticmethod
    def _contains_any_files(directory: Path) -> bool:
        for _, _, files in os.walk(directory):
            if files:
                return True
        return False
