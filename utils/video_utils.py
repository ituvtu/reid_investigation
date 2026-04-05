from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def is_colab_environment() -> bool:
    return os.path.exists("/content")


def resolve_existing_path(path: str | Path, extra_roots: Iterable[str | Path] | None = None) -> Path:
    candidate = Path(path).expanduser()
    if os.path.exists(candidate):
        return candidate

    roots: list[Path] = [Path.cwd()]
    if is_colab_environment():
        roots.extend(
            [
                Path("/content"),
                Path("/content/drive/MyDrive"),
            ]
        )

    if extra_roots is not None:
        roots.extend(Path(root).expanduser() for root in extra_roots)

    normalized = str(path).lstrip("/")
    for root in roots:
        trial = root / normalized
        if os.path.exists(trial):
            return trial

    raise FileNotFoundError(f"Path not found: {path}")


def ensure_video_path(path: str | Path, extra_roots: Iterable[str | Path] | None = None) -> Path:
    resolved = resolve_existing_path(path, extra_roots=extra_roots)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Video file not found: {resolved}")
    return resolved


def open_video_capture(path: str | Path, extra_roots: Iterable[str | Path] | None = None) -> Any:
    import cv2

    resolved = ensure_video_path(path, extra_roots=extra_roots)
    capture = cv2.VideoCapture(str(resolved))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {resolved}")
    return capture


@dataclass(slots=True, frozen=True)
class VideoSource:
    path: str
    extra_roots: tuple[str, ...] = ()

    def resolve(self) -> Path:
        return ensure_video_path(self.path, extra_roots=self.extra_roots)

    def open_capture(self) -> Any:
        return open_video_capture(self.path, extra_roots=self.extra_roots)


__all__ = [
    "VideoSource",
    "ensure_video_path",
    "is_colab_environment",
    "open_video_capture",
    "resolve_existing_path",
]
