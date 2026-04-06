from __future__ import annotations

# pyright: reportMissingImports=false

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def bootstrap_kaggle_workspace() -> Path:
    repo_url = os.environ.get("REID_REPO_URL", "https://github.com/ituvtu/reid_investigation.git")
    workspace_root = Path("/kaggle/working")
    project_root = workspace_root / "reid_investigation"

    if not project_root.exists():
        _run(["git", "clone", repo_url, str(project_root)])

    os.chdir(project_root)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if shutil.which("uv") is None:
        _run([sys.executable, "-m", "pip", "install", "uv"])

    _run(["uv", "pip", "install", "-r", "requirements.txt", "--system"])

    default_dataset_root = Path("/kaggle/input/soccernet-tracking")
    if "SOCCERNET_ROOT_DIR" not in os.environ and default_dataset_root.exists():
        os.environ["SOCCERNET_ROOT_DIR"] = str(default_dataset_root)

    print(f"Project root: {project_root};")
    print(f"SOCCERNET_ROOT_DIR: {os.environ.get('SOCCERNET_ROOT_DIR', '<not-set>')};")
    return project_root


def list_kaggle_soccernet_candidates() -> list[Path]:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return []

    candidates = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and "soccernet" in path.name.lower()
    )

    for path in candidates:
        print(path)

    return candidates


def download_soccernet_with_api(
    target_root: str | Path = "/kaggle/working/data/SoccerNet",
    subset: str = "tracking",
    split: tuple[str, ...] = ("train", "valid", "test"),
    password: str | None = None,
) -> Path:
    output_root = Path(target_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except Exception as error:
        raise RuntimeError("SoccerNet package is missing; install it before calling download_soccernet_with_api") from error

    downloader = SoccerNetDownloader(LocalDirectory=str(output_root))
    downloader.downloadDataTask(task=subset, split=list(split), password=password)

    os.environ["SOCCERNET_ROOT_DIR"] = str(output_root)
    print(f"SOCCERNET_ROOT_DIR: {output_root};")
    return output_root


if __name__ == "__main__":
    bootstrap_kaggle_workspace()
