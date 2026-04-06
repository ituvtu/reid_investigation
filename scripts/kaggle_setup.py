from __future__ import annotations

# pyright: reportMissingImports=false

import os
import shutil
import subprocess
import sys
import importlib
import tempfile
from pathlib import Path


def _run(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return result


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _ensure_torchreid_runtime() -> None:
    if _can_import("torchreid"):
        return

    print("torchreid import failed after requirements install; applying Kaggle fallback install;")
    _run([sys.executable, "-m", "pip", "install", "yacs", "tensorboard", "future", "gdown"])
    _run([sys.executable, "-m", "pip", "install", "git+https://github.com/KaiyangZhou/deep-person-reid.git"])

    if not _can_import("torchreid"):
        raise RuntimeError(
            "torchreid is still unavailable after fallback installation; "
            "check pip output for dependency conflicts."
        )


def _install_requirements_with_fallback(uv_command: list[str], requirements_path: Path) -> None:
    result = _run_capture(uv_command + ["pip", "install", "-r", str(requirements_path), "--system"])
    if result.returncode == 0:
        return

    stderr_text = (result.stderr or "").lower()
    has_dinov2_conflict = "dinov2" in stderr_text and "unsatisfiable" in stderr_text
    if not has_dinov2_conflict:
        raise RuntimeError(f"Failed to install requirements from {requirements_path};")

    print("Detected dinov2 and torchvision resolver conflict; retrying without dinov2;")
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    filtered_lines = [line for line in lines if "dinov2" not in line.lower()]

    with tempfile.NamedTemporaryFile("w", suffix="_requirements_no_dinov2.txt", delete=False, encoding="utf-8") as handle:
        handle.write("\n".join(filtered_lines) + "\n")
        fallback_requirements = Path(handle.name)

    try:
        _run(uv_command + ["pip", "install", "-r", str(fallback_requirements), "--system"])
    finally:
        fallback_requirements.unlink(missing_ok=True)


def bootstrap_kaggle_workspace() -> Path:
    repo_url = os.environ.get("REID_REPO_URL", "https://github.com/ituvtu/reid_investigation.git")
    workspace_root = Path("/kaggle/working")
    project_root = workspace_root / "reid_investigation"

    if not project_root.exists():
        _run(["git", "clone", repo_url, str(project_root)])

    os.chdir(project_root)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    uv_cli = shutil.which("uv")
    if uv_cli is None:
        _run([sys.executable, "-m", "pip", "install", "uv"])
        uv_command = [sys.executable, "-m", "uv"]
    else:
        uv_command = [uv_cli]

    _install_requirements_with_fallback(uv_command, Path("requirements.txt"))
    _ensure_torchreid_runtime()

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
