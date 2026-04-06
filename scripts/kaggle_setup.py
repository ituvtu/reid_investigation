from __future__ import annotations

# pyright: reportMissingImports=false

import os
import shutil
import subprocess
import sys
import importlib
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


def _is_build_isolation_issue(stderr_text: str) -> bool:
    normalized = stderr_text.lower()
    markers = (
        "build backend returned an error",
        "no module named 'numpy'",
        "no module named 'wrapt'",
        "no-build-isolation",
    )
    return any(marker in normalized for marker in markers)


def _is_dinov2_solver_conflict(text: str) -> bool:
    normalized = text.lower()
    return "dinov2" in normalized and "unsatisfiable" in normalized


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _has_compatible_torchreid() -> bool:
    try:
        torchreid_module = importlib.import_module("torchreid")
    except Exception:
        return False

    models_api = getattr(torchreid_module, "models", None)
    has_models_api = models_api is not None and hasattr(models_api, "build_model")
    if not has_models_api:
        return False

    try:
        from torchreid.utils import load_pretrained_weights as _  # noqa: F401
    except Exception:
        return False

    return True


def _ensure_torchreid_runtime() -> None:
    if _has_compatible_torchreid():
        return

    print("Compatible torchreid runtime was not found; applying Kaggle fallback install;")
    _run([sys.executable, "-m", "pip", "install", "yacs", "tensorboard", "future", "gdown"])

    _run([sys.executable, "-m", "pip", "uninstall", "-y", "torchreid"])
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "git+https://github.com/KaiyangZhou/deep-person-reid.git",
        ]
    )

    if not _has_compatible_torchreid():
        raise RuntimeError(
            "Compatible torchreid runtime is still unavailable after fallback installation; "
            "check pip output for dependency conflicts."
        )


def _install_requirements_with_fallback(uv_command: list[str], requirements_path: Path) -> None:
    runtime_requirements = Path("/tmp/requirements_stage2_runtime_no_dinov2.txt")
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    filtered_lines = [line for line in lines if "dinov2" not in line.lower()]
    runtime_requirements.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")

    include_dinov2_optional = os.environ.get("INSTALL_DINOV2_OPTIONAL", "0") == "1"
    requirements_to_install = requirements_path if include_dinov2_optional else runtime_requirements

    if include_dinov2_optional:
        print("INSTALL_DINOV2_OPTIONAL=1; using full requirements;")
    else:
        print("Using runtime requirements without dinov2 for compatibility;")

    result = _run_capture(uv_command + ["pip", "install", "-r", str(requirements_to_install), "--system"])
    if result.returncode == 0:
        return

    merged_output = f"{result.stdout}\n{result.stderr}"

    if _is_dinov2_solver_conflict(merged_output):
        print("Detected dinov2 solver conflict; retrying without dinov2;")
        requirements_to_install = runtime_requirements
        retry_no_dino = _run_capture(
            uv_command + ["pip", "install", "-r", str(requirements_to_install), "--system", "--no-build-isolation"]
        )
        if retry_no_dino.returncode == 0:
            return
        print("uv retry without dinov2 failed; fallback to pip --no-build-isolation;")
        _run([sys.executable, "-m", "pip", "install", "-r", str(requirements_to_install), "--no-build-isolation"])
        return

    if _is_build_isolation_issue(merged_output):
        print("Detected build-isolation issue; retrying with --no-build-isolation;")
        _run([sys.executable, "-m", "pip", "install", "-q", "numpy", "wrapt", "setuptools", "wheel"])
        retry_result = _run_capture(
            uv_command + ["pip", "install", "-r", str(requirements_to_install), "--system", "--no-build-isolation"]
        )

        if retry_result.returncode == 0:
            return

        retry_merged_output = f"{retry_result.stdout}\n{retry_result.stderr}"
        if _is_dinov2_solver_conflict(retry_merged_output):
            print("Detected dinov2 solver conflict after build retry; switching to requirements without dinov2;")
            requirements_to_install = runtime_requirements

        print("uv retry is still failing; falling back to pip --no-build-isolation;")
        _run([sys.executable, "-m", "pip", "install", "-r", str(requirements_to_install), "--no-build-isolation"])
        return

    raise RuntimeError("Failed to install requirements via uv fallback strategy;")


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
