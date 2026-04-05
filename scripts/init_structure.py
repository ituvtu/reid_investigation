from __future__ import annotations

from pathlib import Path


PROJECT_DIRECTORIES = (
    "core",
    "models",
    "utils",
    "configs",
    "experiments",
)

PACKAGE_DIRECTORIES = (
    "core",
    "models",
    "utils",
)


def initialize_structure(project_root: Path) -> None:
    for directory in PROJECT_DIRECTORIES:
        target = project_root / directory
        target.mkdir(parents=True, exist_ok=True)

    for package_directory in PACKAGE_DIRECTORIES:
        init_file = project_root / package_directory / "__init__.py"
        init_file.touch(exist_ok=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    initialize_structure(project_root)
    print(f"Project structure initialized at: {project_root}")


if __name__ == "__main__":
    main()
