from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_STORAGE_ROOT = REPO_ROOT / "storage"
DEFAULT_CONFIG_ROOT = REPO_ROOT / "config"


def repo_root() -> Path:
    return REPO_ROOT


def storage_root() -> Path:
    return DEFAULT_STORAGE_ROOT


def config_root() -> Path:
    return DEFAULT_CONFIG_ROOT


def ensure_lab_directories() -> None:
    for path in [
        storage_root(),
        storage_root() / "docs",
        storage_root() / "datasets",
        storage_root() / "runs",
        storage_root() / "artifacts",
        storage_root() / "eval_results",
        storage_root() / "indexes",
        storage_root() / "synthetic",
        config_root(),
    ]:
        path.mkdir(parents=True, exist_ok=True)

