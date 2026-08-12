"""Small shared helpers: seeding, logging, provenance, JSON I/O."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
    return logging.getLogger(name)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG we touch. `deterministic=True` trades throughput for
    bit-reproducibility — worth it when you are chasing a variance question,
    not worth it for routine runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is always present in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except ImportError:  # pragma: no cover
        pass


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:
        return False


def environment_metadata() -> dict[str, Any]:
    """Everything a reader needs to know whether they can reproduce a number."""
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod in ("torch", "transformers", "trl", "peft", "datasets", "accelerate"):
        try:
            meta[f"{mod}_version"] = __import__(mod).__version__
        except Exception:
            meta[f"{mod}_version"] = None
    try:
        import torch

        meta["cuda"] = torch.version.cuda
        meta["gpu_count"] = torch.cuda.device_count()
        meta["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        pass
    return meta


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def bootstrap_ci(
    values: list[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI over a list of per-example scores.

    Returns (mean, lo, hi). Used for win-rate, so `values` is typically a list
    of 1.0 / 0.5 / 0.0 per comparison.
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)
