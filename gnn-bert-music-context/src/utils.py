"""Shared helpers: config, device, seeding, paths, metrics."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against project root
    for key, val in list(cfg.get("paths", {}).items()):
        if key == "root":
            cfg["paths"][key] = str(ROOT)
        else:
            p = Path(val)
            if not p.is_absolute():
                cfg["paths"][key] = str(ROOT / p)
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ("raw", "processed", "splits", "results", "checkpoints", "plots", "examples"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["debug"]["synthetic_dir"]).mkdir(parents=True, exist_ok=True)
    if not Path(cfg["debug"]["synthetic_dir"]).is_absolute():
        (ROOT / cfg["debug"]["synthetic_dir"]).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(cfg: Dict[str, Any]):
    import torch

    pref = cfg.get("device", "auto")
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def multilabel_f1_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    from sklearn.metrics import f1_score, average_precision_score

    y_pred = (y_prob >= threshold).astype(np.int32)
    out: Dict[str, float] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    try:
        # Skip labels with no positives in y_true for AP stability
        aps = []
        for k in range(y_true.shape[1]):
            if y_true[:, k].sum() == 0:
                continue
            aps.append(average_precision_score(y_true[:, k], y_prob[:, k]))
        out["mean_auc_pr"] = float(np.mean(aps)) if aps else 0.0
    except Exception:
        out["mean_auc_pr"] = 0.0
    return out


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    return out


def mae_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def recall_at_k(sim: np.ndarray, k: int) -> float:
    """sim: [N, N] similarity; diagonal is correct pair."""
    n = sim.shape[0]
    if n == 0:
        return 0.0
    hits = 0
    for i in range(n):
        top = np.argsort(-sim[i])[:k]
        if i in top:
            hits += 1
    return hits / n
