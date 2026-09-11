"""Evaluation, metrics consolidation, and plotting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import MajorityBaseline, RandomBaseline
from src.bert_encoder import BertEncoder, encode_texts, get_tokenizer
from src.data_fma import FMAGraphDataset
from src.data_musiccaps import MusicCapsTextDataset
from src.gnn_model import MusicGNN
from src.utils import (
    ensure_dirs,
    get_device,
    load_config,
    load_json,
    multilabel_f1_auc,
    multiclass_metrics,
    save_json,
    set_seed,
)


def plot_f1_curves(history: Dict[str, List[float]], out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 4))
    if "val_macro_f1" in history:
        plt.plot(history["val_macro_f1"], label="Macro-F1")
    if "val_micro_f1" in history:
        plt.plot(history["val_micro_f1"], label="Micro-F1")
    if "gnn_val_macro_f1" in history:
        plt.plot(history["gnn_val_macro_f1"], label="GNN Macro-F1")
    if "cnn_val_macro_f1" in history:
        plt.plot(history["cnn_val_macro_f1"], label="CNN Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tsne(npz_path: Path, out_path: Path) -> None:
    data = np.load(npz_path)
    z = data["z"]
    labels = data["label_count"]
    try:
        from sklearn.manifold import TSNE

        if len(z) < 2:
            return
        perplexity = min(30, max(2, len(z) // 3))
        emb = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(z)
    except Exception:
        # PCA fallback
        z0 = z - z.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(z0, full_matrices=False)
        emb = z0 @ vt[:2].T
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="viridis", s=40, alpha=0.85)
    plt.colorbar(sc, label="# positive tags")
    plt.title("t-SNE of fused embeddings z")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_retrieval_bars(metrics: Dict[str, Any], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    labels = []
    vals = []
    for direction, d in metrics.items():
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if not isinstance(v, (int, float)):
                continue
            labels.append(f"{direction[:3]}-{k}")
            vals.append(v)
    if not labels:
        plt.close()
        return
    plt.bar(labels, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Recall")
    plt.title("MusicCaps retrieval")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def eval_baselines_musiccaps(cfg: Dict[str, Any]) -> Dict[str, float]:
    splits = load_json(Path(cfg["paths"]["splits"]) / "musiccaps.json")
    n = int(cfg["debug"].get("n_tracks", 0))
    train = splits["train"][: n or None]
    test = splits["test"][: n or None]
    if not train or not test:
        return {}
    y_train = np.stack([r["label"] for r in train])
    y_test = np.stack([r["label"] for r in test])
    maj = MajorityBaseline().fit(y_train)
    y_maj = maj.predict(len(test))
    rnd = RandomBaseline(y_train.shape[1], multilabel=True, seed=cfg["seed"])
    y_rnd = rnd.predict(len(test))
    return {
        "majority_macro_f1": multilabel_f1_auc(y_test, y_maj)["macro_f1"],
        "majority_auc_pr": multilabel_f1_auc(y_test, y_maj)["mean_auc_pr"],
        "random_macro_f1": multilabel_f1_auc(y_test, y_rnd)["macro_f1"],
        "random_auc_pr": multilabel_f1_auc(y_test, y_rnd)["mean_auc_pr"],
    }


def consolidate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    results_dir = Path(cfg["paths"]["results"])
    plots = Path(cfg["paths"]["plots"])
    plots.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Any] = {
        "model": {},
        "baselines": {},
        "notes": "Values from local training runs (debug.n_tracks may apply).",
    }

    # Task 1
    t1 = results_dir / "task1_history.json"
    if t1.exists():
        hist = load_json(t1)
        plot_f1_curves(hist, plots / "task1_f1_curves.png", "Task 1 BERT tag F1")
        metrics["model"]["task1_bert"] = {
            "macro_f1": hist.get("val_macro_f1", [None])[-1],
            "micro_f1": hist.get("val_micro_f1", [None])[-1],
            "auc_pr": hist.get("val_auc_pr", [None])[-1],
        }

    # Task 2
    t2 = results_dir / "task2_history.json"
    if t2.exists():
        hist = load_json(t2)
        plot_f1_curves(hist, plots / "task2_f1_curves.png", "Task 2 GNN vs CNN")
        metrics["model"]["task2_gnn"] = {"macro_f1": hist.get("gnn_val_macro_f1", [None])[-1]}
        metrics["model"]["task2_cnn"] = {"macro_f1": hist.get("cnn_val_macro_f1", [None])[-1]}

    # Task 3
    t3 = results_dir / "task3_ablation_history.json"
    if t3.exists():
        abl = load_json(t3)
        metrics["model"]["task3_ablations"] = {
            mode: {"macro_f1": h.get("val_macro_f1", [None])[-1]} for mode, h in abl.items()
        }
        # plot ablation bars
        plt.figure(figsize=(7, 4))
        modes = list(abl.keys())
        vals = [abl[m].get("val_macro_f1", [0])[-1] for m in modes]
        plt.bar(modes, vals)
        plt.ylabel("Val Macro-F1")
        plt.title("Task 3 fusion ablations")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(plots / "task3_ablations.png", dpi=150)
        plt.close()

    emb = results_dir / "task3_embeddings.npz"
    if emb.exists():
        plot_tsne(emb, plots / "task3_tsne.png")

    # Task 4
    t4 = results_dir / "task4_retrieval.json"
    if t4.exists():
        ret = load_json(t4)
        metrics["model"]["task4_retrieval"] = ret
        plot_retrieval_bars(ret, plots / "task4_retrieval.png")

    try:
        metrics["baselines"].update(eval_baselines_musiccaps(cfg))
    except Exception as e:
        metrics["baselines"]["error"] = str(e)

    # Comparison table inspired by PDF Table 3
    metrics["comparison_table"] = {
        "Random tags": {
            "Macro-F1": metrics["baselines"].get("random_macro_f1"),
            "AUC-PR": metrics["baselines"].get("random_auc_pr"),
            "MAE (emotion)": None,
            "R@5 (retrieval)": None,
        },
        "CNN mel-spec": {
            "Macro-F1": metrics["model"].get("task2_cnn", {}).get("macro_f1"),
            "AUC-PR": None,
            "MAE (emotion)": None,
            "R@5 (retrieval)": None,
        },
        "Task 1: BERT-only": {
            "Macro-F1": metrics["model"].get("task1_bert", {}).get("macro_f1"),
            "AUC-PR": metrics["model"].get("task1_bert", {}).get("auc_pr"),
            "MAE (emotion)": None,
            "R@5 (retrieval)": None,
        },
        "Task 2: GNN-only": {
            "Macro-F1": metrics["model"].get("task2_gnn", {}).get("macro_f1"),
            "AUC-PR": None,
            "MAE (emotion)": None,
            "R@5 (retrieval)": None,
        },
        "Task 3: GNN–BERT": {
            "Macro-F1": metrics["model"]
            .get("task3_ablations", {})
            .get("cross_attention", {})
            .get("macro_f1"),
            "AUC-PR": None,
            "MAE (emotion)": None,
            "R@5 (retrieval)": None,
        },
        "Task 4: Contrastive": {
            "Macro-F1": None,
            "AUC-PR": None,
            "MAE (emotion)": None,
            "R@5 (retrieval)": metrics["model"]
            .get("task4_retrieval", {})
            .get("audio_to_caption", {})
            .get("R@5"),
        },
    }

    save_json(metrics, results_dir / "metrics.json")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    metrics = consolidate(cfg)
    print("Wrote results/metrics.json")
    print(metrics.get("comparison_table", {}))


if __name__ == "__main__":
    main()
