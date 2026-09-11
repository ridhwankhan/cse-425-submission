"""Music structure graph construction (segment + chord-transition)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .audio_features import load_and_extract


def _cosine_sim_matrix(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = X.astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    Xn = X / norms
    return Xn @ Xn.T


def build_segment_graph(
    node_features: np.ndarray,
    mfcc_means: Optional[np.ndarray] = None,
    tau: float = 0.85,
    use_temporal: bool = True,
    use_similarity: bool = True,
) -> Dict[str, Any]:
    """
    Nodes = time segments. Edges = temporal ±1 and/or similarity > tau.
    """
    n = node_features.shape[0]
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []

    if use_temporal and n >= 2:
        for i in range(n - 1):
            edges.append((i, i + 1))
            weights.append(1.0)
            edges.append((i + 1, i))
            weights.append(1.0)

    sim_src = mfcc_means if mfcc_means is not None else node_features
    if use_similarity and n >= 2:
        S = _cosine_sim_matrix(sim_src)
        for i in range(n):
            for j in range(i + 1, n):
                if S[i, j] > tau:
                    edges.append((i, j))
                    weights.append(float(S[i, j]))
                    edges.append((j, i))
                    weights.append(float(S[i, j]))

    if not edges:
        # Self-loop so GNN does not break on single-node / empty edge cases
        for i in range(n):
            edges.append((i, i))
            weights.append(1.0)

    edge_index = np.asarray(edges, dtype=np.int64).T  # [2, E]
    edge_weight = np.asarray(weights, dtype=np.float32)
    return {
        "x": node_features.astype(np.float32),
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "num_nodes": n,
        "graph_type": "segment",
    }


# Major triad chroma templates (C, C#, ..., B)
_CHORD_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _chord_templates() -> np.ndarray:
    templates = []
    for root in range(12):
        t = np.zeros(12, dtype=np.float32)
        t[root] = 1.0
        t[(root + 4) % 12] = 0.8
        t[(root + 7) % 12] = 0.9
        templates.append(t / (np.linalg.norm(t) + 1e-8))
    return np.stack(templates, axis=0)


def estimate_chord_sequence(chroma: np.ndarray) -> List[str]:
    """chroma: [12, T] -> list of chord labels per frame."""
    templates = _chord_templates()
    C = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)
    scores = templates @ C  # [12, T]
    idxs = scores.argmax(axis=0)
    return [_CHORD_NAMES[i] for i in idxs]


def build_chord_transition_graph(chroma: np.ndarray) -> Dict[str, Any]:
    """
    Nodes = unique chords observed; edges = transitions weighted by count.
    Node features = one-hot / chroma template.
    """
    seq = estimate_chord_sequence(chroma)
    unique = sorted(set(seq))
    if not unique:
        unique = ["C"]
        seq = ["C"]
    idx = {c: i for i, c in enumerate(unique)}
    templates = _chord_templates()
    name_to_pc = {n: i for i, n in enumerate(_CHORD_NAMES)}
    x = np.stack([templates[name_to_pc[c]] for c in unique], axis=0)

    counts: Dict[Tuple[int, int], float] = {}
    for a, b in zip(seq[:-1], seq[1:]):
        i, j = idx[a], idx[b]
        counts[(i, j)] = counts.get((i, j), 0.0) + 1.0

    edges = []
    weights = []
    for (i, j), w in counts.items():
        edges.append((i, j))
        weights.append(w)
    if not edges:
        for i in range(len(unique)):
            edges.append((i, i))
            weights.append(1.0)

    edge_index = np.asarray(edges, dtype=np.int64).T
    edge_weight = np.asarray(weights, dtype=np.float32)
    return {
        "x": x.astype(np.float32),
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "num_nodes": len(unique),
        "chord_labels": unique,
        "graph_type": "chord",
    }


def graph_to_torch(g: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out = {
        "x": torch.from_numpy(np.asarray(g["x"], dtype=np.float32)),
        "edge_index": torch.from_numpy(np.asarray(g["edge_index"], dtype=np.int64)),
        "edge_weight": torch.from_numpy(np.asarray(g["edge_weight"], dtype=np.float32)),
        "num_nodes": int(g["num_nodes"]),
    }
    if "chord_labels" in g:
        out["chord_labels"] = g["chord_labels"]
    out["graph_type"] = g.get("graph_type", "segment")
    return out


def graph_to_jsonable(g: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "x": np.asarray(g["x"]).tolist(),
        "edge_index": np.asarray(g["edge_index"]).tolist(),
        "edge_weight": np.asarray(g["edge_weight"]).tolist(),
        "num_nodes": int(g["num_nodes"]),
        "graph_type": g.get("graph_type", "segment"),
        "chord_labels": g.get("chord_labels"),
    }


def build_graphs_from_audio(path: Union[str, Path], cfg: dict) -> Dict[str, Any]:
    feats = load_and_extract(path, cfg)
    # MFCC mean part of node features for similarity (last 2*n_mfcc dims include mean+std;
    # also pass dedicated mfcc means via starts windows — use node_features directly)
    g_cfg = cfg.get("graph", {})
    seg = build_segment_graph(
        feats["node_features"],
        tau=float(g_cfg.get("similarity_tau", 0.85)),
        use_temporal=bool(g_cfg.get("use_temporal_edges", True)),
        use_similarity=bool(g_cfg.get("use_similarity_edges", True)),
    )
    out = {"segment": seg, "features_meta": {"starts": feats["starts"].tolist(), "duration": float(feats["duration"])}}
    if g_cfg.get("chord_enabled", True):
        out["chord"] = build_chord_transition_graph(feats["chroma"])
    out["mel"] = feats["mel"]
    return out


def save_graph_pt(g: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_to_torch(g), path)


def save_graph_json(g: Dict[str, Any], path: Union[str, Path]) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_to_jsonable(g), f)
