"""Export >=20 example graphs as .pt and .json for submission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph_builder import graph_to_jsonable, save_graph_json
from src.utils import load_config, ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    processed = Path(cfg["paths"]["processed"])
    examples = Path(cfg["paths"]["examples"])
    examples.mkdir(parents=True, exist_ok=True)

    candidates = list((processed / "musiccaps").glob("*_segment.pt"))
    subset = cfg["data"]["fma_subset"]
    candidates += list((processed / "fma" / subset).glob("*_segment.pt"))
    candidates += list((processed / "deam").glob("*_segment.pt"))

    if not candidates:
        raise SystemExit("No cached graphs found. Run scripts/build_graphs.py first.")

    n = min(args.n, len(candidates))
    for i, src in enumerate(candidates[:n]):
        g = torch.load(src, map_location="cpu", weights_only=False)
        # normalize to dict with numpy-friendly fields
        gdict = {
            "x": g["x"].numpy() if torch.is_tensor(g["x"]) else g["x"],
            "edge_index": g["edge_index"].numpy() if torch.is_tensor(g["edge_index"]) else g["edge_index"],
            "edge_weight": (
                g["edge_weight"].numpy()
                if "edge_weight" in g and torch.is_tensor(g["edge_weight"])
                else g.get("edge_weight", [])
            ),
            "num_nodes": int(g.get("num_nodes", g["x"].shape[0])),
            "graph_type": g.get("graph_type", "segment"),
        }
        stem = examples / f"example_{i+1:02d}_{src.stem}"
        torch.save(g, Path(str(stem) + ".pt"))
        save_graph_json(gdict, Path(str(stem) + ".json"))
    print(f"Exported {n} example graphs to {examples}")


if __name__ == "__main__":
    main()
