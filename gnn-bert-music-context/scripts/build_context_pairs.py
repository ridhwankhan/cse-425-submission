"""Prefer MusicCaps graph pairs for Task 3/4 when enough clips exist."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fma import write_fma_context_pairs
from src.utils import load_config, load_json, save_json


def main():
    cfg = load_config()
    splits = Path(cfg["paths"]["splits"])
    processed = Path(cfg["paths"]["processed"])
    mc = load_json(splits / "musiccaps.json")

    def with_graph(recs):
        out = []
        for r in recs:
            g = processed / "musiccaps" / f"{r['ytid']}_segment.pt"
            if g.exists():
                rr = dict(r)
                rr["downloaded"] = True
                out.append(rr)
        return out

    train = with_graph(mc["train"])
    val = with_graph(mc["val"])
    test = with_graph(mc["test"])
    # MusicCaps downloads often land in eval/test; redistribute into train/val/test
    all_g = train + val + test
    print(f"musiccaps graphs raw train/val/test = {len(train)}/{len(val)}/{len(test)} total={len(all_g)}")
    if len(all_g) >= 40:
        n = len(all_g)
        n_train = max(1, int(0.7 * n))
        n_val = max(1, int(0.15 * n))
        train, val, test = all_g[:n_train], all_g[n_train : n_train + n_val], all_g[n_train + n_val :]
        if not test:
            test = val[:1] or train[:1]
        ctx = {
            "vocab": mc["vocab"],
            "train": train,
            "val": val,
            "test": test,
            "n_downloaded": len(all_g),
            "n_total": mc["n_total"],
            "source": "musiccaps",
        }
        save_json(ctx, splits / "context_pairs.json")
        print(f"Wrote MusicCaps context_pairs.json n={ctx['n_downloaded']} splits={len(train)}/{len(val)}/{len(test)}")
    else:
        fma = load_json(splits / f"fma_{cfg['data']['fma_subset']}.json")
        ctx = write_fma_context_pairs(fma, splits, processed, subset=cfg["data"]["fma_subset"])
        print(f"Wrote FMA context_pairs.json n={ctx.get('n_downloaded')}")


if __name__ == "__main__":
    main()
