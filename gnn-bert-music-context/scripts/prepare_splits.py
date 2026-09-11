"""Prepare split JSON files from downloaded (or synthetic) raw data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_deam import prepare_deam_splits
from src.data_fma import prepare_fma_table, write_fma_splits, write_fma_context_pairs
from src.data_musiccaps import prepare_musiccaps_splits
from src.utils import load_config, ensure_dirs, load_json, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--prefer-synthetic-if-missing", action="store_true", default=False)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = Path(cfg["paths"]["raw"])
    splits = Path(cfg["paths"]["splits"])
    processed = Path(cfg["paths"]["processed"])
    subset = cfg["data"]["fma_subset"]

    # FMA
    fma_out = None
    try:
        df, genre_to_id = prepare_fma_table(raw, subset=subset)
        fma_out = write_fma_splits(df, splits, subset)
        print(f"FMA {subset}: {len(df)} tracks, {len(genre_to_id)} genres")
    except Exception as e:
        print(f"FMA splits skipped ({e}). Using local subset if present.")
        if not (splits / f"fma_{subset}.json").exists():
            print("Run: python scripts/download_fma.py && python scripts/prepare_splits.py")

    # MusicCaps
    mc_out = None
    try:
        mc_out = prepare_musiccaps_splits(
            raw,
            splits,
            top_k=cfg["data"]["top_k_aspects"],
            audio_dir=raw / "musiccaps" / "audio",
        )
        print(f"MusicCaps: total={mc_out['n_total']} downloaded={mc_out['n_downloaded']}")
    except Exception as e:
        print(f"MusicCaps splits skipped ({e}).")

    # Context pairs for Task 3/4: prefer MusicCaps with audio graphs; else/also FMA
    context = {"vocab": [], "train": [], "val": [], "test": [], "n_downloaded": 0, "n_total": 0}
    if mc_out and mc_out.get("n_downloaded", 0) > 0:
        # Keep only downloaded rows for graph tasks
        def _dl(recs):
            return [r for r in recs if r.get("downloaded")]

        context = {
            "vocab": mc_out["vocab"],
            "train": _dl(mc_out["train"]),
            "val": _dl(mc_out["val"]),
            "test": _dl(mc_out["test"]),
            "n_downloaded": mc_out["n_downloaded"],
            "n_total": mc_out["n_total"],
            "source": "musiccaps",
        }
    if (not context["train"]) and (splits / f"fma_{subset}.json").exists():
        fma_splits = load_json(splits / f"fma_{subset}.json")
        # Graphs may not exist yet — prepare_splits may be run again after build_graphs
        context = write_fma_context_pairs(
            fma_splits,
            splits,
            processed,
            subset=subset,
            max_per_split=int(cfg.get("debug", {}).get("max_fma_graphs", 0) or 0) or 0,
        )
        print(f"FMA context pairs: {context.get('n_downloaded', 0)}")
    elif context["train"]:
        save_json(context, splits / "context_pairs.json")
        print(f"Wrote context_pairs.json from MusicCaps audio ({len(context['train'])} train)")

    # If MusicCaps text exists but audio sparse, still keep full captions for Task 1 in musiccaps.json
    # DEAM
    deam = prepare_deam_splits(raw, splits)
    print(f"DEAM status={deam.get('status')} train={len(deam.get('train', []))}")


if __name__ == "__main__":
    main()
