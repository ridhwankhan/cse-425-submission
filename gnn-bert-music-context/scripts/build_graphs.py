"""Batch-build and cache music structure graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph_builder import build_graphs_from_audio, save_graph_pt
from src.utils import load_config, ensure_dirs, load_json


def process_one(audio_path: Path, out_stem: Path, cfg: dict) -> bool:
    seg_path = Path(str(out_stem) + "_segment.pt")
    if seg_path.exists():
        return True
    if not audio_path.exists():
        return False
    try:
        graphs = build_graphs_from_audio(audio_path, cfg)
        save_graph_pt(graphs["segment"], seg_path)
        if "chord" in graphs:
            save_graph_pt(graphs["chord"], Path(str(out_stem) + "_chord.pt"))
        # cache mel for CNN baseline
        mel = graphs.get("mel")
        if mel is not None:
            torch.save(torch.from_numpy(np.asarray(mel, dtype=np.float32)), Path(str(out_stem) + "_mel.pt"))
        return True
    except Exception as e:
        print(f"fail {audio_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fma", "musiccaps", "deam", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    processed = Path(cfg["paths"]["processed"])
    splits = Path(cfg["paths"]["splits"])
    limit = args.limit

    from concurrent.futures import ThreadPoolExecutor, as_completed

    datasets = ["fma", "musiccaps", "deam"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        ok = fail = 0
        jobs = []
        if ds == "fma":
            subset = cfg["data"]["fma_subset"]
            sp = load_json(splits / f"fma_{subset}.json")
            # Ensure train/val/test all get graph coverage when limiting
            if limit and limit > 0:
                n_train = max(1, int(0.7 * limit))
                n_val = max(1, int(0.15 * limit))
                n_test = max(1, limit - n_train - n_val)
                recs = (
                    sp["train"][:n_train]
                    + sp["val"][:n_val]
                    + sp["test"][:n_test]
                )
            else:
                recs = sp["train"] + sp["val"] + sp["test"]
            out_dir = processed / "fma" / subset
            out_dir.mkdir(parents=True, exist_ok=True)
            for r in recs:
                tid = int(r["track_id"])
                audio = Path(r["audio_path"])
                stem = out_dir / f"{tid:06d}"
                jobs.append((audio, stem))
        elif ds == "musiccaps":
            sp = load_json(splits / "musiccaps.json")
            downloaded = [r for r in (sp["train"] + sp["val"] + sp["test"]) if r.get("downloaded")]
            # Prefer building for all downloaded clips (already capped by download limit)
            if limit:
                downloaded = downloaded[:limit]
            out_dir = processed / "musiccaps"
            out_dir.mkdir(parents=True, exist_ok=True)
            for r in downloaded:
                audio = Path(r["audio_path"])
                if not audio.exists():
                    alt = Path(cfg["paths"]["raw"]) / "synthetic" / "musiccaps" / "audio" / f"{r['ytid']}.wav"
                    if alt.exists():
                        audio = alt
                stem = out_dir / r["ytid"]
                jobs.append((audio, stem))
        elif ds == "deam":
            sp = load_json(splits / "deam.json")
            recs = sp.get("train", []) + sp.get("val", []) + sp.get("test", [])
            if limit:
                recs = recs[:limit]
            out_dir = processed / "deam"
            out_dir.mkdir(parents=True, exist_ok=True)
            for r in recs:
                audio = Path(r["audio_path"])
                stem = out_dir / str(r["song_id"])
                jobs.append((audio, stem))

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(process_one, audio, stem, cfg): (audio, stem) for audio, stem in jobs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{ds} graphs"):
                if fut.result():
                    ok += 1
                else:
                    fail += 1
        print(f"{ds}: ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
