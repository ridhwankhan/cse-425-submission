"""MusicCaps dataset helpers."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import save_json, load_json


def parse_aspect_list(raw: Any) -> List[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [str(a).strip().lower() for a in raw if str(a).strip()]
    s = str(raw).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(a).strip().lower() for a in parsed if str(a).strip()]
    except Exception:
        pass
    # comma / semicolon separated
    parts = re.split(r"[,;|]", s)
    return [p.strip().lower().strip("'\"[]") for p in parts if p.strip()]


def build_aspect_vocab(aspects_per_row: List[List[str]], top_k: int = 50) -> List[str]:
    c = Counter()
    for asp in aspects_per_row:
        c.update(asp)
    return [a for a, _ in c.most_common(top_k)]


def aspects_to_multihot(aspects: List[str], vocab: List[str]) -> np.ndarray:
    idx = {a: i for i, a in enumerate(vocab)}
    y = np.zeros(len(vocab), dtype=np.float32)
    for a in aspects:
        if a in idx:
            y[idx[a]] = 1.0
    return y


def load_musiccaps_table(raw_dir: Path) -> pd.DataFrame:
    csv_path = raw_dir / "musiccaps" / "musiccaps.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    # Try HF cache export
    raise FileNotFoundError(f"MusicCaps CSV not found at {csv_path}. Run scripts/download_musiccaps.py")


def prepare_musiccaps_splits(
    raw_dir: Path,
    splits_dir: Path,
    top_k: int = 50,
    audio_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    df = load_musiccaps_table(raw_dir)
    audio_dir = audio_dir or (raw_dir / "musiccaps" / "audio")
    aspects_all = [parse_aspect_list(r) for r in df.get("aspect_list", df.get("aspects", []))]
    if not aspects_all or all(len(a) == 0 for a in aspects_all):
        # fallback: tokenize captions into pseudo-aspects
        aspects_all = []
        for cap in df["caption"].astype(str).tolist():
            toks = re.findall(r"[a-zA-Z]{3,}", cap.lower())
            aspects_all.append(toks[:20])
    vocab = build_aspect_vocab(aspects_all, top_k=top_k)

    records = []
    for i, row in df.iterrows():
        ytid = str(row.get("ytid", row.get("youtube_id", f"id_{i}")))
        caption = str(row.get("caption", ""))
        aspects = aspects_all[i] if i < len(aspects_all) else parse_aspect_list(row.get("aspect_list"))
        audio_path = audio_dir / f"{ytid}.wav"
        downloaded = audio_path.exists()
        is_eval = bool(row.get("is_audioset_eval", False))
        records.append(
            {
                "ytid": ytid,
                "caption": caption,
                "aspects": aspects,
                "label": aspects_to_multihot(aspects, vocab).tolist(),
                "audio_path": str(audio_path),
                "downloaded": downloaded,
                "is_audioset_eval": is_eval,
                "text": caption if caption else ", ".join(aspects),
            }
        )

    # Prefer official audioset eval as test; rest train/val 90/10
    test = [r for r in records if r["is_audioset_eval"]]
    rest = [r for r in records if not r["is_audioset_eval"]]
    if not test:
        n = len(records)
        test = records[int(0.85 * n) :]
        rest = records[: int(0.85 * n)]
    n_val = max(1, int(0.1 * len(rest)))
    val = rest[:n_val]
    train = rest[n_val:]

    out = {
        "vocab": vocab,
        "train": train,
        "val": val,
        "test": test,
        "n_downloaded": sum(1 for r in records if r["downloaded"]),
        "n_total": len(records),
    }
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_json(out, splits_dir / "musiccaps.json")
    save_json({"vocab": vocab}, splits_dir / "musiccaps_aspect_vocab.json")
    return out


class MusicCapsTextDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], require_audio: bool = False):
        self.records = [r for r in records if (r.get("downloaded") or not require_audio)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        return {
            "ytid": r["ytid"],
            "text": r.get("text") or r.get("caption") or "",
            "label": torch.tensor(r["label"], dtype=torch.float32),
            "audio_path": r.get("audio_path", ""),
            "downloaded": bool(r.get("downloaded", False)),
        }


class MusicCapsGraphDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], processed_dir: Path, require_audio: bool = True):
        self.processed_dir = Path(processed_dir)
        kept = []
        for r in records:
            gpath = self.processed_dir / "musiccaps" / f"{r['ytid']}_segment.pt"
            has_graph = gpath.exists()
            has_audio = bool(r.get("downloaded"))
            if require_audio and not (has_graph or has_audio):
                continue
            if require_audio and not has_graph:
                continue
            if not require_audio and not has_graph and not has_audio:
                # text-only callers should use MusicCapsTextDataset instead
                continue
            if has_graph:
                kept.append(r)
        self.records = kept

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        ytid = r["ytid"]
        gpath = self.processed_dir / "musiccaps" / f"{ytid}_segment.pt"
        if not gpath.exists():
            raise FileNotFoundError(f"Missing MusicCaps graph cache: {gpath}")
        g = torch.load(gpath, map_location="cpu", weights_only=False)
        return {
            "ytid": ytid,
            "x": g["x"].float(),
            "edge_index": g["edge_index"].long(),
            "text": r.get("text") or r.get("caption") or "",
            "label": torch.tensor(r["label"], dtype=torch.float32),
        }
