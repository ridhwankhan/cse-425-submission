"""DEAM valence/arousal dataset helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import save_json


def prepare_deam_splits(raw_dir: Path, splits_dir: Path) -> Dict[str, Any]:
    """
    Expect either:
      data/raw/deam/annotations/annotations averaged per song.csv
    or synthetic CSV written by make_synthetic_data.
    """
    raw_dir = Path(raw_dir)
    candidates = [
        raw_dir / "deam" / "static_annotations.csv",
        raw_dir / "deam" / "annotations" / "annotations averaged per song.csv",
        raw_dir
        / "deam"
        / "annotations"
        / "annotations averaged per song"
        / "song_level"
        / "static_annotations_averaged_songs_1_2000.csv",
        raw_dir / "synthetic" / "deam" / "static_annotations.csv",
    ]
    ann = None
    for c in candidates:
        if c.exists():
            ann = c
            break
    if ann is None:
        # Empty placeholder
        out = {"train": [], "val": [], "test": [], "status": "missing"}
        save_json(out, Path(splits_dir) / "deam.json")
        return out

    df = pd.read_csv(ann)
    # Normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    id_col = cols.get("song_id") or cols.get("id") or list(df.columns)[0]
    # DEAM uses valence_mean / arousal_mean often
    v_col = None
    a_col = None
    for c in df.columns:
        cl = c.lower()
        if "valence" in cl and v_col is None:
            v_col = c
        if "arousal" in cl and a_col is None:
            a_col = c
    if v_col is None or a_col is None:
        # take last two numeric
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        v_col, a_col = num[-2], num[-1]

    records = []
    audio_root = raw_dir / "deam" / "audio"
    synth_audio = raw_dir / "synthetic" / "deam" / "audio"
    for _, row in df.iterrows():
        raw_sid = row[id_col]
        try:
            sid = str(int(float(raw_sid)))
        except Exception:
            sid = str(raw_sid).strip()
        audio_path = audio_root / f"{sid}.wav"
        if not audio_path.exists():
            audio_path = audio_root / f"{sid}.mp3"
        if not audio_path.exists():
            audio_path = synth_audio / f"{sid}.wav"
        records.append(
            {
                "song_id": sid,
                "valence": float(row[v_col]),
                "arousal": float(row[a_col]),
                "audio_path": str(audio_path),
                "exists": audio_path.exists(),
                "text": f"music clip {sid}",
            }
        )

    n = len(records)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    out = {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
        "status": "ok",
        "source": str(ann),
    }
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    save_json(out, Path(splits_dir) / "deam.json")
    return out


class DEAMDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], processed_dir: Path):
        self.records = [r for r in records if r.get("exists", True)]
        self.processed_dir = Path(processed_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        sid = r["song_id"]
        gpath = self.processed_dir / "deam" / f"{sid}_segment.pt"
        if gpath.exists():
            g = torch.load(gpath, map_location="cpu", weights_only=False)
        else:
            x = torch.zeros(1, 2 * (128 + 12 + 13), dtype=torch.float32)
            g = {"x": x, "edge_index": torch.tensor([[0], [0]], dtype=torch.long)}
        return {
            "song_id": sid,
            "x": g["x"].float(),
            "edge_index": g["edge_index"].long(),
            "text": r.get("text", "music"),
            "valence": float(r["valence"]),
            "arousal": float(r["arousal"]),
            "emotion": torch.tensor([r["valence"], r["arousal"]], dtype=torch.float32),
        }
