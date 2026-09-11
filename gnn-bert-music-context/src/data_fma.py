"""FMA dataset helpers and PyTorch dataset."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import ROOT, load_json, save_json


GENRES_SMALL = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]


def fma_audio_path(raw_dir: Path, track_id: int, subset: str = "small") -> Path:
    tid = f"{int(track_id):06d}"
    base = raw_dir / f"fma_{subset}" / tid[:3]
    mp3 = base / f"{tid}.mp3"
    wav = base / f"{tid}.wav"
    if mp3.exists():
        return mp3
    if wav.exists():
        return wav
    return mp3  # default expected path


def build_fma_text(row: pd.Series) -> str:
    """Build BERT text WITHOUT genre labels (no leakage)."""
    parts = []
    for key in ("track_title", "album_title", "artist_name"):
        val = row.get(key, "")
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    tags = row.get("tags", "")
    if isinstance(tags, str) and tags.strip():
        # tags may be a list string
        try:
            parsed = ast.literal_eval(tags)
            if isinstance(parsed, (list, tuple)):
                parts.extend([str(t) for t in parsed if str(t).lower() not in {g.lower() for g in GENRES_SMALL}])
            else:
                parts.append(tags)
        except Exception:
            parts.append(tags)
    text = " . ".join(parts) if parts else "music track"
    # Strip accidental genre words
    for g in GENRES_SMALL:
        text = text.replace(g, "").replace(g.lower(), "")
    return " ".join(text.split())


def load_fma_metadata(raw_dir: Path) -> pd.DataFrame:
    meta = raw_dir / "fma_metadata" / "tracks.csv"
    if not meta.exists():
        raise FileNotFoundError(f"Missing FMA metadata at {meta}")
    tracks = pd.read_csv(meta, header=[0, 1], index_col=0)
    return tracks


def prepare_fma_table_from_hf(raw_dir: Path, manifest_path: Path, subset: str = "small") -> Tuple[pd.DataFrame, Dict[str, int]]:
    data = load_json(manifest_path)
    # Prefer official FMA genre_top when metadata is available (HF ID3 tags are noisy).
    official_genre: Dict[int, str] = {}
    tracks_csv = raw_dir / "fma_metadata" / "tracks.csv"
    if tracks_csv.exists():
        try:
            tracks = pd.read_csv(tracks_csv, header=[0, 1], index_col=0)
            for tid, row in tracks.iterrows():
                g = row.get(("track", "genre_top"))
                if pd.notna(g):
                    official_genre[int(tid)] = str(g)
            print(f"Loaded official genre_top for {len(official_genre)} tracks")
        except Exception as e:
            print(f"Could not read official tracks.csv: {e}")

    rows = []
    for i, r in enumerate(data.get("records", [])):
        tid = int(r["track_id"])
        genre = official_genre.get(tid) or str(r.get("genre", "Unknown"))
        mapped = None
        for g in GENRES_SMALL:
            if g.lower() in genre.lower() or genre.lower() in g.lower():
                mapped = g
                break
        if mapped is None:
            continue  # keep FMA-small 8-way taxonomy only
        genre = mapped
        audio = Path(r.get("audio_path") or fma_audio_path(raw_dir, tid, subset))
        if not audio.exists():
            continue
        rows.append(
            {
                "track_id": tid,
                "subset": subset,
                "split": ["training", "validation", "test"][0 if i % 10 < 7 else (1 if i % 10 < 9 else 2)],
                "genre_top": genre,
                "track_title": r.get("title", f"track_{tid}"),
                "tags": "",
                "album_title": r.get("album", ""),
                "artist_name": r.get("artist", ""),
                "audio_path": str(audio),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No audio files referenced by {manifest_path}")
    df = pd.DataFrame(rows)
    genres = [g for g in GENRES_SMALL if g in set(df["genre_top"])]
    # Keep stable 8-class order even if some are missing
    if len(set(df["genre_top"])) >= 4:
        genres = [g for g in GENRES_SMALL if (df["genre_top"] == g).any()]
    genre_to_id = {g: i for i, g in enumerate(genres)}
    df["genre_id"] = df["genre_top"].map(genre_to_id)
    df = df.dropna(subset=["genre_id"]).copy()
    df["genre_id"] = df["genre_id"].astype(int)
    df["text"] = df.apply(build_fma_text, axis=1)
    return df, genre_to_id


def prepare_fma_table(raw_dir: Path, subset: str = "small") -> Tuple[pd.DataFrame, Dict[str, int]]:
    meta = raw_dir / "fma_metadata" / "tracks.csv"
    hf_manifest = raw_dir / "fma_hf_subset.json"
    if not meta.exists():
        if hf_manifest.exists():
            return prepare_fma_table_from_hf(raw_dir, hf_manifest, subset=subset)
        raise FileNotFoundError(f"Missing FMA metadata at {meta}")

    tracks = load_fma_metadata(raw_dir)
    subset_col = tracks[("set", "subset")]
    split_col = tracks[("set", "split")]
    genre = tracks[("track", "genre_top")]
    title = tracks[("track", "title")]
    tags = tracks[("track", "tags")] if ("track", "tags") in tracks.columns else ""
    album = tracks[("album", "title")] if ("album", "title") in tracks.columns else ""
    artist = tracks[("artist", "name")] if ("artist", "name") in tracks.columns else ""

    df = pd.DataFrame(
        {
            "track_id": tracks.index.astype(int),
            "subset": subset_col.values,
            "split": split_col.values,
            "genre_top": genre.values,
            "track_title": title.values,
            "tags": tags.values if not isinstance(tags, str) else tags,
            "album_title": album.values if not isinstance(album, str) else album,
            "artist_name": artist.values if not isinstance(artist, str) else artist,
        }
    )
    if subset == "small":
        df = df[df["subset"] == "small"].copy()
    else:
        df = df[df["subset"].isin(["small", "medium"])].copy()
    df = df.dropna(subset=["genre_top"])
    df["genre_top"] = df["genre_top"].astype(str)
    df["audio_path"] = df["track_id"].apply(
        lambda t: str(fma_audio_path(raw_dir, int(t), subset if subset == "small" else "medium"))
    )
    df = df[df["audio_path"].map(lambda p: Path(p).exists())].copy()
    if len(df) == 0:
        if hf_manifest.exists():
            return prepare_fma_table_from_hf(raw_dir, hf_manifest, subset=subset)
        raise FileNotFoundError("FMA metadata found but no local audio files.")
    genres = sorted(df["genre_top"].unique().tolist())
    genre_to_id = {g: i for i, g in enumerate(genres)}
    df["genre_id"] = df["genre_top"].map(genre_to_id)
    df["text"] = df.apply(build_fma_text, axis=1)
    return df, genre_to_id


def write_fma_splits(df: pd.DataFrame, splits_dir: Path, subset: str) -> Dict[str, Any]:
    splits_dir.mkdir(parents=True, exist_ok=True)
    out = {"subset": subset, "genres": sorted(df["genre_top"].unique().tolist()), "train": [], "val": [], "test": []}
    split_map = {"training": "train", "validation": "val", "test": "test"}
    for _, row in df.iterrows():
        key = split_map.get(str(row["split"]), None)
        if key is None:
            continue
        out[key].append(
            {
                "track_id": int(row["track_id"]),
                "genre_id": int(row["genre_id"]),
                "genre": row["genre_top"],
                "text": row["text"],
                "audio_path": row["audio_path"],
            }
        )
    path = splits_dir / f"fma_{subset}.json"
    save_json(out, path)
    return out


def write_fma_context_pairs(
    fma_splits: Dict[str, Any],
    splits_dir: Path,
    processed_dir: Path,
    subset: str = "small",
    max_per_split: int = 0,
) -> Dict[str, Any]:
    """Build MusicCaps-compatible paired graph+text records from real FMA tracks.

    Used for Task 3/4 when MusicCaps YouTube audio is scarce, while still using
    official Table-1 audio (FMA) + text (metadata without genre leakage).
    """
    genres = fma_splits.get("genres") or GENRES_SMALL
    vocab = list(genres)
    g2i = {g: i for i, g in enumerate(vocab)}

    def convert(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in recs:
            tid = int(r["track_id"])
            gpath = processed_dir / "fma" / subset / f"{tid:06d}_segment.pt"
            if not gpath.exists():
                continue
            label = [0.0] * len(vocab)
            gid = int(r.get("genre_id", g2i.get(r.get("genre", ""), 0)))
            if 0 <= gid < len(vocab):
                label[gid] = 1.0
            ytid = f"fma{tid:06d}"
            # Mirror graph into musiccaps cache name for the shared loader
            mc_stem = processed_dir / "musiccaps" / ytid
            mc_seg = Path(str(mc_stem) + "_segment.pt")
            if not mc_seg.exists():
                mc_seg.parent.mkdir(parents=True, exist_ok=True)
                try:
                    import shutil

                    shutil.copy2(gpath, mc_seg)
                    mel_src = processed_dir / "fma" / subset / f"{tid:06d}_mel.pt"
                    if mel_src.exists():
                        shutil.copy2(mel_src, Path(str(mc_stem) + "_mel.pt"))
                except Exception:
                    continue
            out.append(
                {
                    "ytid": ytid,
                    "caption": r.get("text", "music track"),
                    "text": r.get("text", "music track"),
                    "aspects": [r.get("genre", vocab[gid] if gid < len(vocab) else "music")],
                    "label": label,
                    "audio_path": r.get("audio_path", ""),
                    "downloaded": True,
                    "is_audioset_eval": False,
                    "source": "fma",
                }
            )
            if max_per_split and len(out) >= max_per_split:
                break
        return out

    train = convert(fma_splits.get("train", []))
    val = convert(fma_splits.get("val", []))
    test = convert(fma_splits.get("test", []))
    out = {
        "vocab": vocab,
        "train": train,
        "val": val,
        "test": test,
        "n_downloaded": len(train) + len(val) + len(test),
        "n_total": len(train) + len(val) + len(test),
        "source": "fma_context_pairs",
    }
    save_json(out, splits_dir / "context_pairs.json")
    return out


class FMAGraphDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        processed_dir: Path,
        subset: str = "small",
        max_mel_frames: int = 256,
        require_graph: bool = True,
    ):
        self.processed_dir = Path(processed_dir)
        self.subset = subset
        self.max_mel_frames = max_mel_frames
        if require_graph:
            self.records = [
                r for r in records if (self.processed_dir / "fma" / subset / f"{int(r['track_id']):06d}_segment.pt").exists()
            ]
        else:
            self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def _graph_path(self, track_id: int) -> Path:
        return self.processed_dir / "fma" / self.subset / f"{int(track_id):06d}_segment.pt"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        tid = int(rec["track_id"])
        gpath = self._graph_path(tid)
        if not gpath.exists():
            raise FileNotFoundError(f"Missing FMA graph cache: {gpath}")
        g = torch.load(gpath, map_location="cpu", weights_only=False)
        mel_path = self.processed_dir / "fma" / self.subset / f"{tid:06d}_mel.pt"
        if mel_path.exists():
            mel = torch.load(mel_path, map_location="cpu", weights_only=False)
        else:
            mel = torch.zeros(128, self.max_mel_frames)
        if mel.size(1) > self.max_mel_frames:
            mel = mel[:, : self.max_mel_frames]
        elif mel.size(1) < self.max_mel_frames:
            pad = torch.zeros(mel.size(0), self.max_mel_frames - mel.size(1))
            mel = torch.cat([mel, pad], dim=1)
        return {
            "track_id": tid,
            "x": g["x"].float(),
            "edge_index": g["edge_index"].long(),
            "mel": mel.float(),
            "label": int(rec["genre_id"]),
            "text": rec.get("text", "music"),
            "genre": rec.get("genre", ""),
        }
