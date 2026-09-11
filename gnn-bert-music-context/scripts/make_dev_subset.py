"""Build a small local development audio/text subset when full downloads are not yet available."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fma import GENRES_SMALL
from src.utils import load_config, ensure_dirs, save_json


ASPECT_VOCAB = [
    "piano", "guitar", "drums", "bass", "synth", "vocal", "female", "male",
    "melody", "harmony", "upbeat", "melancholic", "acoustic", "electronic",
    "fast", "slow", "soft", "loud", "jazz", "rock", "pop", "folk", "ambient",
    "choir", "strings", "brass", "percussion", "reverb", "distorted", "clean",
    "happy", "sad", "calm", "energetic", "dark", "bright", "lofi", "classical",
    "hiphop", "dance", "indie", "soul", "blues", "metal", "trance", "house",
    "orchestra", "beat", "riff", "pad",
]


def write_tone_wav(path: Path, sr: int = 22050, seconds: float = 10.0, freq: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # Mix a few harmonics for richer chroma
    y = 0.4 * np.sin(2 * np.pi * freq * t)
    y += 0.2 * np.sin(2 * np.pi * freq * 1.5 * t)
    y += 0.1 * np.sin(2 * np.pi * (freq / 2) * t)
    y = (y * 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = Path(cfg["paths"]["raw"])
    splits = Path(cfg["paths"]["splits"])
    n = args.n
    sr = cfg["data"]["sample_rate"]
    rng = np.random.default_rng(cfg["seed"])

    # --- Dev FMA subset ---
    fma_audio = raw / "synthetic" / "fma_small"
    fma_recs = {"subset": "small", "genres": GENRES_SMALL, "train": [], "val": [], "test": []}
    for i in range(n):
        tid = 100000 + i
        genre_id = int(i % len(GENRES_SMALL))
        genre = GENRES_SMALL[genre_id]
        wav = fma_audio / f"{tid:06d}.wav"
        write_tone_wav(wav, sr=sr, seconds=10.0, freq=220 + 30 * genre_id + (i % 5) * 10)
        text = f"untitled track by artist {i % 7} album demo tags chill beat"
        rec = {
            "track_id": tid,
            "genre_id": genre_id,
            "genre": genre,
            "text": text,
            "audio_path": str(wav),
        }
        if i < int(0.7 * n):
            fma_recs["train"].append(rec)
        elif i < int(0.85 * n):
            fma_recs["val"].append(rec)
        else:
            fma_recs["test"].append(rec)
    save_json(fma_recs, splits / "fma_small.json")

    # Medium split pointer for config switch testing
    med = dict(fma_recs)
    med["subset"] = "medium"
    save_json(med, splits / "fma_medium.json")

    # --- Dev MusicCaps subset ---
    mc_audio = raw / "synthetic" / "musiccaps" / "audio"
    mc_csv_dir = raw / "musiccaps"
    mc_csv_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    records = []
    for i in range(n):
        ytid = f"syn{i:05d}"
        aspects = list(rng.choice(ASPECT_VOCAB, size=5, replace=False))
        caption = f"A {aspects[0]} and {aspects[1]} piece with {aspects[2]} textures, {aspects[3]} mood and {aspects[4]} tone."
        wav = mc_audio / f"{ytid}.wav"
        write_tone_wav(wav, sr=sr, seconds=10.0, freq=300 + i * 7)
        label = [1.0 if a in aspects else 0.0 for a in ASPECT_VOCAB]
        is_eval = i >= int(0.85 * n)
        rows.append(
            {
                "ytid": ytid,
                "start_s": 0,
                "end_s": 10,
                "caption": caption,
                "aspect_list": str(aspects),
                "is_audioset_eval": is_eval,
            }
        )
        records.append(
            {
                "ytid": ytid,
                "caption": caption,
                "aspects": aspects,
                "label": label,
                "audio_path": str(wav),
                "downloaded": True,
                "is_audioset_eval": is_eval,
                "text": caption,
            }
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(mc_csv_dir / "musiccaps.csv", index=False)
    # Also copy audio path expected by loader under raw/musiccaps/audio for downloaded=True checks
    # prepare_musiccaps will recompute; write splits directly too
    train = [r for r in records if not r["is_audioset_eval"]]
    n_val = max(1, int(0.1 * len(train)))
    mc_splits = {
        "vocab": ASPECT_VOCAB,
        "train": train[n_val:],
        "val": train[:n_val],
        "test": [r for r in records if r["is_audioset_eval"]],
        "n_downloaded": n,
        "n_total": n,
        "dev_subset": True,
    }
    save_json(mc_splits, splits / "musiccaps.json")
    save_json({"vocab": ASPECT_VOCAB}, splits / "musiccaps_aspect_vocab.json")

    # --- Dev DEAM subset ---
    deam_dir = raw / "synthetic" / "deam"
    audio = deam_dir / "audio"
    anns = []
    for i in range(n):
        sid = 2000 + i
        write_tone_wav(audio / f"{sid}.wav", sr=sr, seconds=8.0, freq=250 + i * 5)
        anns.append({"song_id": sid, "valence_mean": float(rng.uniform(1, 9)), "arousal_mean": float(rng.uniform(1, 9))})
    pd.DataFrame(anns).to_csv(deam_dir / "static_annotations.csv", index=False)
    from src.data_deam import prepare_deam_splits

    prepare_deam_splits(raw, splits)

    status = {
        "dev_subset": True,
        "n": n,
        "fma_small_split": str(splits / "fma_small.json"),
        "musiccaps_split": str(splits / "musiccaps.json"),
        "deam_split": str(splits / "deam.json"),
        "note": "Local development subset. Prefer real FMA/MusicCaps/DEAM downloads for final experiments.",
    }
    save_json(status, raw / "STATUS.md".replace(".md", ".json") if False else raw / "STATUS.json")
    (raw / "STATUS.md").write_text(
        "# Data status\n\n"
        f"- Development subset created with n={n}.\n"
        "- Real downloads: run `scripts/download_fma.py`, `scripts/download_musiccaps.py`, `scripts/download_deam.py`.\n"
        "- Then re-run `scripts/prepare_splits.py`.\n",
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
