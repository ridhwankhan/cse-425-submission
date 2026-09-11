"""Download a real FMA-small subset from Hugging Face (faster than the full ZIP mirror).

Saves audio into the standard FMA layout expected by this repo:
  data/raw/fma_small/{tid[:3]}/{tid:06d}.mp3

Also writes a lightweight tracks table if official metadata ZIP is incomplete.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, ensure_dirs, save_json


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=-1)
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1200, help="Number of real FMA tracks to cache locally")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="benjamin-paine/free-music-archive-small")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = Path(cfg["paths"]["raw"])
    out_root = raw / "fma_small"
    out_root.mkdir(parents=True, exist_ok=True)

    from datasets import Audio, load_dataset

    print(f"Loading {args.dataset} (streaming, decode=False)…")
    ds = load_dataset(args.dataset, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    records = []
    ok = 0
    for i, row in enumerate(ds):
        if ok >= args.limit:
            break
        # Track id from audio filename when present (e.g. 000002.mp3)
        tid = None
        audio = row.get("audio") or {}
        path_hint = ""
        if isinstance(audio, dict):
            path_hint = str(audio.get("path") or "")
        if path_hint:
            stem = Path(path_hint).stem
            if stem.isdigit():
                tid = int(stem)
        if tid is None:
            tid = 100000 + i

        genres_raw = row.get("genres")
        if isinstance(genres_raw, list) and genres_raw:
            genre = str(genres_raw[0])
        else:
            genre = str(row.get("genre") or row.get("label") or "Unknown")
        # Map known FMA top-level genre ids when possible
        GENRE_ID_MAP = {
            "15": "Electronic",
            "38": "Experimental",
            "17": "Folk",
            "21": "Hip-Hop",
            "1235": "Instrumental",
            "10": "Pop",
            "12": "Rock",
            "14": "Jazz",
            "3": "Blues",
            "9": "Country",
            "2": "International",
            "4": "Jazz",
            "5": "Classical",
            "8": "Easy Listening",
            "66": "Pop",
            "70": "Hip-Hop",
            "31": "Classical",
            "183": "Folk",
            "297": "Electronic",
            "359": "Rock",
            "763": "Experimental",
            "811": "Instrumental",
            "1032": "International",
        }
        genre = GENRE_ID_MAP.get(str(genre), genre if not str(genre).isdigit() else f"genre_{genre}")
        # Snap to FMA-small names when substring matches
        for g in (
            "Electronic",
            "Experimental",
            "Folk",
            "Hip-Hop",
            "Instrumental",
            "International",
            "Pop",
            "Rock",
        ):
            if g.lower() in str(genre).lower():
                genre = g
                break

        title = str(row.get("title") or f"track_{tid}")
        artist = str(row.get("artist") or "unknown artist")
        album = str(row.get("album_title") or row.get("album") or "")

        rel_dir = out_root / f"{tid:06d}"[:3]
        rel_dir.mkdir(parents=True, exist_ok=True)
        # Prefer raw bytes (mp3/wav) when available
        audio_path = None
        if isinstance(audio, dict) and audio.get("bytes"):
            raw_bytes = audio["bytes"]
            # Detect container by magic
            ext = ".mp3"
            if raw_bytes[:4] == b"RIFF":
                ext = ".wav"
            elif raw_bytes[:4] == b"fLaC":
                ext = ".flac"
            audio_path = rel_dir / f"{tid:06d}{ext}"
            if not audio_path.exists():
                audio_path.write_bytes(raw_bytes)
        elif isinstance(audio, dict) and audio.get("array") is not None:
            audio_path = rel_dir / f"{tid:06d}.wav"
            if not audio_path.exists():
                _write_wav(audio_path, np.asarray(audio["array"], dtype=np.float32), int(audio.get("sampling_rate") or 22050))
        elif isinstance(audio, dict) and audio.get("path"):
            src = Path(audio["path"])
            if src.exists():
                audio_path = rel_dir / f"{tid:06d}{src.suffix or '.mp3'}"
                if not audio_path.exists():
                    audio_path.write_bytes(src.read_bytes())
        if audio_path is None or not audio_path.exists():
            continue
        records.append(
            {
                "track_id": tid,
                "genre": genre,
                "title": title,
                "artist": artist,
                "album": album,
                "audio_path": str(audio_path),
            }
        )
        ok += 1
        if ok % 25 == 0:
            print(f"cached {ok}/{args.limit} real FMA tracks", flush=True)
            meta_path = raw / "fma_hf_subset.json"
            save_json({"n": len(records), "dataset": args.dataset, "records": records}, meta_path)

    meta_path = raw / "fma_hf_subset.json"
    save_json({"n": len(records), "dataset": args.dataset, "records": records}, meta_path)
    print(f"Done. Saved {len(records)} tracks. Manifest: {meta_path}")


if __name__ == "__main__":
    main()
