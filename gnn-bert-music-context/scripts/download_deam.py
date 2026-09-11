"""Download DEAM annotations + audio from the official CVML mirror (resume-safe)."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, ensure_dirs

ANN_URL = "https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip"
AUDIO_URL = "https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip"

DEAM_INFO = """
DEAM (Database for Emotional Analysis of Music)
Official: https://cvml.unige.ch/databases/DEAM/

Expected layout:
  data/raw/deam/annotations/.../static_annotations_averaged_songs_*.csv
  data/raw/deam/audio/*.mp3
"""


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"skip existing {dest} ({dest.stat().st_size} bytes)")
        return
    print(f"downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"saved {dest} ({dest.stat().st_size} bytes)")


def _extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _flatten_audio(raw_deam: Path) -> Path:
    """Move extracted mp3/wav files into data/raw/deam/audio/."""
    audio_dir = raw_deam / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    existing = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
    if len(existing) > 100:
        print(f"audio already present: {len(existing)} files in {audio_dir}")
        return audio_dir

    candidates = []
    for pat in ("**/*.mp3", "**/*.wav"):
        candidates.extend(p for p in raw_deam.glob(pat) if p.parent != audio_dir)
    moved = 0
    for src in candidates:
        dst = audio_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            moved += 1
    print(f"flattened {moved} audio files into {audio_dir} (total={(len(list(audio_dir.glob('*.mp3'))+list(audio_dir.glob('*.wav'))))})")
    return audio_dir


def _merge_static_annotations(raw_deam: Path) -> Path | None:
    """Create a single static_annotations.csv the rest of the pipeline can read."""
    song_level = (
        raw_deam
        / "annotations"
        / "annotations averaged per song"
        / "song_level"
    )
    parts = sorted(song_level.glob("static_annotations_averaged_songs_*.csv"))
    if not parts:
        # older / alternate layout
        alt = raw_deam / "annotations" / "annotations averaged per song.csv"
        if alt.exists():
            return alt
        return None

    import pandas as pd

    frames = [pd.read_csv(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    out = raw_deam / "static_annotations.csv"
    # also keep expected legacy filename if useful
    legacy_dir = raw_deam / "annotations"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy = legacy_dir / "annotations averaged per song.csv"
    df.to_csv(out, index=False)
    df.to_csv(legacy, index=False)
    print(f"merged static annotations: {len(df)} rows -> {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = Path(cfg["paths"]["raw"]) / "deam"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "README_DOWNLOAD.txt").write_text(DEAM_INFO, encoding="utf-8")

    zips = raw / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    ann_zip = zips / "DEAM_Annotations.zip"
    audio_zip = zips / "DEAM_audio.zip"

    if not args.skip_download:
        _download(ANN_URL, ann_zip)
        if not args.annotations_only:
            _download(AUDIO_URL, audio_zip)

    if ann_zip.exists():
        _extract(ann_zip, raw)
    if not args.annotations_only and audio_zip.exists():
        _extract(audio_zip, raw)
        _flatten_audio(raw)

    merged = _merge_static_annotations(raw)
    audio_n = len(list((raw / "audio").glob("*.mp3"))) + len(list((raw / "audio").glob("*.wav")))
    if merged is not None:
        print(f"DEAM ready: annotations={merged}, audio_files={audio_n}")
    else:
        print(DEAM_INFO)
        print("DEAM annotations still missing after download/extract.")


if __name__ == "__main__":
    main()
