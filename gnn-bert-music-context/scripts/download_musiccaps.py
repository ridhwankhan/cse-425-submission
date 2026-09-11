"""Download MusicCaps metadata (HF) and audio clips (yt-dlp), resume-safe."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, ensure_dirs, save_json


def _resolve_bin(name: str) -> str:
    import shutil

    found = shutil.which(name)
    if found:
        return found
    # Common WinGet FFmpeg location after install (PATH may lag)
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget.exists():
        for p in winget.rglob(f"{name}.exe"):
            return str(p)
    return name


def download_clip(ytid: str, start_s: int, end_s: int, outfile: Path) -> bool:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.exists() and outfile.stat().st_size > 1000:
        return True
    tmp = outfile.with_suffix(".tmp.%(ext)s")
    url = f"https://www.youtube.com/watch?v={ytid}"
    ytdlp = [sys.executable, "-m", "yt_dlp"]
    ffmpeg = _resolve_bin("ffmpeg")
    cmd = [
        *ytdlp,
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "--download-sections",
        f"*{start_s}-{end_s}",
        "-o",
        str(tmp),
        "--force-overwrites",
        "--quiet",
        "--no-warnings",
        url,
    ]
    if ffmpeg and ffmpeg != "ffmpeg":
        cmd[1:1] = []  # keep as is; pass ffmpeg location via env below
    env = os.environ.copy()
    ff_dir = str(Path(ffmpeg).parent) if Path(ffmpeg).exists() else ""
    if ff_dir:
        env["PATH"] = ff_dir + os.pathsep + env.get("PATH", "")
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180, env=env)
    except Exception:
        return False
    # Find produced temp file
    produced = list(outfile.parent.glob(outfile.stem + ".tmp.*"))
    if not produced:
        return False
    src = produced[0]
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-ar", "22050", "-ac", "1", str(outfile)],
            check=True,
            capture_output=True,
            timeout=90,
            env=env,
        )
        for p in produced:
            p.unlink(missing_ok=True)
        return outfile.exists() and outfile.stat().st_size > 1000
    except Exception:
        try:
            src.rename(outfile)
            return outfile.exists()
        except Exception:
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = Path(cfg["paths"]["raw"]) / "musiccaps"
    raw.mkdir(parents=True, exist_ok=True)
    audio_dir = raw / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw / "musiccaps.csv"
    try:
        from datasets import load_dataset
        import pandas as pd

        ds = load_dataset(cfg["data"]["musiccaps_hf"], split="train")
        df = ds.to_pandas()
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path} ({len(df)} rows)")
    except Exception as e:
        print(f"HF download failed: {e}")
        if not csv_path.exists():
            print("No MusicCaps CSV available. Use make_dev_subset.py for local debugging.")
            return
        import pandas as pd

        df = pd.read_csv(csv_path)

    if args.metadata_only:
        return

    # Prefer filling missing clips across the full CSV (many early YTIDs are dead).
    target = int(args.limit) if args.limit and args.limit > 0 else len(df)
    already = [p for p in audio_dir.glob("*.wav") if p.stat().st_size > 1000]
    print(f"MusicCaps target={target}, already_ok={len(already)}, csv_rows={len(df)}")

    ok = len(already)
    fail = 0
    attempted = 0
    for _, row in df.iterrows():
        if ok >= target:
            break
        ytid = str(row.get("ytid", row.get("youtube_id", "")))
        if not ytid or ytid == "nan":
            continue
        start_s = int(row.get("start_s", 0))
        end_s = int(row.get("end_s", start_s + 10))
        out = audio_dir / f"{ytid}.wav"
        if out.exists() and out.stat().st_size > 1000:
            continue
        attempted += 1
        success = download_clip(ytid, start_s, end_s, out)
        if success:
            ok += 1
        else:
            fail += 1
        if (attempted) % 25 == 0:
            print(f"progress: ok={ok} fail={fail} attempted={attempted}", flush=True)

    status = {
        "ok": ok,
        "fail": fail,
        "attempted_new": attempted,
        "target": target,
        "success_rate": ok / max(1, ok + fail),
    }
    save_json(status, raw / "download_status.json")
    print(status)


if __name__ == "__main__":
    main()
