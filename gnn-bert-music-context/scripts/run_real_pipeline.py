"""End-to-end real-data pipeline for CSE425 submission quality run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    env = os.environ.copy()
    # Ensure WinGet ffmpeg is visible
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget.exists():
        for p in winget.rglob("ffmpeg.exe"):
            env["PATH"] = str(p.parent) + os.pathsep + env.get("PATH", "")
            break
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def main():
    py = sys.executable
    # 1) Downloads
    run([py, "scripts/download_fma.py", "--subset", "small"])
    run([py, "scripts/download_musiccaps.py", "--metadata-only"])
    # Attempt a capped MusicCaps audio pull (YouTube; resume-safe)
    run([py, "scripts/download_musiccaps.py", "--limit", "400"])
    run([py, "scripts/download_deam.py"])

    # 2) Splits from real metadata
    run([py, "scripts/prepare_splits.py"])

    # 3) Graphs on real FMA (capped) + whatever MusicCaps audio succeeded
    run([py, "scripts/build_graphs.py", "--dataset", "fma", "--limit", "1200"])
    run([py, "scripts/build_graphs.py", "--dataset", "musiccaps", "--limit", "600"])

    # 4) Rebuild context pairs now that graphs exist
    run([py, "scripts/prepare_splits.py"])
    run([py, "scripts/export_example_graphs.py", "--n", "20"])

    # 5) Train + evaluate
    run([py, "-m", "src.train", "--task", "all"])
    run([py, "-m", "src.evaluate", "--task", "all"])
    run([py, "scripts/extra_analysis.py"])
    print("\nPipeline complete. See results/metrics.json")


if __name__ == "__main__":
    main()
