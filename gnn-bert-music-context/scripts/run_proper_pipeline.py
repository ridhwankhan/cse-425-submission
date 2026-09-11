"""Full-quality real-data pipeline (longer downloads + training)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    env = os.environ.copy()
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget.exists():
        for p in winget.rglob("ffmpeg.exe"):
            env["PATH"] = str(p.parent) + os.pathsep + env.get("PATH", "")
            break
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def sync_submission() -> None:
    src = ROOT
    dst = ROOT / "submitted" / "gnn-bert-music-context"
    dst.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("config.yaml", "config.yaml"),
        ("src", "src"),
        ("scripts", "scripts"),
        ("data/splits", "data/splits"),
        ("data/processed/examples", "data/processed/examples"),
        ("results", "results"),
        ("report", "report"),
        ("README.md", "README.md"),
        ("DELIVERABLES.md", "DELIVERABLES.md"),
        ("requirements.txt", "requirements.txt"),
    ]
    for a, b in pairs:
        s, d = src / a, dst / b
        if not s.exists():
            continue
        if s.is_dir():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    print(f"Synced submission package -> {dst}")


def main():
    py = sys.executable
    # 1) Expand real corpora
    run([py, "scripts/download_fma_hf.py", "--limit", "2000"])
    run([py, "scripts/download_deam.py"])
    # MusicCaps: keep pulling until ~800 successful clips (YouTube attrition is high)
    run([py, "scripts/download_musiccaps.py", "--limit", "800"])

    # 2) Splits + graphs
    run([py, "scripts/prepare_splits.py"])
    run([py, "scripts/build_graphs.py", "--dataset", "fma", "--limit", "2000", "--workers", "2"])
    run([py, "scripts/build_graphs.py", "--dataset", "musiccaps", "--limit", "0", "--workers", "2"])
    run([py, "scripts/build_graphs.py", "--dataset", "deam", "--limit", "800", "--workers", "2"])
    run([py, "scripts/prepare_splits.py"])
    run([py, "scripts/build_context_pairs.py"])
    run([py, "scripts/export_example_graphs.py", "--n", "20"])

    # 3) Train / eval / report
    run([py, "-m", "src.train", "--task", "all"])
    run([py, "-m", "src.evaluate", "--task", "all"])
    run([py, "scripts/extra_analysis.py"])
    run([py, "scripts/build_neurips_pdf.py"])
    sync_submission()
    print("\nProper pipeline complete. See results/metrics.json and submitted/gnn-bert-music-context/")


if __name__ == "__main__":
    main()
