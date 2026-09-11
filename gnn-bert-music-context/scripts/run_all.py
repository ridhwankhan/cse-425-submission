"""Run download-free local subset build, graph export, train all tasks, evaluate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    py = sys.executable
    run([py, "scripts/make_dev_subset.py", "--n", "40"])
    run([py, "scripts/build_graphs.py", "--dataset", "all"])
    run([py, "scripts/export_example_graphs.py", "--n", "20"])
    run([py, "-m", "src.train", "--task", "all"])
    run([py, "-m", "src.evaluate", "--task", "all"])
    print("\nDone. See results/metrics.json")


if __name__ == "__main__":
    main()
