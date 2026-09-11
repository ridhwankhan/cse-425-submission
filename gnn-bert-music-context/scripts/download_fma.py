"""Download FMA metadata and audio subsets."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, ensure_dirs

SHA1 = {
    "fma_metadata.zip": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
    "fma_small.zip": "ade154f733639d52e35e32f5593efe5be76c6d70",
    "fma_medium.zip": "c67b69ea232021025fca9231fc1c7c1a063ab50b",
}


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Exists: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)


def extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
    except Exception as e:
        print(f"zipfile failed ({e}); trying 7z...")
        subprocess.check_call(["7z", "x", str(zip_path), f"-o{out_dir}", "-y"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["small", "medium", "metadata"], default="small")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = Path(cfg["paths"]["raw"])

    meta_zip = raw / "fma_metadata.zip"
    download(cfg["data"]["fma_metadata_url"], meta_zip)
    if SHA1["fma_metadata.zip"]:
        digest = sha1_file(meta_zip)
        print(f"sha1 metadata: {digest}")
    if not (raw / "fma_metadata").exists():
        extract(meta_zip, raw)

    if args.subset == "metadata" or args.skip_audio:
        print("Metadata only done.")
        return

    key = f"fma_{args.subset}.zip"
    url = cfg["data"][f"fma_{args.subset}_url"]
    zpath = raw / key
    download(url, zpath)
    expected = SHA1.get(key)
    if expected and zpath.exists():
        digest = sha1_file(zpath)
        print(f"sha1 {key}: {digest} (expected {expected})")
    target = raw / f"fma_{args.subset}"
    if not target.exists():
        extract(zpath, raw)
    print("FMA download complete.")


if __name__ == "__main__":
    main()
