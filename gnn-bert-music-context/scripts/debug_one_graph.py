from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, load_json

# inline process to avoid import path issues
from scripts.build_graphs import process_one

cfg = load_config()
sp = load_json(Path(cfg["paths"]["splits"]) / "fma_small.json")
r = sp["train"][0]
audio = Path(r["audio_path"])
print("audio", audio, audio.exists())
stem = Path(cfg["paths"]["processed"]) / "fma" / "small" / f"{int(r['track_id']):06d}"
print("stem", stem)
print("result", process_one(audio, stem, cfg))
seg = Path(str(stem) + "_segment.pt")
print("exists", seg.exists(), seg)
