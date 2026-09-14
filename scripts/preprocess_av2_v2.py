import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import _preprocess_av2_impl as base


OUT = Path(os.environ.get("RW_AV2_V2_OUT", str(ROOT / "data" / "processed" / "av2_v2")))
base.MANIFEST = Path(os.environ.get("RW_AV2_V2_MANIFEST", str(ROOT / "data" / "manifests" / "av2_manifest1.json")))
base.OUT = OUT
base.DB = OUT / "progress.sqlite3"
base.SHARDS = Path(os.environ.get("RW_AV2_V2_SHARDS", str(OUT / "shards")))
base.LOGS = OUT / "logs"
base.LOG = base.LOGS / "preprocess.log"
base.PID = OUT / "preprocess.pid"
base.DONE = OUT / "processed_manifest.jsonl"
base.LEFT = OUT / "remaining_manifest.jsonl"
base.TRASH = Path(os.environ.get("RW_AV2_V2_TRASH", str(ROOT / "data" / "_trash_cleanup_av2_v2")))


def parity_rows():
    rows = base.parse_manifest()
    if not rows:
        raise ValueError("empty v2 manifest")
    n = min(10, len(rows))
    if n == len(rows):
        return rows
    picks = []
    for i in range(n):
        idx = round(i * (len(rows) - 1) / max(n - 1, 1))
        picks.append(rows[idx])
    out = []
    seen = set()
    for row in picks:
        if row["sample_id"] in seen:
            continue
        seen.add(row["sample_id"])
        out.append(row)
    return out


def print_config():
    cfg = {
        "manifest": str(base.MANIFEST),
        "output_root": str(base.OUT),
        "history_steps": 50,
        "future_steps": 60,
        "window_steps": 110,
    }
    print(json.dumps(cfg, indent=2))


def main():
    base.parity_rows = parity_rows
    print_config()
    base.main()


if __name__ == "__main__":
    main()
