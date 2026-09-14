import os
import json
import sqlite3
import signal
import sys
import time
from pathlib import Path
import torch
import multiprocessing as mp

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from kinexar.dataset.preprocessing.graph_builder import prebuild

# Constants
OLD_SHARDS_ROOT = ROOT / "data" / "processed" / "old_shards"
OUT_ROOT = ROOT / "data" / "processed" / "av2_v2"
SHARDS_OUT = OUT_ROOT / "shards"
DB_PATH = OUT_ROOT / "progress.sqlite3"

MAX_AGENTS = 10  # Focal + 9 neighbors

def select_nearest_indices(feat, mask):
    n_agents = feat.shape[0]
    if n_agents <= MAX_AGENTS:
        return torch.arange(n_agents)
    focal_pos = feat[0, -1, :2]
    distances = []
    for i in range(1, n_agents):
        valid_ts = torch.where(mask[i])[0]
        if len(valid_ts) == 0:
            dist_val = 1e9
        else:
            last_ts = valid_ts[-1]
            dist_val = torch.norm(feat[i, last_ts, :2] - focal_pos).item()
        distances.append((dist_val, i))
    distances.sort()
    keep = [0] + [idx for _, idx in distances[:MAX_AGENTS - 1]]
    return torch.tensor(keep)

def process_shard(shard_path):
    shard_name = shard_path.name
    print(f"[*] Processing {shard_name}...")
    samples = torch.load(shard_path, map_location="cpu")
    n_samples = len(samples)
    s0 = samples[0]
    hist_steps = s0['feat'].shape[1]
    future_steps = s0['future'].shape[0]
    
    inp_batch = torch.zeros((n_samples, MAX_AGENTS, hist_steps, 5), dtype=torch.float16)
    out_batch = torch.zeros((n_samples, future_steps, 2), dtype=torch.float16)
    msk_batch = torch.zeros((n_samples, MAX_AGENTS, hist_steps), dtype=torch.bool)
    agent_counts = torch.zeros((n_samples,), dtype=torch.long)
    sample_ids, scenario_ids, start_ts, graphs = [], [], [], []

    for i, s in enumerate(samples):
        if i % 100 == 0: print(f" {i}", end="", flush=True)
        feat, mask, future, sc_id = s['feat'], s['history_mask'], s['future'], s['scenario_id']
        keep = select_nearest_indices(feat, mask)
        c_feat, c_mask = feat[keep].to(torch.float16), mask[keep].to(torch.bool)
        n_a = len(keep)
        inp_batch[i, :n_a] = c_feat
        out_batch[i] = future.to(torch.float16)
        msk_batch[i, :n_a] = c_mask
        agent_counts[i] = n_a
        sample_ids.append(f"{sc_id}:0")
        scenario_ids.append(sc_id)
        start_ts.append(0)
        graphs.append(prebuild(c_feat, c_mask, topk=9, lag=1, span=1))

    print("\n[*] Saving...")
    payload = {
        "input": inp_batch, "output": out_batch, "mask": msk_batch, "graphs": graphs,
        "history_steps": hist_steps, "future_steps": future_steps, "agent_count": agent_counts,
        "sample_ids": sample_ids, "scenario_ids": scenario_ids, "start_ts": start_ts,
    }
    torch.save(payload, SHARDS_OUT / shard_name)
    print(f"[+] Finished {shard_name}")

if __name__ == "__main__":
    process_shard(OLD_SHARDS_ROOT / "shard_00053.pt")
