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
LOG_PATH = OUT_ROOT / "logs" / "reprocess.log"

MAX_AGENTS = 10  # Focal + 9 neighbors
STOP_FLAG = False

def handle_sigint(sig, frame):
    global STOP_FLAG
    print("\n[!] Received Ctrl+C. Completing current shard and stopping...")
    STOP_FLAG = True

signal.signal(signal.SIGINT, handle_sigint)

def setup_dirs():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SHARDS_OUT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS shard_progress (shard_name TEXT PRIMARY KEY, status TEXT, updated_at TEXT)")
    conn.commit()
    return conn

def select_nearest_indices(feat, mask):
    """
    feat: [A, T, 5]
    mask: [A, T]
    Returns indices for focal agent + top 9 neighbors.
    """
    n_agents = feat.shape[0]
    if n_agents <= MAX_AGENTS:
        return torch.arange(n_agents)
    
    # Focal is index 0
    # For others, calculate distance to focal at last valid timestep
    focal_pos = feat[0, -1, :2]
    
    distances = []
    for i in range(1, n_agents):
        # Find last valid pos
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

def process_shard(shard_path, conn):
    shard_name = shard_path.name
    print(f"[*] Processing {shard_name}...")
    
    try:
        samples = torch.load(shard_path, map_location="cpu")
    except Exception as e:
        print(f"[!] Error loading {shard_name}: {e}")
        return

    n_samples = len(samples)
    
    # Extract dimensions from first sample
    s0 = samples[0]
    hist_steps = s0['feat'].shape[1]
    future_steps = s0['future'].shape[0]
    
    # New format tensors
    inp_batch = torch.zeros((n_samples, MAX_AGENTS, hist_steps, 5), dtype=torch.float16)
    out_batch = torch.zeros((n_samples, future_steps, 2), dtype=torch.float16)
    msk_batch = torch.zeros((n_samples, MAX_AGENTS, hist_steps), dtype=torch.bool)
    agent_counts = torch.zeros((n_samples,), dtype=torch.long)
    sample_ids = []
    scenario_ids = []
    start_ts = []
    graphs = []

    for i, s in enumerate(samples):
        feat = s['feat']
        mask = s['history_mask']
        future = s['future']
        sc_id = s['scenario_id']
        
        # Cap agents
        keep = select_nearest_indices(feat, mask)
        c_feat = feat[keep].to(torch.float16)
        c_mask = mask[keep].to(torch.bool)
        n_a = len(keep)
        
        # Fill tensors
        inp_batch[i, :n_a] = c_feat
        out_batch[i] = future.to(torch.float16)
        msk_batch[i, :n_a] = c_mask
        agent_counts[i] = n_a
        
        # Metadata
        # Use scenario_id as sample_id if missing
        s_id = f"{sc_id}:0" 
        sample_ids.append(s_id)
        scenario_ids.append(sc_id)
        start_ts.append(0)
        
        # Build Graph
        graph = prebuild(c_feat, c_mask, topk=9, lag=1, span=1)
        graphs.append(graph)

    payload = {
        "input": inp_batch,
        "output": out_batch,
        "mask": msk_batch,
        "graphs": graphs,
        "history_steps": hist_steps,
        "future_steps": future_steps,
        "agent_count": agent_counts,
        "sample_ids": sample_ids,
        "scenario_ids": scenario_ids,
        "start_ts": start_ts,
    }
    
    out_path = SHARDS_OUT / shard_name
    torch.save(payload, out_path)
    
    # Mark as done
    conn.execute("INSERT OR REPLACE INTO shard_progress (shard_name, status, updated_at) VALUES (?, 'done', ?)", 
                 (shard_name, time.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    print(f"[+] Finished {shard_name}")

def main():
    setup_dirs()
    conn = connect_db()
    
    # Get list of shards recursively
    all_shards = sorted(list(OLD_SHARDS_ROOT.rglob("shard_*.pt")))
    if not all_shards:
        print(f"[!] No shards starting with 'shard_' found in {OLD_SHARDS_ROOT}.")
        return

    print(f"[*] Found {len(all_shards)} shards to process.")
    
    for shard_path in all_shards:
        if STOP_FLAG:
            break
            
        # Get relative path for uniqueness and output structure
        rel_path = shard_path.relative_to(OLD_SHARDS_ROOT)
        shard_key = str(rel_path)
        
        # Check if already done
        row = conn.execute("SELECT status FROM shard_progress WHERE shard_name = ?", (shard_key,)).fetchone()
        if row and row[0] == 'done':
            continue
            
        # Ensure output subdirectory exists
        out_shard_path = SHARDS_OUT / rel_path
        out_shard_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process and save
        process_shard_v2(shard_path, out_shard_path, shard_key, conn)

def process_shard_v2(shard_path, out_path, shard_key, conn):
    print(f"[*] Processing {shard_key}...")
    
    try:
        samples = torch.load(shard_path, map_location="cpu")
    except Exception as e:
        print(f"[!] Error loading {shard_key}: {e}")
        return

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

    payload = {
        "input": inp_batch, "output": out_batch, "mask": msk_batch, "graphs": graphs,
        "history_steps": hist_steps, "future_steps": future_steps, "agent_count": agent_counts,
        "sample_ids": sample_ids, "scenario_ids": scenario_ids, "start_ts": start_ts,
    }
    
    torch.save(payload, out_path)
    conn.execute("INSERT OR REPLACE INTO shard_progress (shard_name, status, updated_at) VALUES (?, 'done', ?)", 
                 (shard_key, time.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    print(f"[+] Finished {shard_key}")

if __name__ == "__main__":
    main()
