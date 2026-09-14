import torch
from pathlib import Path
import sys

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from kinexar.dataset.preprocessing.graph_builder import prebuild

def debug_shard(shard_name):
    shard_path = ROOT / "data" / "processed" / "old_shards" / shard_name
    print(f"[*] Debugging {shard_name} at {shard_path}...")
    
    samples = torch.load(shard_path, map_location="cpu")
    print(f"[*] Loaded {len(samples)} samples.")
    
    for i, s in enumerate(samples):
        if i % 100 == 0:
            print(f"[*] Sample {i}...")
        feat = s['feat']
        mask = s['history_mask']
        
        # This is where it might hang or crash
        try:
            graph = prebuild(feat[:10], mask[:10], topk=9, lag=1, span=1)
        except Exception as e:
            print(f"[!] Error on sample {i}: {e}")
            return
            
    print(f"[+] Shard {shard_name} processed fine in debug mode.")

if __name__ == "__main__":
    debug_shard("shard_00053.pt")
