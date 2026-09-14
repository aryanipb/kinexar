import sys
from pathlib import Path
import torch
from torch.optim import AdamW

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from kinexar.models.model import TrajectoryModel
from kinexar.training.batching import collate_dense_batch
from kinexar.training.config.defaults import load_config, trajectory_model_kwargs
from kinexar.training.losses import moe

def smoke_test():
    print("[*] Starting smoke test on new dataset...")
    device = torch.device("cpu")
    
    # 1. Load a sample from the NEW dataset
    shard_path = ROOT / "data" / "processed" / "av2_v2" / "shards" / "shard_00000.pt"
    if not shard_path.exists():
        print(f"[!] Shard not found at {shard_path}. Is the background job running?")
        return
        
    print(f"[*] Loading shard: {shard_path}")
    shard = torch.load(shard_path, map_location="cpu")
    
    # Create a minimal 'row' structure expected by collate_dense_batch
    # We take the first sample
    idx = 0
    row = {
        "input": shard["input"][idx],
        "output": shard["output"][idx],
        "mask": shard["mask"][idx],
        "graph": shard["graphs"][idx],
        "sample_id": shard["sample_ids"][idx],
        "scenario_id": shard["scenario_ids"][idx],
    }
    
    # 2. Collate into a batch of 1
    batch = collate_dense_batch([row], device=device)
    print(f"[*] Batch collated. Keys: {batch.keys()}")
    
    # Pre-compute scale as required by model
    from kinexar.training.batching import scale_from_past_batch
    batch["scale"] = scale_from_past_batch(batch["feat"], batch["history_mask"])
    
    # 3. Initialize Model
    config = load_config()
    # Ensure map usage is disabled in config if there's a flag
    config.ablation.use_explicit_map_tokens = False 
    
    model = TrajectoryModel(
        row["input"].shape[-1],
        future_steps=row["output"].shape[0],
        **trajectory_model_kwargs(config),
    ).to(device)
    print("[*] Model initialized.")
    
    # 4. Training Forward Pass
    print("[*] Running Training Forward Pass...")
    model.train()
    out = model(
        feat=batch["feat"],
        history_mask=batch["history_mask"],
        agent_mask=batch["agent_mask"],
        scale=batch.get("scale"), # might be None if not computed yet
        graph=batch.get("graph"),
        future=batch["future"],
        mode="train"
    )
    
    # Calculate Loss
    from kinexar.training.batching import future_local_from_past_scale
    future_local_norm, _ = future_local_from_past_scale(batch["feat"], batch["future"], batch["history_mask"])
    
    loss_pack = moe(
        out["y_final_local"],
        future_local_norm,
        expert_aux=out["expert_aux"],
        post_mu=out.get("post_mu"),
        post_logvar=out.get("post_logvar"),
        prior_mu=out.get("prior_mu"),
        prior_logvar=out.get("prior_logvar"),
        global_step=0,
        free_bits=config.ablation.free_bits,
        use_physics_loss=config.ablation.use_physics_losses,
        warmup_steps=config.ablation.warmup_steps,
        lambda_ownership=config.ablation.lambda_ownership,
        lambda_leakage=config.ablation.lambda_leakage,
        lambda_stability=config.ablation.lambda_stability,
        use_cfcp_ownership=config.ablation.use_cfcp_ownership,
        use_cfcp_leakage=config.ablation.use_cfcp_leakage,
        use_cfcp_stability=config.ablation.use_cfcp_stability,
        compute_cri_metrics=config.ablation.compute_cri_metrics,
    )
    
    loss = loss_pack["loss"]
    print(f"[+] Training Forward Pass Successful. Loss: {loss.item():.4f}")
    
    # 5. Backward Pass
    print("[*] Running Backward Pass...")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss.backward()
    optimizer.step()
    print("[+] Backward Pass Successful.")
    
    # 6. Validation Forward Pass
    print("[*] Running Validation Forward Pass...")
    model.eval()
    with torch.no_grad():
        val_out = model(
            feat=batch["feat"],
            history_mask=batch["history_mask"],
            agent_mask=batch["agent_mask"],
            scale=batch.get("scale"),
            graph=batch.get("graph"),
            mode="eval"
        )
    print(f"[+] Validation Forward Pass Successful. Output shape: {val_out['y_all'].shape}")
    
    print("\n[***] SMOKE TEST PASSED PERFECTLY [***]")

if __name__ == "__main__":
    smoke_test()
