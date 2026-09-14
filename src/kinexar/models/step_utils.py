import torch


def safe_step(delta, max_step=0.5):
    norm = delta.norm(dim=-1, keepdim=True) + 1e-6
    scale = torch.clamp(max_step / norm, max=1.0)
    return delta * scale
