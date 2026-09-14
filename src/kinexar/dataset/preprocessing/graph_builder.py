import torch

DEFAULT_AGENT_RADIUS_METERS = 50.0


def _as_batch(inp, msk):
    squeeze = False
    if inp.ndim == 3:
        inp = inp.unsqueeze(0)
        squeeze = True
    if msk.ndim == 2:
        msk = msk.unsqueeze(0)
        squeeze = True
    if inp.ndim != 4:
        raise ValueError("bad input")
    if msk.ndim != 3:
        raise ValueError("bad mask")
    if inp.shape[:3] != msk.shape:
        raise ValueError("shape mismatch")
    return inp, msk.to(torch.bool), squeeze


def _node_id(batch_id, agent_id, time_id, agents, steps):
    return batch_id * (agents * steps) + agent_id * steps + time_id


def _empty(dev):
    return {
        "src_node": torch.empty((0,), dtype=torch.long, device=dev),
        "dst_node": torch.empty((0,), dtype=torch.long, device=dev),
    }


def _pack(src_node, dst_node, dev):
    if not src_node:
        return _empty(dev)
    return {
        "src_node": torch.tensor(src_node, dtype=torch.long, device=dev),
        "dst_node": torch.tensor(dst_node, dtype=torch.long, device=dev),
    }


def _topk_edges(pos, msk, src_t, dst_t, rad, topk):
    dev = pos.device
    bsz, agents, steps, _ = pos.shape
    rr = float(rad) * float(rad)
    
    # Fully Vectorized GPU Graph Builder
    # src_t, dst_t are indices into the time dimension
    src_t = torch.as_tensor(list(src_t), device=dev, dtype=torch.long)
    dst_t = torch.as_tensor(list(dst_t), device=dev, dtype=torch.long)
    num_pairs = len(src_t)
    
    # pos: [B, A, T, 2]
    # Extract all required time slices: [B, A, P, 2]
    src_pos = pos[:, :, src_t]
    dst_pos = pos[:, :, dst_t]
    
    # Calculate all-to-all distances for each time pair: [B, P, A, A]
    # (B, A, P, 2) -> (B, P, A, 1, 2)
    src_pos = src_pos.permute(0, 2, 1, 3).unsqueeze(3)
    # (B, A, P, 2) -> (B, P, 1, A, 2)
    dst_pos = dst_pos.permute(0, 2, 1, 3).unsqueeze(2)
    
    diff = src_pos - dst_pos
    dist = (diff * diff).sum(dim=-1) # [B, P, A, A]
    
    # Handle masks: [B, A, T] -> [B, P, A, A]
    src_msk = msk[:, :, src_t].permute(0, 2, 1).unsqueeze(3) # [B, P, A, 1]
    dst_msk = msk[:, :, dst_t].permute(0, 2, 1).unsqueeze(2) # [B, P, 1, A]
    valid = src_msk & dst_msk
    
    # Self-interaction avoidance if same timestep
    if (src_t == dst_t).all():
        eye = torch.eye(agents, device=dev, dtype=torch.bool).view(1, 1, agents, agents)
        valid &= ~eye
        
    dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
    
    # Top-K across all agents for each (Batch, Pair, Agent)
    vals, idx = torch.topk(dist, min(topk, agents), dim=-1, largest=False) # [B, P, A, K]
    
    keep = torch.isfinite(vals) & (vals <= rr)
    
    # Sink Node Strategy: Invalid edges point to index N (node_count)
    sink_idx = bsz * agents * steps
    
    # Generate node IDs
    # node_id = b * (A * S) + a * S + t
    batch_idx = torch.arange(bsz, device=dev).view(bsz, 1, 1, 1)
    agent_idx = torch.arange(agents, device=dev).view(1, 1, agents, 1)
    time_src = src_t.view(1, num_pairs, 1, 1)
    time_dst = dst_t.view(1, num_pairs, 1, 1)
    
    src_nodes = _node_id(batch_idx, agent_idx, time_src, agents, steps).expand_as(idx)
    dst_nodes = _node_id(batch_idx, idx, time_dst, agents, steps)
    
    # Zero-sync masking
    src_nodes = torch.where(keep, src_nodes, torch.full_like(src_nodes, sink_idx))
    dst_nodes = torch.where(keep, dst_nodes, torch.full_like(dst_nodes, sink_idx))
    
    return {
        "src_node": src_nodes.view(-1),
        "dst_node": dst_nodes.view(-1),
    }


def temporal(msk):
    if msk.ndim != 3:
        raise ValueError("bad mask")
    dev = msk.device
    bsz, agents, steps = msk.shape
    sink_idx = bsz * agents * steps
    
    batch_idx = torch.arange(bsz, device=dev).view(bsz, 1, 1)
    agent_idx = torch.arange(agents, device=dev).view(1, agents, 1)
    time_idx = torch.arange(steps - 1, device=dev).view(1, 1, steps - 1)
    
    src_valid = msk[:, :, :-1]
    dst_valid = msk[:, :, 1:]
    keep = src_valid & dst_valid
    
    src_nodes = _node_id(batch_idx, agent_idx, time_idx, agents, steps)
    dst_nodes = _node_id(batch_idx, agent_idx, time_idx + 1, agents, steps)
    
    src_nodes = torch.where(keep, src_nodes, torch.full_like(src_nodes, sink_idx))
    dst_nodes = torch.where(keep, dst_nodes, torch.full_like(dst_nodes, sink_idx))
    
    return {
        "src_node": src_nodes.view(-1),
        "dst_node": dst_nodes.view(-1),
    }


def spatial(pos, msk, rad=30.0, topk=8):
    steps = pos.shape[2]
    return _topk_edges(pos, msk, range(steps), range(steps), rad, topk)


def cross(pos, msk, lag=1, span=None, rad=30.0, topk=8):
    steps = pos.shape[2]
    if span is None:
        span = steps - 1
    src_t = []
    dst_t = []
    for ts in range(steps):
        end = min(steps, ts + span + 1)
        for td in range(ts + lag, end):
            src_t.append(ts)
            dst_t.append(td)
    return _topk_edges(pos, msk, src_t, dst_t, rad, topk)


def build(inp, msk, rad=30.0, topk=8, lag=1, span=1):
    inp, msk, _ = _as_batch(inp, msk)
    pos = inp[..., :2]
    agent_mask = msk.any(dim=-1)
    return {
        "history_mask": msk,
        "agent_mask": agent_mask,
        "temporal": temporal(msk),
        "spatial": spatial(pos, msk, rad=rad, topk=topk),
        "cross": cross(pos, msk, lag=lag, span=span, rad=rad, topk=topk),
    }


def _edge_to_coords(edge, agents, steps):
    sink = int(agents) * int(steps)
    src = edge["src_node"].detach().cpu().to(torch.long)
    dst = edge["dst_node"].detach().cpu().to(torch.long)
    keep = (src < sink) & (dst < sink)
    src = src[keep]
    dst = dst[keep]
    return {
        "src_agent": (src // steps).to(torch.int16),
        "src_time": (src % steps).to(torch.int16),
        "dst_agent": (dst // steps).to(torch.int16),
        "dst_time": (dst % steps).to(torch.int16),
    }


def prebuild(inp, msk, rad=DEFAULT_AGENT_RADIUS_METERS, topk=9, lag=1, span=1):
    """Build a sample-local graph that can be stored in preprocessed rows.

    Flat graph node IDs depend on the eventual padded batch shape, so this
    function stores only valid endpoint coordinates. Collation later converts
    them back into batch-local flat node IDs.
    """
    batched, mask, squeeze = _as_batch(inp, msk)
    if not squeeze or batched.shape[0] != 1:
        raise ValueError("prebuild expects one sample")
    graph = build(batched, mask, rad=rad, topk=topk, lag=lag, span=span)
    agents = int(batched.shape[1])
    steps = int(batched.shape[2])
    return {
        "format": "agent_time_edges_v1",
        "agents": agents,
        "steps": steps,
        "rad": float(rad),
        "topk": int(topk),
        "lag": int(lag),
        "span": int(span),
        "temporal": _edge_to_coords(graph["temporal"], agents, steps),
        "spatial": _edge_to_coords(graph["spatial"], agents, steps),
        "cross": _edge_to_coords(graph["cross"], agents, steps),
    }


def _coords_to_edge(coords, batch_id, agents, steps, device):
    if coords is None:
        return _empty(device)
    src_agent = coords["src_agent"].to(device=device, dtype=torch.long)
    src_time = coords["src_time"].to(device=device, dtype=torch.long)
    dst_agent = coords["dst_agent"].to(device=device, dtype=torch.long)
    dst_time = coords["dst_time"].to(device=device, dtype=torch.long)
    src_node = _node_id(batch_id, src_agent, src_time, agents, steps)
    dst_node = _node_id(batch_id, dst_agent, dst_time, agents, steps)
    return {"src_node": src_node, "dst_node": dst_node}


def collate_prebuilt(graphs, history_mask, device=None):
    if history_mask.ndim != 3:
        raise ValueError("bad history mask")
    if len(graphs) != int(history_mask.shape[0]):
        raise ValueError("graph batch mismatch")
    dev = device if device is not None else history_mask.device
    bsz, agents, steps = history_mask.shape
    out = {
        "history_mask": history_mask.to(device=dev, dtype=torch.bool),
        "agent_mask": history_mask.to(device=dev, dtype=torch.bool).any(dim=-1),
    }
    for key in ("temporal", "spatial", "cross"):
        parts = [_coords_to_edge(graph[key], i, agents, steps, dev) for i, graph in enumerate(graphs)]
        if parts:
            out[key] = {
                "src_node": torch.cat([part["src_node"] for part in parts], dim=0),
                "dst_node": torch.cat([part["dst_node"] for part in parts], dim=0),
            }
        else:
            out[key] = _empty(dev)
    return out
