import torch

from kinexar.models.trajectory_types import ModelContext


def flatten_context(h_nodes, history_mask):
    batch_size = h_nodes.shape[0]
    return h_nodes.view(batch_size, -1, h_nodes.shape[-1]), history_mask.view(batch_size, -1)


def masked_mean(x, mask, dims):
    if isinstance(dims, int):
        dims = (dims,)
    weights = mask.to(x.dtype).unsqueeze(-1)
    numerator = (x * weights).sum(dim=dims)
    denominator = weights.sum(dim=dims).clamp_min_(1.0)
    return numerator / denominator


def root_last_index(history_mask):
    if history_mask.ndim != 3:
        raise ValueError("bad history mask")
    root_mask = history_mask[:, 0]
    if not root_mask.any(dim=1).all():
        raise ValueError("missing root history")
    steps = history_mask.shape[-1]
    step_index = torch.arange(steps, device=history_mask.device, dtype=torch.long).view(1, steps)
    return torch.where(root_mask, step_index, -1).amax(dim=1)


def root_anchor(feat, history_mask):
    last_idx = root_last_index(history_mask)
    batch_index = torch.arange(feat.shape[0], device=feat.device)
    return feat[batch_index, 0, last_idx, :2].unsqueeze(1)


def encode_context(encoder, feat, graph):
    history_mask = graph["history_mask"]
    # root_anchor picks the last valid ego position used as the local frame origin.
    anchor = root_anchor(feat, history_mask)
    # encoder converts the history graph into per-node hidden features.
    h_nodes = encoder(feat, graph)
    # flatten_context converts node features into a single sequence-like context.
    h, ctx_mask = flatten_context(h_nodes, history_mask)
    return ModelContext(
        history_mask=history_mask,
        anchor=anchor,
        h_nodes=h_nodes,
        h=h,
        ctx_mask=ctx_mask,
        # masked_mean ignores padded history entries while pooling.
        h_global=masked_mean(h_nodes, history_mask, dims=(1, 2)),
        ctx_mean=masked_mean(h, ctx_mask, dims=1),
    )
