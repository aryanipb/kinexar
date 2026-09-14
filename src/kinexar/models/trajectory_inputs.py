from kinexar.models.trajectory_types import NormalizedBatch


def normalize_inputs(feat, history_mask, agent_mask, scale, future):
    squeeze = feat.ndim == 3
    if squeeze:
        feat = feat.unsqueeze(0)
    if feat.ndim != 4:
        raise ValueError("bad feat")

    if future is not None:
        if future.ndim == 2:
            future = future.unsqueeze(0)
        if future.ndim != 3:
            raise ValueError("bad future")

    if scale.ndim == 2:
        scale = scale.unsqueeze(1)

    if scale.ndim != 3:
        raise ValueError("bad scale")

    batch_size = feat.shape[0]
    if scale.shape[0] != batch_size:
        raise ValueError("batch mismatch")
    if future is not None and future.shape[0] != batch_size:
        raise ValueError("future batch mismatch")

    if history_mask.ndim == 2:
        history_mask = history_mask.unsqueeze(0)
    if agent_mask is not None and agent_mask.ndim == 1:
        agent_mask = agent_mask.unsqueeze(0)

    if history_mask.shape != feat.shape[:3]:
        raise ValueError("bad history mask")

    return NormalizedBatch(
        feat=feat,
        graph={
            "history_mask": history_mask,
            "agent_mask": agent_mask,
        },
        scale=scale,
        future=future,
        squeeze=squeeze,
    )
