from __future__ import annotations

from typing import Any

import torch

from kinexar.dataset.preprocessing.graph_builder import collate_prebuilt


def _row_feat(row: dict[str, Any]) -> torch.Tensor:
    return row["feat"] if "feat" in row else row["input"]


def _row_future(row: dict[str, Any]) -> torch.Tensor:
    return row["future"] if "future" in row else row["output"]


def collate_dense_batch(rows: list[dict[str, Any]], device: torch.device | None = None) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty batch")
    feats = [_row_feat(row) for row in rows]
    futures = [_row_future(row) for row in rows]
    masks = [row["mask"].to(torch.bool) for row in rows]
    bsz = len(rows)
    agents = max(feat.shape[0] for feat in feats)
    hist_steps = feats[0].shape[1]
    dim = feats[0].shape[2]
    fut_steps = futures[0].shape[0]
    feat = feats[0].new_zeros((bsz, agents, hist_steps, dim))
    history_mask = torch.zeros((bsz, agents, hist_steps), dtype=torch.bool, device=feat.device)
    future = futures[0].new_zeros((bsz, fut_steps, futures[0].shape[-1]))

    sample_ids = []
    scenario_ids = []

    for idx, row in enumerate(rows):
        cur_feat = feats[idx]
        cur_future = futures[idx]
        cur_mask = masks[idx]
        count = cur_feat.shape[0]
        feat[idx, :count] = cur_feat
        history_mask[idx, :count] = cur_mask
        future[idx] = cur_future

        sample_ids.append(str(row.get("sample_id", "")))
        scenario_ids.append(str(row.get("scenario_id", "")))
    agent_mask = history_mask.any(dim=-1)
    out = {
        "feat": feat,
        "future": future,
        "history_mask": history_mask,
        "agent_mask": agent_mask,
        "rows": rows,
        "sample_ids": sample_ids,
        "scenario_ids": scenario_ids,
    }
    if all("graph" in row for row in rows):
        out["graph"] = collate_prebuilt([row["graph"] for row in rows], history_mask)
    if device is None:
        return out
    return move_batch_to_device(out, device)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def move(value):
        if isinstance(value, torch.Tensor):
            out = value.to(device, non_blocking=True)
            if device.type == "cpu" and out.is_floating_point() and out.dtype in (torch.float16, torch.bfloat16):
                return out.to(torch.float32)
            return out
        if isinstance(value, dict):
            return {key: move(val) for key, val in value.items()}
        if isinstance(value, list):
            return [move(val) for val in value]
        if isinstance(value, tuple):
            return tuple(move(val) for val in value)
        return value

    return {key: move(value) for key, value in batch.items()}


def compute_scale_from_past(past_abs: torch.Tensor) -> torch.Tensor:
    # Forced to 1.0 because data is already normalized in shards
    return torch.ones((past_abs.shape[0], 1, 1), device=past_abs.device, dtype=past_abs.dtype)


def _masked_first_last(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.ndim != 2:
        raise ValueError("bad mask")
    steps = mask.shape[1]
    idx = torch.arange(steps, device=mask.device, dtype=torch.long).view(1, steps)
    first = torch.where(mask, idx, steps).amin(dim=1)
    last = torch.where(mask, idx, -1).amax(dim=1)
    return first, last


def root_anchor_from_past_batch(feat: torch.Tensor, history_mask: torch.Tensor | None = None) -> torch.Tensor:
    # Forced to 0.0 because data is already anchored in shards
    return torch.zeros((feat.shape[0], 1, 2), device=feat.device, dtype=feat.dtype)


def compute_scale_from_past_masked(past_abs: torch.Tensor, past_mask: torch.Tensor) -> torch.Tensor:
    # Forced to 1.0 because data is already normalized in shards
    return torch.ones((past_abs.shape[0], 1, 1), device=past_abs.device, dtype=past_abs.dtype)


def scale_from_past_batch(feat: torch.Tensor, history_mask: torch.Tensor | None = None) -> torch.Tensor:
    if feat.ndim != 4:
        raise ValueError("bad feat")
    if history_mask is not None:
        if history_mask.ndim != 3 or history_mask.shape != feat.shape[:3]:
            raise ValueError("bad history mask")
        return compute_scale_from_past_masked(feat[:, 0, :, :2], history_mask[:, 0])
    return compute_scale_from_past(feat[:, 0, :, :2])


def scale_from_observed_batch(feat: torch.Tensor, history_mask: torch.Tensor | None = None) -> torch.Tensor:
    return scale_from_past_batch(feat, history_mask)


def future_local_from_past_scale(
    feat: torch.Tensor,
    future: torch.Tensor,
    history_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feat.ndim != 4 or future.ndim != 3:
        raise ValueError("bad feat/future")
    anchor = root_anchor_from_past_batch(feat, history_mask)
    scale = scale_from_past_batch(feat, history_mask)
    future_local = (future - anchor) / scale
    return future_local, scale
