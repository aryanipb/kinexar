from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class SampleMetrics:
    loss: float
    recon: float
    kl: float
    kl_weight: float
    vel_loss: float
    expert1_loss: float
    expert2_loss: float
    expert3_loss: float
    expert4_loss: float
    expert5_loss: float
    expert6_loss: float
    expert1_physics: float
    expert2_physics: float
    expert3_physics: float
    expert4_physics: float
    expert5_physics: float
    expert6_physics: float
    ade: float
    fde: float
    raw_ade: float
    raw_fde: float
    min_ade: float
    min_fde: float
    raw_min_ade: float
    raw_min_fde: float
    miss: float
    selected_expert: int
    endpoint_ade: float
    length_ratio: float
    pred_avg_velocity: float
    gt_avg_velocity: float
    coord_scale: float
    per_timestep_ade: list[float]
    avg_min_fde: float
    avg_min_ade: float
    actor_miss_rate: float
    avg_brier_min_fde: float
    avg_min6_fde: float
    avg_min6_ade: float
    selected_probability: float
    sample_id: str
    scenario_id: str


def compute_errors(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    err = torch.linalg.vector_norm(pred - gt, dim=-1)
    return err.mean(dim=-1), err[..., -1]


def _mode_probabilities(out: dict[str, torch.Tensor]) -> torch.Tensor:
    probs = out.get("probs")
    if isinstance(probs, torch.Tensor):
        return probs
    scores = out.get("scores")
    if isinstance(scores, torch.Tensor):
        return torch.softmax(scores, dim=-1)
    modes = int(out["y_all"].shape[1])
    return out["y_all"].new_full((out["y_all"].shape[0], modes), 1.0 / max(modes, 1))


def collect_batch_sample_metrics(
    rows: list[dict[str, Any]],
    out: dict[str, torch.Tensor],
    loss_pack: dict[str, torch.Tensor],
    future: torch.Tensor,
) -> list[SampleMetrics]:
    samples = []
    # SHARD DATA IS ALREADY NORMALIZED (Divided by 50)
    # INTERNAL SCALE IS NOW FORCED TO 1.0
    # To get REAL METERS, we must multiply by the base scale (50.0)
    BASE_SCALE = 50.0
    
    # out["pred_abs"] is currently normalized (matching future in shards)
    ade_all, fde_all = compute_errors(out["pred_abs"] * BASE_SCALE, future[:, None] * BASE_SCALE)
    raw_ade_all, raw_fde_all = ade_all, fde_all
    best_fde_idx = fde_all.argmin(dim=-1)
    best_ade_idx = ade_all.argmin(dim=-1)
    batch_idx = torch.arange(out["pred_abs"].shape[0], device=out["pred_abs"].device)
    pred_selected = out["pred_abs"][batch_idx, best_fde_idx] * BASE_SCALE
    probs = _mode_probabilities(out)
    selected_prob = probs[batch_idx, best_fde_idx]
    
    future_meters = future * BASE_SCALE
    ade_selected, fde_selected = compute_errors(pred_selected, future_meters)
    per_timestep = torch.linalg.vector_norm(pred_selected - future_meters, dim=-1)
    pred_disp = torch.linalg.vector_norm(pred_selected[:, -1] - pred_selected[:, 0], dim=-1)
    gt_disp = torch.linalg.vector_norm(future_meters[:, -1] - future_meters[:, 0], dim=-1)
    pred_vel = pred_selected[:, 1:] - pred_selected[:, :-1]
    gt_vel = future_meters[:, 1:] - future_meters[:, :-1]
    pred_avg_velocity = torch.linalg.vector_norm(pred_vel, dim=-1).mean(dim=-1)
    gt_avg_velocity = torch.linalg.vector_norm(gt_vel, dim=-1).mean(dim=-1)
    ade_mean = ade_all.mean(dim=-1)
    fde_mean = fde_all.mean(dim=-1)
    min_ade = ade_all.min(dim=-1).values
    min_fde = fde_all.min(dim=-1).values
    raw_min_ade = raw_ade_all.min(dim=-1).values
    raw_min_fde = raw_fde_all.min(dim=-1).values
    # Miss rate is 2.0 meters
    miss = (min_fde > 2.0).to(torch.float32)
    brier_min_fde = fde_selected + (1.0 - selected_prob).pow(2)
    for idx, row in enumerate(rows):
        samples.append(
            SampleMetrics(
                loss=float(loss_pack["loss"].detach().cpu()),
                recon=float(loss_pack["recon"].detach().cpu()),
                kl=float(loss_pack["kl"].detach().cpu()),
                kl_weight=float(loss_pack["kl_weight"].detach().cpu()),
                vel_loss=float(loss_pack["vel_loss"].detach().cpu()),
                expert1_loss=float(loss_pack["expert1_loss"].detach().cpu()),
                expert2_loss=float(loss_pack["expert2_loss"].detach().cpu()),
                expert3_loss=float(loss_pack["expert3_loss"].detach().cpu()),
                expert4_loss=float(loss_pack["expert4_loss"].detach().cpu()),
                expert5_loss=float(loss_pack["expert5_loss"].detach().cpu()),
                expert6_loss=float(loss_pack["expert6_loss"].detach().cpu()),
                expert1_physics=float(loss_pack["expert1_physics"].detach().cpu()),
                expert2_physics=float(loss_pack["expert2_physics"].detach().cpu()),
                expert3_physics=float(loss_pack["expert3_physics"].detach().cpu()),
                expert4_physics=float(loss_pack["expert4_physics"].detach().cpu()),
                expert5_physics=float(loss_pack["expert5_physics"].detach().cpu()),
                expert6_physics=float(loss_pack["expert6_physics"].detach().cpu()),
                ade=float(ade_mean[idx].detach().cpu()),
                fde=float(fde_mean[idx].detach().cpu()),
                raw_ade=float(raw_ade_all.mean(dim=-1)[idx].detach().cpu()),
                raw_fde=float(raw_fde_all.mean(dim=-1)[idx].detach().cpu()),
                min_ade=float(min_ade[idx].detach().cpu()),
                min_fde=float(min_fde[idx].detach().cpu()),
                raw_min_ade=float(raw_min_ade[idx].detach().cpu()),
                raw_min_fde=float(raw_min_fde[idx].detach().cpu()),
                miss=float(miss[idx].detach().cpu()),
                selected_expert=int(best_fde_idx[idx].detach().cpu()),
                endpoint_ade=float(fde_selected[idx].detach().cpu()),
                length_ratio=float((pred_disp[idx] / gt_disp[idx].clamp_min(1e-6)).detach().cpu()),
                pred_avg_velocity=float(pred_avg_velocity[idx].detach().cpu()),
                gt_avg_velocity=float(gt_avg_velocity[idx].detach().cpu()),
                coord_scale=float(out["scale"][idx].view(-1)[0].detach().cpu()),
                per_timestep_ade=[float(x) for x in per_timestep[idx].detach().cpu().tolist()],
                avg_min_fde=float(fde_selected[idx].detach().cpu()),
                avg_min_ade=float(ade_selected[idx].detach().cpu()),
                actor_miss_rate=float((fde_selected[idx] > 2.0).to(torch.float32).detach().cpu()),
                avg_brier_min_fde=float(brier_min_fde[idx].detach().cpu()),
                avg_min6_fde=float(min_fde[idx].detach().cpu()),
                avg_min6_ade=float(ade_all[idx, best_ade_idx[idx]].detach().cpu()),
                selected_probability=float(selected_prob[idx].detach().cpu()),
                sample_id=str(row.get("sample_id", "")),
                scenario_id=str(row.get("scenario_id", "")),
            )
        )
    return samples


LOSS_METRIC_KEYS = (
    "loss",
    "recon",
    "kl",
    "kl_weight",
    "vel_loss",
    "expert1_loss",
    "expert2_loss",
    "expert3_loss",
    "expert4_loss",
    "expert5_loss",
    "expert6_loss",
    "expert1_physics",
    "expert2_physics",
    "expert3_physics",
    "expert4_physics",
    "expert5_physics",
    "expert6_physics",
    "mode_entropy",
    "cri",
    "loss_ownership",
    "loss_leakage",
    "loss_stability",
    "delta_route_route",
    "delta_route_speed",
    "delta_route_interaction",
    "delta_route_style",
    "delta_speed_route",
    "delta_speed_speed",
    "delta_speed_interaction",
    "delta_speed_style",
    "delta_interaction_route",
    "delta_interaction_speed",
    "delta_interaction_interaction",
    "delta_interaction_style",
    "delta_style_route",
    "delta_style_speed",
    "delta_style_interaction",
    "delta_style_style",
)


def init_epoch_metric_sums(
    *,
    expert_count: int,
    future_steps: int,
    device=torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {
        "count": torch.zeros((), device=device, dtype=torch.float64),
        "expert_usage_histogram": torch.zeros((expert_count,), device=device, dtype=torch.float64),
        "per_timestep_ade": torch.zeros((future_steps,), device=device, dtype=torch.float64),
    }
    scalar_keys = list(LOSS_METRIC_KEYS) + [
        "ade",
        "fde",
        "raw_ade",
        "raw_fde",
        "min_ade",
        "min_fde",
        "raw_min_ade",
        "raw_min_fde",
        "miss_rate",
        "avg_min_fde",
        "avg_min_ade",
        "actor_miss_rate",
        "avg_brier_min_fde",
        "avg_min6_fde",
        "avg_min6_ade",
        "selected_probability",
        "endpoint_ade",
        "length_ratio",
        "pred_avg_velocity",
        "gt_avg_velocity",
        "coord_scale",
    ]
    for key in scalar_keys:
        sums[key] = torch.zeros((), device=device, dtype=torch.float64)
    return sums


def accumulate_batch_metric_sums(
    sums: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    loss_pack: dict[str, torch.Tensor],
    future: torch.Tensor,
) -> None:
    # SHARD DATA IS ALREADY NORMALIZED (Divided by 50)
    # INTERNAL SCALE IS NOW FORCED TO 1.0
    # To get REAL METERS, we must multiply by the base scale (50.0)
    BASE_SCALE = 50.0
    
    # pred_abs is currently normalized (matching future in shards)
    ade_all, fde_all = compute_errors(out["pred_abs"] * BASE_SCALE, future[:, None] * BASE_SCALE)
    raw_ade_all, raw_fde_all = ade_all, fde_all
    best_fde_idx = fde_all.argmin(dim=-1)
    best_ade_idx = ade_all.argmin(dim=-1)
    batch_idx = torch.arange(out["pred_abs"].shape[0], device=out["pred_abs"].device)
    pred_selected = out["pred_abs"][batch_idx, best_fde_idx] * BASE_SCALE
    probs = _mode_probabilities(out)
    selected_prob = probs[batch_idx, best_fde_idx]
    
    future_meters = future * BASE_SCALE
    ade_selected, fde_selected = compute_errors(pred_selected, future_meters)
    per_timestep = torch.linalg.vector_norm(pred_selected - future_meters, dim=-1)
    pred_disp = torch.linalg.vector_norm(pred_selected[:, -1] - pred_selected[:, 0], dim=-1)
    gt_disp = torch.linalg.vector_norm(future_meters[:, -1] - future_meters[:, 0], dim=-1)
    pred_vel = pred_selected[:, 1:] - pred_selected[:, :-1]
    gt_vel = future_meters[:, 1:] - future_meters[:, :-1]
    pred_avg_velocity = torch.linalg.vector_norm(pred_vel, dim=-1).mean(dim=-1)
    gt_avg_velocity = torch.linalg.vector_norm(gt_vel, dim=-1).mean(dim=-1)
    ade_mean = ade_all.mean(dim=-1)
    fde_mean = fde_all.mean(dim=-1)
    min_ade = ade_all.min(dim=-1).values
    min_fde = fde_all.min(dim=-1).values
    raw_ade = raw_ade_all.mean(dim=-1)
    raw_fde = raw_fde_all.mean(dim=-1)
    raw_min_ade = raw_ade_all.min(dim=-1).values
    raw_min_fde = raw_fde_all.min(dim=-1).values
    # Miss rate is 2.0 meters
    miss = (min_fde > 2.0).to(torch.float32)
    actor_miss = (fde_selected > 2.0).to(torch.float32)
    brier_min_fde = fde_selected + (1.0 - selected_prob).pow(2)
    min6_ade = ade_all[batch_idx, best_ade_idx]
    batch_count = future.shape[0]
    weight = torch.tensor(float(batch_count), device=sums["count"].device, dtype=torch.float64)

    sums["count"].add_(weight)
    sums["expert_usage_histogram"].add_(torch.bincount(best_fde_idx, minlength=sums["expert_usage_histogram"].numel()).to(device=sums["count"].device, dtype=torch.float64))
    sums["per_timestep_ade"].add_(per_timestep.sum(dim=0).to(device=sums["count"].device, dtype=torch.float64))

    for key in LOSS_METRIC_KEYS:
        if key in loss_pack:
            sums[key].add_(loss_pack[key].to(device=sums["count"].device, dtype=torch.float64) * weight)

    scalar_updates = {
        "ade": ade_mean.sum(),
        "fde": fde_mean.sum(),
        "raw_ade": raw_ade.sum(),
        "raw_fde": raw_fde.sum(),
        "min_ade": min_ade.sum(),
        "min_fde": min_fde.sum(),
        "raw_min_ade": raw_min_ade.sum(),
        "raw_min_fde": raw_min_fde.sum(),
        "miss_rate": miss.sum(),
        "avg_min_fde": fde_selected.sum(),
        "avg_min_ade": ade_selected.sum(),
        "actor_miss_rate": actor_miss.sum(),
        "avg_brier_min_fde": brier_min_fde.sum(),
        "avg_min6_fde": min_fde.sum(),
        "avg_min6_ade": min6_ade.sum(),
        "selected_probability": selected_prob.sum(),
        "endpoint_ade": fde_selected.sum(),
        "length_ratio": (pred_disp / gt_disp.clamp_min(1e-6)).sum(),
        "pred_avg_velocity": pred_avg_velocity.sum(),
        "gt_avg_velocity": gt_avg_velocity.sum(),
        "coord_scale": out["scale"].view(out["scale"].shape[0], -1)[:, 0].sum(),
    }
    for key, value in scalar_updates.items():
        sums[key].add_(value.to(device=sums["count"].device, dtype=torch.float64))


def finalize_epoch_metric_sums(
    sums: dict[str, torch.Tensor],
    *,
    nonfinite_batches: int = 0,
) -> dict[str, Any]:
    count = int(sums["count"].item())
    if count <= 0:
        raise ValueError("no samples to aggregate")
    per_timestep_tensor = (sums["per_timestep_ade"] / sums["count"]).tolist()
    steps = len(per_timestep_tensor)
    cut1 = max(steps // 3, 1)
    cut2 = max((2 * steps) // 3, cut1 + 1)
    early = sum(per_timestep_tensor[:cut1]) / len(per_timestep_tensor[:cut1])
    mid = sum(per_timestep_tensor[cut1:cut2]) / max(len(per_timestep_tensor[cut1:cut2]), 1)
    late = sum(per_timestep_tensor[cut2:]) / max(len(per_timestep_tensor[cut2:]), 1)

    out: dict[str, Any] = {"count": count, "per_timestep_ade": [float(x) for x in per_timestep_tensor]}
    for key in LOSS_METRIC_KEYS:
        out[key] = float((sums[key] / sums["count"]).item())
    for key in (
        "ade",
        "fde",
        "raw_ade",
        "raw_fde",
        "min_ade",
        "min_fde",
        "raw_min_ade",
        "raw_min_fde",
        "miss_rate",
        "avg_min_fde",
        "avg_min_ade",
        "actor_miss_rate",
        "avg_brier_min_fde",
        "avg_min6_fde",
        "avg_min6_ade",
        "selected_probability",
        "endpoint_ade",
        "length_ratio",
        "pred_avg_velocity",
        "gt_avg_velocity",
        "coord_scale",
    ):
        out[key] = float((sums[key] / sums["count"]).item())
    usage = [int(x) for x in sums["expert_usage_histogram"].tolist()]
    out["early_ade"] = float(early)
    out["mid_ade"] = float(mid)
    out["late_ade"] = float(late)
    out["expert_usage_histogram"] = usage
    out["expert_usage_ratio_max"] = max(usage) / max(count, 1)
    out["nonfinite_batches"] = int(nonfinite_batches)
    out["avgMinFDE"] = out["avg_min_fde"]
    out["avgMinADE"] = out["avg_min_ade"]
    out["actorMR"] = out["actor_miss_rate"]
    out["avgBrierMinFDE"] = out["avg_brier_min_fde"]
    out["avgMin6FDE"] = out["avg_min6_fde"]
    out["avgMin6ADE"] = out["avg_min6_ade"]
    out["delta_matrix_4x4"] = {
        "route": {
            "route": out.get("delta_route_route", 0.0),
            "speed": out.get("delta_route_speed", 0.0),
            "interaction": out.get("delta_route_interaction", 0.0),
            "style": out.get("delta_route_style", 0.0),
        },
        "speed": {
            "route": out.get("delta_speed_route", 0.0),
            "speed": out.get("delta_speed_speed", 0.0),
            "interaction": out.get("delta_speed_interaction", 0.0),
            "style": out.get("delta_speed_style", 0.0),
        },
        "interaction": {
            "route": out.get("delta_interaction_route", 0.0),
            "speed": out.get("delta_interaction_speed", 0.0),
            "interaction": out.get("delta_interaction_interaction", 0.0),
            "style": out.get("delta_interaction_style", 0.0),
        },
        "style": {
            "route": out.get("delta_style_route", 0.0),
            "speed": out.get("delta_style_speed", 0.0),
            "interaction": out.get("delta_style_interaction", 0.0),
            "style": out.get("delta_style_style", 0.0),
        },
    }
    return out


def aggregate_epoch_metrics(
    samples: list[SampleMetrics],
    expert_count: int,
    nonfinite_batches: int = 0,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("no samples to aggregate")
    usage = [0 for _ in range(expert_count)]
    for sample in samples:
        usage[sample.selected_expert] += 1
    count = len(samples)
    steps = len(samples[0].per_timestep_ade)
    per_timestep = [
        sum(sample.per_timestep_ade[idx] for sample in samples) / count
        for idx in range(steps)
    ]
    cut1 = max(steps // 3, 1)
    cut2 = max((2 * steps) // 3, cut1 + 1)
    early = sum(per_timestep[:cut1]) / len(per_timestep[:cut1])
    mid = sum(per_timestep[cut1:cut2]) / max(len(per_timestep[cut1:cut2]), 1)
    late = sum(per_timestep[cut2:]) / max(len(per_timestep[cut2:]), 1)
    return {
        "count": count,
        "loss": sum(x.loss for x in samples) / count,
        "recon": sum(x.recon for x in samples) / count,
        "kl": sum(x.kl for x in samples) / count,
        "kl_weight": sum(x.kl_weight for x in samples) / count,
        "vel_loss": sum(x.vel_loss for x in samples) / count,
        "expert1_loss": sum(x.expert1_loss for x in samples) / count,
        "expert2_loss": sum(x.expert2_loss for x in samples) / count,
        "expert3_loss": sum(x.expert3_loss for x in samples) / count,
        "expert4_loss": sum(x.expert4_loss for x in samples) / count,
        "expert5_loss": sum(x.expert5_loss for x in samples) / count,
        "expert6_loss": sum(x.expert6_loss for x in samples) / count,
        "expert1_physics": sum(x.expert1_physics for x in samples) / count,
        "expert2_physics": sum(x.expert2_physics for x in samples) / count,
        "expert3_physics": sum(x.expert3_physics for x in samples) / count,
        "expert4_physics": sum(x.expert4_physics for x in samples) / count,
        "expert5_physics": sum(x.expert5_physics for x in samples) / count,
        "expert6_physics": sum(x.expert6_physics for x in samples) / count,
        "ade": sum(x.ade for x in samples) / count,
        "fde": sum(x.fde for x in samples) / count,
        "raw_ade": sum(x.raw_ade for x in samples) / count,
        "raw_fde": sum(x.raw_fde for x in samples) / count,
        "min_ade": sum(x.min_ade for x in samples) / count,
        "min_fde": sum(x.min_fde for x in samples) / count,
        "raw_min_ade": sum(x.raw_min_ade for x in samples) / count,
        "raw_min_fde": sum(x.raw_min_fde for x in samples) / count,
        "miss_rate": sum(x.miss for x in samples) / count,
        "avg_min_fde": sum(x.avg_min_fde for x in samples) / count,
        "avg_min_ade": sum(x.avg_min_ade for x in samples) / count,
        "actor_miss_rate": sum(x.actor_miss_rate for x in samples) / count,
        "avg_brier_min_fde": sum(x.avg_brier_min_fde for x in samples) / count,
        "avg_min6_fde": sum(x.avg_min6_fde for x in samples) / count,
        "avg_min6_ade": sum(x.avg_min6_ade for x in samples) / count,
        "selected_probability": sum(x.selected_probability for x in samples) / count,
        "endpoint_ade": sum(x.endpoint_ade for x in samples) / count,
        "length_ratio": sum(x.length_ratio for x in samples) / count,
        "pred_avg_velocity": sum(x.pred_avg_velocity for x in samples) / count,
        "gt_avg_velocity": sum(x.gt_avg_velocity for x in samples) / count,
        "coord_scale": sum(x.coord_scale for x in samples) / count,
        "per_timestep_ade": per_timestep,
        "early_ade": early,
        "mid_ade": mid,
        "late_ade": late,
        "expert_usage_histogram": usage,
        "expert_usage_ratio_max": max(usage) / max(count, 1),
        "nonfinite_batches": nonfinite_batches,
        "avgMinFDE": sum(x.avg_min_fde for x in samples) / count,
        "avgMinADE": sum(x.avg_min_ade for x in samples) / count,
        "actorMR": sum(x.actor_miss_rate for x in samples) / count,
        "avgBrierMinFDE": sum(x.avg_brier_min_fde for x in samples) / count,
        "avgMin6FDE": sum(x.avg_min6_fde for x in samples) / count,
        "avgMin6ADE": sum(x.avg_min6_ade for x in samples) / count,
    }
