from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch
from torch.optim import AdamW

from kinexar.training.config.defaults import ExperimentConfig
from kinexar.training.config.defaults import config_to_dict
from kinexar.training.config.defaults import resolve_ablation_config
from kinexar.training.config.defaults import save_config
from kinexar.training.config.defaults import trajectory_model_kwargs
from kinexar.training.evaluate import run_evaluation
from kinexar.training.subset_builder import build_subset_indices
from kinexar.training.subset_builder import subset_metadata
from kinexar.training.train import _autocast
from kinexar.training.train import _configure_cuda_memory_budget
from kinexar.training.train import _enforce_vram_cap
from kinexar.training.train import _maybe_compile_model
from kinexar.training.train import _save_checkpoint
from kinexar.training.train import _train_epoch
from kinexar.training.train import _validate_gradients
from kinexar.training.train import _validate_model_outputs
from kinexar.training.utils import choose_device
from kinexar.training.utils import has_rows_bundle_split
from kinexar.training.utils import ensure_dir
from kinexar.training.utils import load_processed_dataset
from kinexar.training.utils import load_rows_bundle_dataset
from kinexar.training.utils import make_loader
from kinexar.training.utils import make_subset
from kinexar.training.utils import preload_rows
from kinexar.training.utils import seed_all
from kinexar.training.utils import setup_logging
from kinexar.training.utils import write_json
from kinexar.models.model import TrajectoryModel
from kinexar.training.batching import collate_dense_batch
from kinexar.training.batching import future_local_from_past_scale
from kinexar.training.batching import scale_from_past_batch
from kinexar.training.losses import moe


ALLOWED_COMPILE_MODES = ("off", "reduce-overhead", "max-autotune")
VALIDATION_PROCESSED_ROOT_ENV = "KINEXAR_VAL_PROCESSED_ROOT"


def normalize_compile_mode(raw: str | None) -> str:
    value = str(raw or "off").strip().lower()
    if value not in ALLOWED_COMPILE_MODES:
        raise ValueError(f"compile_mode must be one of {ALLOWED_COMPILE_MODES}, got {raw!r}")
    return value


def git_commit_hash() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def collect_env_info(config: ExperimentConfig) -> dict[str, Any]:
    gpu_name = None
    gpu_memory_gb = None
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        gpu_name = props.name
        gpu_memory_gb = round(float(props.total_memory) / (1024**3), 3)
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "amp_enabled": bool(config.train.use_amp),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "compile_mode": str(config.train.compile_mode),
        "compile_enabled": bool(config.train.use_compile),
        "git_commit_hash": git_commit_hash(),
        "resolved_config": config_to_dict(config),
    }


def save_runtime_bundle(run_dir: Path, config: ExperimentConfig) -> dict[str, Any]:
    run_dir = ensure_dir(run_dir)
    save_config(config, run_dir / "resolved_config.json")
    env = collect_env_info(config)
    write_json(run_dir / "env.json", env)
    return env


def has_validation_rows_bundle(config: ExperimentConfig) -> bool:
    bundle_root = Path(os.environ.get(VALIDATION_PROCESSED_ROOT_ENV, config.processed_root)).expanduser()
    return has_rows_bundle_split(bundle_root, "val")


def prepare_train_val_datasets(config: ExperimentConfig) -> tuple[Any, Any | None]:
    train_root = Path(config.processed_root).expanduser()
    validation_root = Path(os.environ.get(VALIDATION_PROCESSED_ROOT_ENV, str(train_root))).expanduser()
    if has_rows_bundle_split(train_root, "train"):
        train_dataset = load_rows_bundle_dataset(train_root, split="train")
        if has_rows_bundle_split(validation_root, "val"):
            val_dataset = load_rows_bundle_dataset(validation_root, split="val")
        else:
            val_dataset = None
        return train_dataset, val_dataset
    dataset = load_processed_dataset(str(train_root))
    subsets = build_subset_indices(dataset, config.subset)
    train_rows = preload_rows(dataset, subset_metadata(dataset, subsets.train, "train"))
    val_rows = preload_rows(dataset, subset_metadata(dataset, subsets.val, "val"))
    return make_subset(train_rows), make_subset(val_rows)


def build_model_and_optimizer(config: ExperimentConfig, train_dataset: Any, device: torch.device, logger: Any) -> tuple[TrajectoryModel, AdamW, torch.amp.GradScaler]:
    probe = train_dataset[0]
    model = TrajectoryModel(
        probe["input"].shape[-1],
        future_steps=probe["output"].shape[0],
        **trajectory_model_kwargs(config),
    ).to(device)
    model = _maybe_compile_model(model, config, logger)
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=(config.train.use_amp and device.type == "cuda"))
    return model, optimizer, scaler


def save_named_checkpoint(
    path: Path,
    *,
    model: TrajectoryModel,
    optimizer: AdamW,
    config: ExperimentConfig,
    epoch: int,
    global_step: int,
    best_val_min6ade: float | None = None,
    best_val_min6fde: float | None = None,
    best_train_mean_min6ade: float | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_min6ade": best_val_min6ade,
        "best_val_min6fde": best_val_min6fde,
        "best_train_mean_min6ade": best_train_mean_min6ade,
        "config": config_to_dict(config),
        "future_steps": int(model.cvae_dec.steps),
    }
    _save_checkpoint(payload, path)


def load_resume_checkpoint(path: Path, model: TrajectoryModel, optimizer: AdamW) -> dict[str, Any]:
    payload = torch.load(path, map_location=next(model.parameters()).device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def run_one_batch_smoke(
    *,
    config: ExperimentConfig,
    train_dataset: Any,
    run_dir: Path,
    logger: Any,
    validate_split: str | None,
) -> dict[str, Any]:
    device = choose_device(config.train.device)
    model, optimizer, scaler = build_model_and_optimizer(config, train_dataset, device, logger)
    train_loader = make_loader(train_dataset, batch_size=1, shuffle=False, num_workers=0, seed=config.seed)
    batch = next(iter(train_loader))
    optimizer.zero_grad(set_to_none=True)
    dense = collate_dense_batch(batch, device=device)
    feat = dense["feat"]
    future = dense["future"]
    scale = scale_from_past_batch(feat, dense["history_mask"])
    future_local_norm, _ = future_local_from_past_scale(feat, future, dense["history_mask"])
    with _autocast(device, config.train.use_amp):
        out = model(
            feat,
            dense["history_mask"],
            dense["agent_mask"],
            scale,
            graph=dense.get("graph"),
            future=future,
            mode="train",
        )
        _validate_model_outputs(out, future)
        loss_pack = moe(
            out["y_final_local"],
            future_local_norm,
            expert_aux=out["expert_aux"],
            post_mu=out["post_mu"],
            post_logvar=out["post_logvar"],
            prior_mu=out["prior_mu"],
            prior_logvar=out["prior_logvar"],
            free_bits=config.ablation.free_bits,
            warmup_steps=config.ablation.warmup_steps,
            use_physics_loss=config.ablation.use_physics_losses,
            lambda_ownership=config.ablation.lambda_ownership,
            lambda_leakage=config.ablation.lambda_leakage,
            lambda_stability=config.ablation.lambda_stability,
            use_cfcp_ownership=config.ablation.use_cfcp_ownership,
            use_cfcp_leakage=config.ablation.use_cfcp_leakage,
            use_cfcp_stability=config.ablation.use_cfcp_stability,
            compute_cri_metrics=config.ablation.compute_cri_metrics,
        )
    if not torch.isfinite(loss_pack["loss"]).all():
        pass
    elif scaler.is_enabled():
        scaler.scale(loss_pack["loss"]).backward()
        scaler.unscale_(optimizer)
    else:
        loss_pack["loss"].backward()
    _validate_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip_norm)
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    smoke_ckpt = run_dir / "smoke.ckpt"
    save_named_checkpoint(
        smoke_ckpt,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=0,
        global_step=1,
        best_val_min6ade=None,
        best_val_min6fde=None,
    )
    if validate_split is None:
        return {
            "checkpoint": str(smoke_ckpt),
            "validation_skipped": {
                "reason": "full training profile has no validation split; all rows are used for training",
                "split": "val",
            },
        }
    smoke_metrics = run_evaluation(config, checkpoint_path=smoke_ckpt, split=validate_split)
    metrics_path = Path(config.run_dir) / f"evaluation_{validate_split}" / config.evaluation.val_metrics_filename
    if not metrics_path.exists():
        raise RuntimeError(f"smoke.validation: missing validation metrics file at {metrics_path}")
    return {
        "checkpoint": str(smoke_ckpt),
        "validation": smoke_metrics["aggregate"],
    }


def configure_runtime(config: ExperimentConfig, logger: Any | None = None) -> torch.device:
    config.ablation = resolve_ablation_config(config)
    compile_mode = normalize_compile_mode(config.train.compile_mode)
    config.train.compile_mode = compile_mode
    config.train.use_compile = compile_mode != "off"
    device = choose_device(config.train.device)
    _configure_cuda_memory_budget(device, config.train.vram_target_gb, config.train.vram_cap_gb)
    if logger is not None:
        logger.info("runtime compile_mode=%s use_compile=%s", config.train.compile_mode, config.train.use_compile)
    return device


def maybe_log_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"gpu_allocated_gb": 0.0, "gpu_reserved_gb": 0.0, "gpu_peak_gb": 0.0}
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "gpu_allocated_gb": float(torch.cuda.memory_allocated(idx) / (1024**3)),
        "gpu_reserved_gb": float(torch.cuda.memory_reserved(idx) / (1024**3)),
        "gpu_peak_gb": float(torch.cuda.max_memory_allocated(idx) / (1024**3)),
    }


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
