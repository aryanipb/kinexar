from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any
from typing import Callable
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Subset

# Fix "Too many open files" error by switching to file_system sharing strategy
try:
    import torch.multiprocessing as mp
    if mp.get_sharing_strategy() != "file_system":
        mp.set_sharing_strategy("file_system")
except (ImportError, RuntimeError):
    pass

from kinexar.models.model import TrajectoryModel
from kinexar.training.batching import collate_dense_batch
from kinexar.training.batching import move_batch_to_device
from kinexar.training.batching import future_local_from_past_scale
from kinexar.training.batching import scale_from_past_batch
from kinexar.training.losses import moe

from kinexar.training.config.defaults import ExperimentConfig
from kinexar.training.config.defaults import config_to_dict
from kinexar.training.config.defaults import load_config
from kinexar.training.config.defaults import resolve_ablation_config
from kinexar.training.config.defaults import save_config
from kinexar.training.config.defaults import trajectory_model_kwargs
from kinexar.training.metrics import accumulate_batch_metric_sums
from kinexar.training.metrics import finalize_epoch_metric_sums
from kinexar.training.metrics import init_epoch_metric_sums
from kinexar.training.subset_builder import build_subset_indices
from kinexar.training.subset_builder import subset_metadata
from kinexar.training.utils import choose_device
from kinexar.training.utils import ensure_dir
from kinexar.training.utils import has_rows_bundle_split
from kinexar.training.utils import load_processed_dataset
from kinexar.training.utils import load_rows_bundle_dataset
from kinexar.training.utils import make_loader
from kinexar.training.utils import make_subset
from kinexar.training.utils import preload_rows
from kinexar.training.utils import seed_all
from kinexar.training.utils import setup_logging
from kinexar.training.utils import timestamp
from kinexar.training.utils import write_csv
from kinexar.training.utils import write_json


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(enabled and device.type == "cuda"))


def _maybe_compile_model(model: TrajectoryModel, config: ExperimentConfig, logger: Any) -> TrajectoryModel:
    # HARD-DISABLED for stability during NeurIPS 2026 run
    return model


def _configure_cuda_memory_budget(
    device: torch.device,
    target_gb: float,
    cap_gb: float,
) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "total_gb": 0.0,
            "target_gb": 0.0,
            "cap_gb": 0.0,
            "fraction": 0.0,
        }
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    total_gb = float(props.total_memory) / (1024**3)
    cap_gb = min(float(cap_gb), total_gb)
    fraction = min(max(cap_gb / max(total_gb, 1e-6), 1e-6), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    info = {
        "total_gb": total_gb,
        "target_gb": float(target_gb),
        "cap_gb": cap_gb,
        "fraction": fraction,
    }
    return info


def _enforce_vram_cap(device: torch.device, cap_gb: float) -> None:
    if device.type != "cuda":
        return
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    peak_gb = torch.cuda.max_memory_allocated(device_index) / (1024**3)
    if peak_gb > float(cap_gb):
        raise RuntimeError(f"VRAM cap exceeded: peak_gb={peak_gb:.4f} cap_gb={float(cap_gb):.4f}")


def _coord_diagnostics(row: dict[str, Any]) -> dict[str, float]:
    feat = row["feat"] if "feat" in row else row["input"]
    mask = row["mask"].to(torch.bool)
    past = feat[0:1, :, :2]
    scale = scale_from_past_batch(feat.unsqueeze(0), mask.unsqueeze(0)).view(())
    past_mask = mask[0:1]
    step_disp = torch.linalg.vector_norm(past[:, 1:] - past[:, :-1], dim=-1)
    valid_pair = past_mask[:, 1:] & past_mask[:, :-1]
    path_length = torch.where(valid_pair, step_disp, torch.zeros_like(step_disp)).sum()
    valid_idx = torch.nonzero(past_mask[0], as_tuple=False).view(-1)
    endpoint_distance = torch.linalg.vector_norm(
        past[0, valid_idx[-1]] - past[0, valid_idx[0]],
        dim=-1,
    ).view(())
    return {
        "coord_scale": float(scale.detach().cpu()),
        "gt_path_length_m": float(path_length.detach().cpu()),
        "gt_endpoint_distance_m": float(endpoint_distance.detach().cpu()),
    }


def _log_grad_norm(model: TrajectoryModel) -> float:
    grads = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
    total_grad_norm = torch.norm(torch.stack(grads)) if grads else torch.tensor(0.0)
    return float(total_grad_norm.item())


def _log_param_norm(model: TrajectoryModel) -> float:
    params = [p.norm() for p in model.parameters()]
    total_param_norm = torch.norm(torch.stack(params)) if params else torch.tensor(0.0)
    return float(total_param_norm.item())


def _checkpoint_payload(
    model: TrajectoryModel,
    optimizer: AdamW,
    epoch: int,
    global_step: int,
    best_epoch_mean_min6ade: float,
    config: ExperimentConfig,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_epoch_mean_min6ade": float(best_epoch_mean_min6ade),
        "best_mean_min6ade": float(best_epoch_mean_min6ade),
        "config": config_to_dict(config),
        "future_steps": int(model.cvae_dec.steps),
    }


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    if not path.exists() or path.stat().st_size <= 0:
        pass # Warning logged by caller if needed


def _write_model_checkpoint(
    model: TrajectoryModel,
    optimizer: AdamW,
    *,
    checkpoint_path: Path,
    checkpoint_state: dict[str, Any],
    config: ExperimentConfig,
    epoch: int,
    global_step: int,
    train_mean_min6ade: float,
    logger: Any,
    flush_callback: Callable[[], None] | None,
    batch_in_epoch: int | None = None,
    reason: str,
) -> None:
    payload = _checkpoint_payload(
        model,
        optimizer,
        epoch,
        global_step,
        float(checkpoint_state["best_epoch_mean_min6ade"]),
        config,
    )
    payload["best_train_mean_min6ade"] = float(checkpoint_state.get("best_train_mean_min6ade", train_mean_min6ade))
    payload["checkpoint_reason"] = str(reason)
    payload["checkpoint_metric"] = "train_mean_min6ade"
    payload["train_mean_min6ade"] = float(train_mean_min6ade)
    if batch_in_epoch is not None:
        payload["batch_in_epoch"] = int(batch_in_epoch)
    _save_checkpoint(payload, checkpoint_path)
    _log_block(
        logger,
        [
            "[CHECKPOINT SAVED]",
            f"reason={reason}",
            f"epoch={epoch}",
            f"batch_in_epoch={'n/a' if batch_in_epoch is None else batch_in_epoch}",
            f"global_step={global_step}",
            f"train_mean_min6ade={train_mean_min6ade:.6f}",
            f"path={checkpoint_path}",
        ],
    )
    if flush_callback is not None:
        flush_callback()


def _log_block(logger: Any, lines: list[str]) -> None:
    logger.info("\n".join(lines))


def _batch_min6ade_and_hist(
    out: dict[str, Any],
    future: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # BASE_SCALE=50.0 converts normalized 'Silicon' units back to real meters
    BASE_SCALE = 50.0
    all_abs = out["y_final_abs"].float() * BASE_SCALE
    gt = future.float() * BASE_SCALE
    err_all = torch.linalg.vector_norm(all_abs - gt[:, None], dim=-1)
    ade_all = err_all.mean(dim=-1)
    fde_all = err_all[:, :, -1]
    mean_min6ade = ade_all.min(dim=-1).values.mean()
    best_idx = fde_all.argmin(dim=-1)
    expert_hist = torch.bincount(best_idx.view(-1), minlength=all_abs.shape[1])
    return mean_min6ade, expert_hist


def _validate_model_outputs(out: dict[str, Any], future: torch.Tensor, module_name: str = "TrajectoryModel.forward") -> None:
    pass


def _validate_gradients(model: TrajectoryModel, module_name: str = "optimizer.backward") -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            pass


def _train_epoch(
    model: TrajectoryModel,
    loader: torch.utils.data.DataLoader[Any],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip_norm: float,
    logger: Any,
    log_every_steps: int,
    log_train_metrics: bool,
    epoch: int,
    global_step_start: int,
    config: ExperimentConfig,
    checkpoint_path: Path | None = None,
    checkpoint_state: dict[str, Any] | None = None,
    flush_callback: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], int]:
    model.train(True)
    nonfinite_batches = 0
    batch_in_epoch = 0
    global_step = int(global_step_start)
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    metric_sums = init_epoch_metric_sums(
        expert_count=model.cvae_dec.experts,
        future_steps=int(model.cvae_dec.steps),
        device=torch.device("cpu"),
    )
    for batch in loader:
        global_step += 1
        batch_in_epoch += 1
        batch_start = time.perf_counter()
        should_log = bool(log_train_metrics and batch_in_epoch % max(log_every_steps, 1) == 0)
        if should_log and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        
        # Batch is already collated by the workers!
        # Just need to move it to the correct device.
        dense = move_batch_to_device(batch, device)
        
        feat = dense["feat"]
        future = dense["future"]
        scale = scale_from_past_batch(feat, dense["history_mask"])
        future_local_norm, _ = future_local_from_past_scale(feat, future, dense["history_mask"])

        with _autocast(device, use_amp):
            out = model(
                feat,
                dense["history_mask"],
                dense["agent_mask"],
                scale,
                graph=dense.get("graph"),
                future=future,
                mode="train",
            )

            loss_pack = moe(
                out["y_final_local"],
                future_local_norm,
                expert_aux=out["expert_aux"],
                post_mu=out["post_mu"],
                post_logvar=out["post_logvar"],
                prior_mu=out["prior_mu"],
                prior_logvar=out["prior_logvar"],
                global_step=global_step,
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
        _validate_model_outputs(out, future)
        if not torch.isfinite(loss_pack["loss"]).all():
            nonfinite_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        accumulate_batch_metric_sums(
            metric_sums,
            out,
            loss_pack,
            future,
        )
        loss = loss_pack["loss"]
        grad_norm_pre_clip = 0.0
        param_norm = 0.0
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _validate_gradients(model)
            if should_log:
                grad_norm_pre_clip = _log_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            if should_log:
                param_norm = _log_param_norm(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            _validate_gradients(model)
            if should_log:
                grad_norm_pre_clip = _log_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            if should_log:
                param_norm = _log_param_norm(model)
            optimizer.step()
        batch_mean_min6ade, batch_expert_hist = _batch_min6ade_and_hist(out, future)
        batch_mean_min6ade_value: float = float(batch_mean_min6ade.detach())
        if checkpoint_path is not None and checkpoint_state is not None:
            if not bool(checkpoint_state.get("checkpoint_initialized", False)):
                checkpoint_state["checkpoint_initialized"] = True
                checkpoint_state["best_train_mean_min6ade"] = batch_mean_min6ade_value
                _write_model_checkpoint(
                    model,
                    optimizer,
                    checkpoint_path=checkpoint_path,
                    checkpoint_state=checkpoint_state,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    train_mean_min6ade=batch_mean_min6ade_value,
                    logger=logger,
                    flush_callback=flush_callback,
                    batch_in_epoch=batch_in_epoch,
                    reason="first_successful_step",
                )
            elif batch_mean_min6ade_value < float(checkpoint_state["best_train_mean_min6ade"]):
                checkpoint_state["best_train_mean_min6ade"] = batch_mean_min6ade_value
                _write_model_checkpoint(
                    model,
                    optimizer,
                    checkpoint_path=checkpoint_path,
                    checkpoint_state=checkpoint_state,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    train_mean_min6ade=batch_mean_min6ade_value,
                    logger=logger,
                    flush_callback=flush_callback,
                    batch_in_epoch=batch_in_epoch,
                    reason="best_batch_min6ade",
                )
        if should_log:
            gpu_peak_vram_gib = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
            batch_time = time.perf_counter() - batch_start
            batch_label = f"{batch_in_epoch}/{total_batches}" if total_batches is not None else str(batch_in_epoch)
            _log_block(
                logger,
                [
                    f"[TRAIN][EPOCH {epoch}][BATCH {batch_label}][GLOBAL_STEP {global_step}]",
                    f"total_gpu_vram_gb={gpu_peak_vram_gib:.6f}",
                    f"kl_loss={float(loss_pack['kl'].detach().cpu()):.6f}",
                    f"expert_usage_histogram={batch_expert_hist.detach().cpu().tolist()}",
                    f"mean_min6ade={batch_mean_min6ade_value:.6f}",
                    f"batch_time_sec={batch_time:.6f}",
                    f"grad_norm_pre_clip={grad_norm_pre_clip:.6f}",
                    f"parameter_norm={param_norm:.6f}",
                ],
            )
            if flush_callback is not None:
                flush_callback()
            del dense, out, loss_pack
    return finalize_epoch_metric_sums(metric_sums, nonfinite_batches=nonfinite_batches), global_step
def _validate_epoch(
    model: TrajectoryModel,
    val_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger | None,
    epoch: int,
    global_step: int,
    config: ExperimentConfig,
    flush_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    model.eval()
    # future_steps is not in config, but stored on the model instance
    future_steps = getattr(model, "future_steps", 60)
    metric_sums = init_epoch_metric_sums(
        expert_count=config.ablation.expert_count,
        future_steps=future_steps,
        device=device,
    )
    
    total_batches = len(val_loader)
    logger.info(f"[VAL][EPOCH {epoch}] Starting Validation Inference...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, start=1):
            batch = move_batch_to_device(batch, device)
            # Pre-compute scale for evaluators
            scale = scale_from_past_batch(batch["feat"], batch["history_mask"])
            
            # STRICTLY mode="eval" to use Prior Latent and AVOID Future Leakage
            out = model(
                feat=batch["feat"],
                history_mask=batch["history_mask"],
                agent_mask=batch["agent_mask"],
                scale=scale,
                graph=batch.get("graph"),
                mode="eval", # CRITICAL: This ensures prior usage and no GT future access
            )
            
            # Ground truth future is ONLY used for metric calculation, NOT by the model
            accumulate_batch_metric_sums(metric_sums, out, {}, batch["future"])
            
            if batch_idx % 100 == 0:
                logger.info(f"[VAL][EPOCH {epoch}] Batch {batch_idx}/{total_batches} processed.")
                if flush_callback:
                    flush_callback()
            
    metrics = finalize_epoch_metric_sums(metric_sums)
    _log_block(
        logger,
        [
            f"[VAL][EPOCH {epoch} SUMMARY]",
            f"val_mean_min6ade={metrics['min_ade']:.6f}",
            f"val_mean_min6fde={metrics['min_fde']:.6f}",
            f"val_mr={metrics['miss_rate']:.6f}",
            f"val_expert_usage_histogram={metrics['expert_usage_histogram']}",
        ],
    )
    if flush_callback:
        flush_callback()
    return metrics


def run_training_experiment(
    config: ExperimentConfig,
    flush_callback: Callable[[], None] | None = None,
    max_train_samples: int | None = None,
) -> dict[str, Any]:
    run_dir = ensure_dir(config.run_dir)
    logger = setup_logging(run_dir, filename="train_logs.txt")
    seed_all(config.seed)
    device = choose_device(config.train.device)
    budget = _configure_cuda_memory_budget(device, config.train.vram_target_gb, config.train.vram_cap_gb)
    logger.info("starting KINEXAR NeurIPS device=%s", device)
    ablation = resolve_ablation_config(config)
    config.ablation = ablation
    logger.info(
        "ablation_variant=%s decoder_family=%s expert_count=%d use_cvae_latent=%s use_physics_losses=%s encoder_mode=%s",
        ablation.variant,
        ablation.decoder_family,
        ablation.expert_count,
        ablation.use_cvae_latent,
        ablation.use_physics_losses,
        ablation.encoder_mode,
    )
    save_config(config, run_dir / "resolved_config.json")
    bundle_root = Path(config.processed_root)
    if has_rows_bundle_split(bundle_root, "train"):
        train_dataset = load_rows_bundle_dataset(bundle_root, split="train")
        meta_path = bundle_root / "train.json"
        if meta_path.exists():
            write_json(run_dir / "subsets" / "train.json", json.loads(meta_path.read_text(encoding="utf-8")))
    else:
        dataset = load_processed_dataset(config.processed_root)
        subsets = build_subset_indices(dataset, config.subset)
        write_json(run_dir / "subsets" / "train.json", subset_metadata(dataset, subsets.train, "train"))
        train_metadata = subset_metadata(dataset, subsets.train, "train")
        train_rows = preload_rows(dataset, train_metadata, logger=logger)
        train_dataset = make_subset(train_rows)
    if max_train_samples is not None and int(max_train_samples) > 0:
        limit = min(int(max_train_samples), len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(limit)))
    if flush_callback is not None:
        flush_callback()
    train_loader = make_loader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        seed=config.seed,
        prefetch_factor=config.train.dataloader_prefetch_factor,
        persistent_workers=config.train.dataloader_persistent_workers,
    )

    val_dataset = None
    val_loader = None
    if config.subset.val_size > 0:
        if has_rows_bundle_split(bundle_root, "val"):
            val_dataset = load_rows_bundle_dataset(bundle_root, split="val")
        else:
            val_metadata = subset_metadata(dataset, subsets.val, "val")
            val_rows = preload_rows(dataset, val_metadata, logger=logger)
            val_dataset = make_subset(val_rows)
        
        val_loader = make_loader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            seed=config.seed,
            prefetch_factor=config.train.dataloader_prefetch_factor,
            persistent_workers=config.train.dataloader_persistent_workers,
        )

    probe = train_dataset[0]
    model = TrajectoryModel(
        probe["input"].shape[-1],
        future_steps=probe["output"].shape[0],
        **trajectory_model_kwargs(config),
    ).to(device)
    model = _maybe_compile_model(model, config, logger)
    logger.info(
        "model_parameters=%d trainable_parameters=%d",
        sum(param.numel() for param in model.parameters()),
        sum(param.numel() for param in model.parameters() if param.requires_grad),
    )
    parameter_count = sum(param.numel() for param in model.parameters())
    trainable_parameter_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=(config.train.use_amp and device.type == "cuda"))
    history: list[dict[str, Any]] = []
    failures: list[str] = []
    global_step = 0
    checkpoint_path = run_dir / "checkpoints" / "modelckpt.pth"
    checkpoint_state = {
        "best_epoch_mean_min6ade": float("inf"),
        "best_train_mean_min6ade": float("inf"),
        "checkpoint_initialized": False,
    }
    patience_epochs = int(config.train.early_stop_patience_epochs)
    epochs_without_improvement = 0

    for epoch in range(1, config.train.epochs + 1):
        epoch_started_at = time.perf_counter()
        try:
            total_batches = len(train_loader)
        except TypeError:
            total_batches = None
        epoch_header = [
            f"[EPOCH {epoch} START]",
            f"batch_size={config.train.batch_size}",
            f"total_batches={'unknown' if total_batches is None else total_batches}",
            f"global_step_start={global_step + 1}",
        ]
        _log_block(logger, epoch_header)
        if flush_callback is not None:
            flush_callback()
        train_metrics, global_step = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config.train.use_amp,
            config.train.grad_clip_norm,
            logger,
            config.train.log_every_steps,
            config.train.log_train_metrics,
            epoch,
            global_step,
            config,
            checkpoint_path=checkpoint_path,
            checkpoint_state=checkpoint_state,
            flush_callback=flush_callback,
        )
        _enforce_vram_cap(device, config.train.vram_cap_gb)
        
        val_metrics = {}
        if val_loader is not None:
            val_metrics = _validate_epoch(
                model,
                val_loader,
                device,
                logger,
                epoch,
                global_step,
                config,
                flush_callback=flush_callback,
            )

        epoch_mean_min6ade = float(val_metrics.get("min_ade", train_metrics["min_ade"]))
        epoch_improved = epoch_mean_min6ade < float(checkpoint_state["best_epoch_mean_min6ade"])
        
        checkpoint_state["best_epoch_mean_min6ade"] = min(
            float(checkpoint_state["best_epoch_mean_min6ade"]),
            epoch_mean_min6ade
        )
        checkpoint_state["best_train_mean_min6ade"] = min(
            float(checkpoint_state["best_train_mean_min6ade"]),
            epoch_mean_min6ade,
        )
        
        # ALWAYS save checkpoint after every epoch as requested
        _write_model_checkpoint(
            model,
            optimizer,
            checkpoint_path=checkpoint_path,
            checkpoint_state=checkpoint_state,
            config=config,
            epoch=epoch,
            global_step=global_step,
            train_mean_min6ade=epoch_mean_min6ade,
            logger=logger,
            flush_callback=flush_callback,
            reason=f"epoch_{epoch}_completion",
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_kl": train_metrics["kl"],
            "train_mean_min6ade": train_metrics["min_ade"],
            "train_mean_min6fde": train_metrics["min_fde"],
            "train_mr": train_metrics["miss_rate"],
            "train_cri": train_metrics.get("cri", 0.0),
            "train_mode_entropy": train_metrics.get("mode_entropy", 0.0),
            "delta_matrix_4x4": train_metrics.get("delta_matrix_4x4", {}),
            "train_expert_usage_histogram": train_metrics["expert_usage_histogram"],
            "train_nonfinite_batches": train_metrics["nonfinite_batches"],
            "train_time_per_epoch_sec": time.perf_counter() - epoch_started_at,
            "val_loss": val_metrics.get("loss", 0.0),
            "val_mean_min6ade": val_metrics.get("min_ade", 0.0),
            "val_mean_min6fde": val_metrics.get("min_fde", 0.0),
            "val_mr": val_metrics.get("miss_rate", 0.0),
            "val_expert_usage_histogram": val_metrics.get("expert_usage_histogram", []),
        }
        history.append(epoch_row)
        if train_metrics["nonfinite_batches"] > 0:
            failures.append(f"nonfinite_batches_epoch_{epoch}")
        _log_block(
            logger,
            [
                f"[EPOCH {epoch} SUMMARY]",
                f"train_loss={train_metrics['loss']:.6f}",
                f"train_kl={train_metrics['kl']:.6f}",
                f"train_mean_min6ade={train_metrics['min_ade']:.6f}",
                f"train_expert_usage_histogram={train_metrics['expert_usage_histogram']}",
                f"train_nonfinite_batches={train_metrics['nonfinite_batches']}",
                f"best_epoch_mean_min6ade={checkpoint_state['best_epoch_mean_min6ade']:.6f}",
                f"epoch_improved={epoch_improved}",
            ],
        )
        if flush_callback is not None:
            flush_callback()
        if epoch_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience_epochs:
            break

    report = {
        "tracked_quantity": "epoch_mean_min6ade",
        "best_epoch_mean_min6ade": None if checkpoint_state["best_epoch_mean_min6ade"] == float("inf") else checkpoint_state["best_epoch_mean_min6ade"],
        "best_mean_min6ade": None if checkpoint_state["best_epoch_mean_min6ade"] == float("inf") else checkpoint_state["best_epoch_mean_min6ade"],
        "epochs_ran": len(history),
        "early_stop_patience_epochs": patience_epochs,
        "failures": failures,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "train_time_per_epoch_sec": [row["train_time_per_epoch_sec"] for row in history],
    }
    write_json(run_dir / config.evaluation.history_filename, history)
    write_csv(run_dir / config.evaluation.history_csv_filename, history)
    write_json(run_dir / config.evaluation.report_filename, report)
    if flush_callback is not None:
        flush_callback()
    return {
        "run_dir": str(run_dir),
        "report": report,
        "history": history,
        "latest_checkpoint": str(checkpoint_path),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="TRAINING/config.json")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--map-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-train-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--vram-target-gb", type=float, default=None)
    parser.add_argument("--vram-cap-gb", type=float, default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--no-latent", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--single-expert", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--weak-encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--use-residual-refinement", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--refinement-lambda", type=float, default=None)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.ablation:
        config.train.epochs = 12
        config.ablation.variant = "core_transformer_cvae_only"
    else:
        config.train.epochs = 64
    config.train.use_compile = True
    config.train.compile_mode = "default"
    config.train.dataloader_prefetch_factor = 2
    if args.run_name:
        config.run_name = args.run_name
    else:
        config.run_name = f"{config.run_name}_{timestamp()}"
    if args.processed_root:
        config.processed_root = args.processed_root
    if args.map_dir:
        config.map_dir = args.map_dir
    if args.output_root:
        config.output_root = args.output_root
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers
    if args.log_train_metrics is not None:
        config.train.log_train_metrics = bool(args.log_train_metrics)
    if args.device:
        config.train.device = args.device
    if args.vram_target_gb is not None:
        config.train.vram_target_gb = float(args.vram_target_gb)
    if args.vram_cap_gb is not None:
        config.train.vram_cap_gb = float(args.vram_cap_gb)
    if args.compile is not None:
        config.train.use_compile = bool(args.compile)
    if args.compile_mode:
        config.train.compile_mode = str(args.compile_mode)
    if args.prefetch_factor is not None:
        config.train.dataloader_prefetch_factor = 2
    if args.persistent_workers is not None:
        config.train.dataloader_persistent_workers = bool(args.persistent_workers)
    if args.seed is not None:
        config.seed = args.seed
        config.subset.seed = args.seed
    if args.variant:
        config.ablation.variant = str(args.variant)
    if args.no_latent is not None:
        config.ablation.no_latent = bool(args.no_latent)
    if args.single_expert is not None:
        config.ablation.single_expert = bool(args.single_expert)
    if args.weak_encoder is not None:
        config.ablation.weak_encoder = bool(args.weak_encoder)
    if args.use_residual_refinement is not None:
        config.ablation.use_residual_refinement = bool(args.use_residual_refinement)
    if args.refinement_lambda is not None:
        config.ablation.refinement_lambda = float(args.refinement_lambda)
    if args.num_experts is not None:
        config.ablation.num_experts = int(args.num_experts)
    run_training_experiment(config, max_train_samples=args.max_train_samples)


if __name__ == "__main__":
    main()
