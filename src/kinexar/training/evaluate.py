from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from kinexar.models.model import TrajectoryModel
from kinexar.training.batching import collate_dense_batch
from kinexar.training.batching import move_batch_to_device
from kinexar.training.batching import future_local_from_past_scale
from kinexar.training.batching import scale_from_past_batch
from kinexar.training.losses import moe

from kinexar.training.config.defaults import ExperimentConfig
from kinexar.training.config.defaults import config_from_dict
from kinexar.training.config.defaults import load_config
from kinexar.training.config.defaults import resolve_ablation_config
from kinexar.training.config.defaults import trajectory_model_kwargs
from kinexar.training.metrics import SampleMetrics
from kinexar.training.metrics import aggregate_epoch_metrics
from kinexar.training.metrics import collect_batch_sample_metrics
from kinexar.training.utils import choose_device
from kinexar.training.utils import ensure_dir
from kinexar.training.utils import has_rows_bundle_split
from kinexar.training.utils import load_processed_dataset
from kinexar.training.utils import load_rows_bundle_dataset
from kinexar.training.utils import make_loader
from kinexar.training.utils import make_subset
from kinexar.training.utils import preload_rows
from kinexar.training.utils import setup_logging
from kinexar.training.utils import write_csv
from kinexar.training.utils import write_json
from kinexar.training.subset_builder import build_subset_indices
from kinexar.training.subset_builder import subset_metadata


VALIDATION_PROCESSED_ROOT_ENV = "KINEXAR_VAL_PROCESSED_ROOT"


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(enabled and device.type == "cuda"))


def _validate_model_outputs(out: dict[str, Any], future: torch.Tensor, module_name: str = "TrajectoryModel.forward.eval") -> None:
    for key in ("y_all", "y_final_local", "y_final_abs", "pred_abs", "scale"):
        if key not in out:
            return
    batch = future.shape[0]
    expected_t = future.shape[1]
    for key in ("y_all", "y_final_local", "y_final_abs", "pred_abs"):
        value = out[key]
        if not isinstance(value, torch.Tensor):
            return
        if value.ndim != 4 or value.shape[0] != batch or value.shape[2] != expected_t or value.shape[3] != future.shape[2]:
            return


def evaluate_loader(
    model: TrajectoryModel,
    loader: torch.utils.data.DataLoader[Any],
    device: torch.device,
    use_amp: bool,
    use_physics_loss: bool,
    config: ExperimentConfig,
    visualize_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples: list[SampleMetrics] = []
    model.train(False)
    nonfinite_batches = 0
    total_inference_sec = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            # Batch is already collated by the workers!
            dense = move_batch_to_device(batch, device)
            feat = dense["feat"]
            future = dense["future"]
            scale = scale_from_past_batch(feat, dense["history_mask"])
            future_local_norm, _ = future_local_from_past_scale(feat, future, dense["history_mask"])
            started_at = time.perf_counter()
            with _autocast(device, use_amp):
                out = model(
                    feat,
                    dense["history_mask"],
                    dense["agent_mask"],
                    scale,
                    graph=dense.get("graph"),
                    future=None,
                    mode="eval",
                )
                loss_pack = moe(
                    out["y_final_local"],
                    future_local_norm,
                    expert_aux=out["expert_aux"],
                    prior_mu=out["prior_mu"],
                    prior_logvar=out["prior_logvar"],
                    free_bits=config.ablation.free_bits,
                    use_physics_loss=use_physics_loss,
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_inference_sec += time.perf_counter() - started_at
            total_samples += int(future.shape[0])
            if not torch.isfinite(loss_pack["loss"]).all():
                nonfinite_batches += 1
            
            batch_samples = collect_batch_sample_metrics(
                dense["rows"],
                out,
                loss_pack,
                future,
            )
            
            if visualize_dir:
                pass

            samples.extend(batch_samples)
    aggregate = aggregate_epoch_metrics(samples, expert_count=model.cvae_dec.experts, nonfinite_batches=nonfinite_batches)
    aggregate["inference_ms_per_sample"] = 1000.0 * total_inference_sec / max(total_samples, 1)
    sample_rows = [asdict(sample) for sample in samples]
    return aggregate, sample_rows


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device, din: int) -> TrajectoryModel:
    payload = torch.load(checkpoint_path, map_location=device)
    future_steps = int(payload.get("future_steps", 60))
    model_kwargs: dict[str, Any] = {}
    if isinstance(payload.get("config"), dict):
        config = config_from_dict(payload["config"])
        model_kwargs = trajectory_model_kwargs(config)
    model = TrajectoryModel(din, future_steps=future_steps, **model_kwargs).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model


def run_evaluation(config: ExperimentConfig, checkpoint_path: str | Path, split: str = "val", visualize: bool = False) -> dict[str, Any]:
    logger = setup_logging(Path(config.run_dir), filename="train.log")
    config.ablation = resolve_ablation_config(config)
    device = choose_device(config.train.device)
    bundle_root = Path(config.processed_root)
    if split == "val":
        bundle_root = Path(os.environ.get(VALIDATION_PROCESSED_ROOT_ENV, str(bundle_root))).expanduser()
    if has_rows_bundle_split(bundle_root, split):
        eval_dataset = load_rows_bundle_dataset(bundle_root, split=split)
        meta_path = bundle_root / f"{split}.json"
        if meta_path.exists():
            write_json(ensure_dir(Path(config.run_dir) / f"evaluation_{split}") / f"{split}_subset.json", json.loads(meta_path.read_text(encoding="utf-8")))
    else:
        dataset = load_processed_dataset(config.processed_root)
        subsets = build_subset_indices(dataset, config.subset)
        indices = subsets.val if split == "val" else subsets.train if split == "train" else subsets.debug
        metadata = subset_metadata(dataset, indices, split)
        rows = preload_rows(dataset, metadata, logger=logger)
        eval_dataset = make_subset(rows)
        write_json(ensure_dir(Path(config.run_dir) / f"evaluation_{split}") / f"{split}_subset.json", subset_metadata(dataset, indices, split))
    loader = make_loader(
        eval_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        seed=config.seed,
        prefetch_factor=config.train.dataloader_prefetch_factor,
        persistent_workers=config.train.dataloader_persistent_workers,
    )
    probe = eval_dataset[0]
    model = load_model_from_checkpoint(checkpoint_path, device, probe["input"].shape[-1])
    compile_mode = str(getattr(config.train, "compile_mode", "off") or "off").strip().lower()
    if bool(getattr(config.train, "use_compile", False)) and compile_mode != "off" and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, dynamic=True, mode=compile_mode)
        except Exception:
            pass
    logger.info("evaluating split=%s checkpoint=%s", split, checkpoint_path)
    eval_dir = ensure_dir(Path(config.run_dir) / f"evaluation_{split}")
    visualize_dir = ensure_dir(eval_dir / "plots") if visualize else None

    aggregate, sample_rows = evaluate_loader(
        model,
        loader,
        device,
        config.train.use_amp,
        use_physics_loss=config.ablation.use_physics_losses,
        config=config,
        visualize_dir=visualize_dir,
    )
    write_json(eval_dir / config.evaluation.val_metrics_filename, aggregate)
    metrics_path = eval_dir / config.evaluation.val_metrics_filename
    if not metrics_path.exists() or metrics_path.stat().st_size <= 0:
        raise RuntimeError(f"evaluation.metrics: missing validation metrics file at {metrics_path}")
    write_csv(eval_dir / config.evaluation.val_samples_filename, sample_rows)
    result = {
        "aggregate": aggregate,
        "sample_rows": sample_rows,
        "split": split,
        "checkpoint_path": str(checkpoint_path),
    }
    logger.info(
        "evaluation finished split=%s min6ade=%.6f min6fde=%.6f miss_rate_2m=%.6f ade_mean=%.6f fde_mean=%.6f",
        split,
        aggregate["min_ade"],
        aggregate["min_fde"],
        aggregate["miss_rate"],
        aggregate["ade"],
        aggregate["fde"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/small_modal_1500_500_seed7.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "debug"])
    parser.add_argument("--variant", default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.variant:
        config.ablation.variant = str(args.variant)
    run_evaluation(config, args.checkpoint, split=args.split, visualize=args.visualize)


if __name__ == "__main__":
    main()
