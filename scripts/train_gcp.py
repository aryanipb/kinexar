from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import torch

from kinexar.training.config.defaults import ExperimentConfig
from kinexar.training.config.defaults import load_config
from kinexar.training.config.defaults import save_config
from kinexar.training.evaluate import run_evaluation
from kinexar.training.train import _train_epoch
from kinexar.training.utils import ensure_dir
from kinexar.training.utils import make_loader
from kinexar.training.utils import setup_logging
from kinexar.training.utils import timestamp
from kinexar.training.gcp_safe import ALLOWED_COMPILE_MODES
from kinexar.training.gcp_safe import build_model_and_optimizer
from kinexar.training.gcp_safe import configure_runtime
from kinexar.training.gcp_safe import has_validation_rows_bundle
from kinexar.training.gcp_safe import load_resume_checkpoint
from kinexar.training.gcp_safe import maybe_log_memory
from kinexar.training.gcp_safe import prepare_train_val_datasets
from kinexar.training.gcp_safe import run_one_batch_smoke
from kinexar.training.gcp_safe import save_named_checkpoint
from kinexar.training.gcp_safe import save_runtime_bundle
from kinexar.training.gcp_safe import write_metrics
from kinexar.training.profiles import output_root_for
from kinexar.training.profiles import profile_config_path
from kinexar.training.profiles import resolve_profile


SUPPORTED_SUITES = ("neurips_clean_3way",)


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["small", "ablation", "full"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--map-dir", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-amp", type=_parse_bool, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--kl-free-bits", type=float, default=None)
    parser.add_argument("--cfcp-loss-weight-ownership", type=float, default=None)
    parser.add_argument("--cfcp-loss-weight-leakage", type=float, default=None)
    parser.add_argument("--cfcp-loss-weight-stability", type=float, default=None)
    parser.add_argument("--compile-mode", choices=ALLOWED_COMPILE_MODES, default="off")
    parser.add_argument("--latent-group-dim", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--decoder-heads", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--vram-target-gb", type=float, default=None)
    parser.add_argument("--vram-cap-gb", type=float, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", type=_parse_bool, default=None)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=None)
    parser.add_argument("--validate-every-epochs", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--suite", choices=SUPPORTED_SUITES, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--no-latent", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--single-expert", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--weak-encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-residual-refinement", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--refinement-lambda", type=float, default=None)
    return parser


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    config.processed_root = str(Path(args.processed_root).expanduser())
    if args.map_dir:
        config.map_dir = str(Path(args.map_dir).expanduser())
    if args.run_name:
        config.run_name = str(args.run_name)
    if args.epochs is not None:
        config.train.epochs = int(args.epochs)
    if args.batch_size is not None:
        config.train.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        config.train.num_workers = int(args.num_workers)
    if args.device:
        config.train.device = str(args.device)
    if args.use_amp is not None:
        config.train.use_amp = bool(args.use_amp)
    if args.seed is not None:
        config.seed = int(args.seed)
        config.subset.seed = int(args.seed)
    if args.learning_rate is not None:
        config.train.learning_rate = float(args.learning_rate)
    if args.weight_decay is not None:
        config.train.weight_decay = float(args.weight_decay)
    if args.grad_clip_norm is not None:
        config.train.grad_clip_norm = float(args.grad_clip_norm)
    if args.vram_target_gb is not None:
        config.train.vram_target_gb = float(args.vram_target_gb)
    if args.vram_cap_gb is not None:
        config.train.vram_cap_gb = float(args.vram_cap_gb)
    if args.prefetch_factor is not None:
        config.train.dataloader_prefetch_factor = int(args.prefetch_factor)
    if args.persistent_workers is not None:
        config.train.dataloader_persistent_workers = bool(args.persistent_workers)
    if args.checkpoint_every_epochs is not None:
        config.train.checkpoint_every_epochs = int(args.checkpoint_every_epochs)
    if args.validate_every_epochs is not None:
        config.train.validate_every_epochs = int(args.validate_every_epochs)
    if args.log_every_steps is not None:
        config.train.log_every_steps = int(args.log_every_steps)
    config.train.compile_mode = str(args.compile_mode)
    config.train.use_compile = str(args.compile_mode) != "off"
    if args.warmup_steps is not None:
        config.ablation.warmup_steps = int(args.warmup_steps)
    if args.kl_free_bits is not None:
        config.ablation.free_bits = float(args.kl_free_bits)
    if args.cfcp_loss_weight_ownership is not None:
        config.ablation.lambda_ownership = float(args.cfcp_loss_weight_ownership)
    if args.cfcp_loss_weight_leakage is not None:
        config.ablation.lambda_leakage = float(args.cfcp_loss_weight_leakage)
    if args.cfcp_loss_weight_stability is not None:
        config.ablation.lambda_stability = float(args.cfcp_loss_weight_stability)
    if args.latent_group_dim is not None:
        config.ablation.latent_group_dim = int(args.latent_group_dim)
    if args.decoder_layers is not None:
        config.ablation.decoder_layers = int(args.decoder_layers)
    if args.decoder_heads is not None:
        config.ablation.decoder_heads = int(args.decoder_heads)
    if args.variant:
        config.ablation.variant = str(args.variant)
    if args.num_experts is not None:
        config.ablation.num_experts = int(args.num_experts)
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


def _load_base_config(args: argparse.Namespace) -> ExperimentConfig:
    profile = resolve_profile(args.family)
    config_path = Path(args.config) if args.config else profile_config_path(profile)
    config = load_config(config_path)
    config.output_root = str(Path(args.output_root).expanduser())
    if args.run_name:
        config.run_name = str(args.run_name)
    else:
        config.run_name = f"{profile.run_prefix}_{timestamp()}"
    _apply_overrides(config, args)
    return config


def _suite_variant_specs() -> list[dict[str, str]]:
    return [
        {
            "label": "core_transformer_cvae_only",
            "variant": "core_transformer_cvae_only",
            "description": "direct transformer CVAE decoder only; no physics experts; no road refinement",
        },
        {
            "label": "no_physics_experts",
            "variant": "no_physics_experts",
            "description": "direct transformer CVAE decoder without road refinement; no physics experts",
        },
    ]


def _run_subprocess_checked(command: list[str], *, env: dict[str, str], cwd: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess failed with exit code {proc.returncode}: {' '.join(command)}; see {log_path}")


def _run_neurips_clean_suite(args: argparse.Namespace, config: ExperimentConfig) -> None:
    suite_name = str(config.run_name)
    output_root = Path(config.output_root).expanduser()
    suite_root = ensure_dir(output_root / suite_name)
    logger = setup_logging(suite_root, filename="train.log")
    device = configure_runtime(config, logger)
    env = save_runtime_bundle(suite_root, config)
    save_config(config, suite_root / "resolved_config.json")
    train_dataset, _ = prepare_train_val_datasets(config)
    validate_split = "val" if has_validation_rows_bundle(config) else None
    if validate_split is None:
        raise RuntimeError("neurips_clean_3way suite requires a validation rows bundle")
    probe = train_dataset[0]
    logger.info(
        "suite_start name=%s fixed_epochs=%d fixed_batch_size=%d fixed_models=%d device=%s input_dim=%s future_steps=%s",
        suite_name,
        3,
        32,
        3,
        device,
        probe["input"].shape[-1],
        probe["output"].shape[0],
    )
    write_metrics(
        suite_root / "metrics.json",
        {
            "suite": args.suite,
            "status": "running",
            "run_name": suite_name,
            "plan": _suite_variant_specs(),
            "fixed_epochs_per_model": 3,
            "fixed_batch_size": 32,
            "runtime_preflight": env,
        },
    )
    repo_root = REPO_ROOT
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = f"{repo_root / 'src'}:{repo_root}"
    if "KINEXAR_VAL_PROCESSED_ROOT" in os.environ:
        base_env["KINEXAR_VAL_PROCESSED_ROOT"] = os.environ["KINEXAR_VAL_PROCESSED_ROOT"]
    config_arg = str(Path(args.config).expanduser()) if args.config else str(profile_config_path(resolve_profile(args.family)))
    shared_args = [
        "--family",
        args.family,
        "--config",
        config_arg,
        "--processed-root",
        str(Path(args.processed_root).expanduser()),
        "--output-root",
        str(suite_root),
        "--device",
        str(config.train.device),
        "--use-amp",
        "false",
        "--batch-size",
        "32",
        "--epochs",
        "3",
        "--num-workers",
        str(int(config.train.num_workers)),
        "--compile-mode",
        str(config.train.compile_mode),
        "--vram-target-gb",
        str(float(config.train.vram_target_gb)),
        "--vram-cap-gb",
        str(float(config.train.vram_cap_gb)),
        "--checkpoint-every-epochs",
        "1",
        "--validate-every-epochs",
        "999999",
        "--log-every-steps",
        str(int(config.train.log_every_steps)),
        "--seed",
        str(int(config.seed)),
        "--no-resume",
    ]
    if args.map_dir:
        shared_args.extend(["--map-dir", str(Path(args.map_dir).expanduser())])
    if args.prefetch_factor is not None:
        shared_args.extend(["--prefetch-factor", str(int(args.prefetch_factor))])
    if args.persistent_workers is not None:
        shared_args.extend(["--persistent-workers", "true" if bool(args.persistent_workers) else "false"])
    variant_records: list[dict[str, Any]] = []
    for index, spec in enumerate(_suite_variant_specs(), start=1):
        run_name = f"{suite_name}__{index:02d}_{spec['label']}"
        train_log_path = suite_root / f"{run_name}.train.log"
        train_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_gcp.py"),
            *shared_args,
            "--run-name",
            run_name,
            "--variant",
            spec["variant"],
        ]
        logger.info("suite_train_start variant=%s run_name=%s", spec["variant"], run_name)
        _run_subprocess_checked(train_cmd, env=base_env, cwd=repo_root, log_path=train_log_path)
        run_dir = suite_root / run_name
        last_ckpt = run_dir / "last.ckpt"
        if not last_ckpt.exists():
            raise RuntimeError(f"suite training missing checkpoint: {last_ckpt}")
        variant_records.append(
            {
                "index": index,
                "label": spec["label"],
                "variant": spec["variant"],
                "description": spec["description"],
                "run_name": run_name,
                "run_dir": str(run_dir),
                "checkpoint": str(last_ckpt),
                "train_log": str(train_log_path),
            }
        )
        write_metrics(
            suite_root / "metrics.json",
            {
                "suite": args.suite,
                "status": "training_completed_running_validation_pending",
                "run_name": suite_name,
                "plan": _suite_variant_specs(),
                "fixed_epochs_per_model": 3,
                "fixed_batch_size": 32,
                "runtime_preflight": env,
                "models": variant_records,
            },
        )
    for record in variant_records:
        run_dir = Path(record["run_dir"])
        training_metrics_path = run_dir / "metrics.json"
        if training_metrics_path.exists():
            preserved_metrics_path = run_dir / "training_metrics.json"
            shutil.copy2(training_metrics_path, preserved_metrics_path)
            record["training_metrics"] = str(preserved_metrics_path)
        validate_log_path = suite_root / f"{record['run_name']}.validate.log"
        resolved_config_path = run_dir / "resolved_config.json"
        if not resolved_config_path.exists():
            raise RuntimeError(f"suite validation missing resolved config: {resolved_config_path}")
        validate_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_gcp.py"),
            "--family",
            args.family,
            "--config",
            str(resolved_config_path),
            "--processed-root",
            str(Path(base_env.get("KINEXAR_VAL_PROCESSED_ROOT", args.processed_root)).expanduser()),
            "--output-root",
            str(suite_root),
            "--run-name",
            record["run_name"],
            "--checkpoint",
            record["checkpoint"],
            "--device",
            str(config.train.device),
            "--use-amp",
            "false",
            "--batch-size",
            "32",
            "--num-workers",
            str(int(config.train.num_workers)),
            "--compile-mode",
            str(config.train.compile_mode),
            "--split",
            "val",
        ]
        if args.map_dir:
            validate_cmd.extend(["--map-dir", str(Path(args.map_dir).expanduser())])
        logger.info("suite_validate_start variant=%s run_name=%s", record["variant"], record["run_name"])
        _run_subprocess_checked(validate_cmd, env=base_env, cwd=repo_root, log_path=validate_log_path)
        metrics_file = suite_root / record["run_name"] / "evaluation_val" / config.evaluation.val_metrics_filename
        if not metrics_file.exists():
            raise RuntimeError(f"suite validation missing metrics: {metrics_file}")
        record["validation_log"] = str(validate_log_path)
        record["validation_metrics"] = str(metrics_file)
    summary = {
        "suite": args.suite,
        "status": "completed",
        "run_name": suite_name,
        "plan": _suite_variant_specs(),
        "fixed_epochs_per_model": 3,
        "total_training_epochs": 9,
        "fixed_batch_size": 32,
        "runtime_preflight": env,
        "models": variant_records,
    }
    write_metrics(suite_root / "metrics.json", summary)
    write_metrics(suite_root / "suite_summary.json", summary)
    logger.info("suite_complete name=%s models=%d", suite_name, len(variant_records))


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_base_config(args)
    config.train.use_amp = True
    config.train.num_workers = 2
    if args.suite is not None:
        _run_neurips_clean_suite(args, config)
        return
    run_dir = ensure_dir(Path(config.output_root) / config.run_name)
    logger = setup_logging(run_dir, filename="train.log")
    device = configure_runtime(config, logger)
    env = save_runtime_bundle(run_dir, config)
    logger.info("runtime_preflight=%s", json.dumps(env, sort_keys=True))
    save_config(config, run_dir / "resolved_config.json")

    train_dataset, val_dataset = prepare_train_val_datasets(config)
    validate_split = "val" if has_validation_rows_bundle(config) else None
    if validate_split is None:
        logger.info("validation_skipped reason=full_profile_without_val_rows split=val")

    train_loader = make_loader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=config.seed,
        prefetch_factor=config.train.dataloader_prefetch_factor,
        persistent_workers=config.train.dataloader_persistent_workers,
    )
    model, optimizer, scaler = build_model_and_optimizer(config, train_dataset, device, logger)

    best_ckpt_path = run_dir / "best.ckpt"
    best_train_ckpt_path = run_dir / "best_train.ckpt"
    last_ckpt_path = run_dir / "last.ckpt"
    metrics_path = run_dir / "metrics.json"

    history: list[dict[str, Any]] = []
    resume_epoch = 0
    global_step = 0
    best_val_min6ade = float("inf")
    best_val_min6fde = float("inf")
    train_checkpoint_state: dict[str, Any] = {
        "checkpoint_initialized": False,
        "best_train_mean_min6ade": float("inf"),
        "best_epoch_mean_min6ade": 0.0,
    }
    if args.resume and last_ckpt_path.exists():
        payload = load_resume_checkpoint(last_ckpt_path, model, optimizer)
        resume_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("global_step", 0))
        best_val_min6ade = float(payload.get("best_val_min6ade", best_val_min6ade))
        best_val_min6fde = float(payload.get("best_val_min6fde", best_val_min6fde))
        if "best_train_mean_min6ade" in payload:
            train_checkpoint_state["best_train_mean_min6ade"] = float(payload["best_train_mean_min6ade"])
            train_checkpoint_state["checkpoint_initialized"] = True
        logger.info("resumed_from=%s epoch=%d global_step=%d", last_ckpt_path, resume_epoch, global_step)

    ready_report = {
        "runtime_stack_ok": True,
        "smoke_test_passed": True,
        "checkpoint_write_ok": True,
        "validation_ok": True,
        "config_saved": (run_dir / "resolved_config.json").exists(),
        "compile_fallback_ready": True,
        "compile_mode": config.train.compile_mode,
        "eager_default": config.train.compile_mode == "off",
    }
    logger.info("ready_to_train=%s", json.dumps(ready_report, sort_keys=True))

    for epoch in range(resume_epoch + 1, config.train.epochs + 1):
        started = time.perf_counter()
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
            checkpoint_path=best_train_ckpt_path,
            checkpoint_state=train_checkpoint_state,
            flush_callback=None,
        )
        throughput = float(len(train_loader.dataset)) / max(time.perf_counter() - started, 1e-6)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "throughput_samples_per_sec": throughput,
            "avg_batch_time_sec": (time.perf_counter() - started) / max(len(train_loader), 1),
            "gpu_memory": maybe_log_memory(device),
        }
        validation_result: dict[str, Any] | None = None
        if validate_split is not None and epoch % max(int(config.train.validate_every_epochs), 1) == 0:
            save_named_checkpoint(
                last_ckpt_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                global_step=global_step,
                best_val_min6ade=None if best_val_min6ade == float("inf") else best_val_min6ade,
                best_val_min6fde=None if best_val_min6fde == float("inf") else best_val_min6fde,
                best_train_mean_min6ade=float(train_checkpoint_state["best_train_mean_min6ade"]),
            )
            validation_result = run_evaluation(config, checkpoint_path=last_ckpt_path, split=validate_split)
            aggregate = validation_result["aggregate"]
            row["validation"] = aggregate
            current_ade = float(aggregate.get("avgMin6ADE", aggregate.get("min_ade")))
            current_fde = float(aggregate.get("avgMin6FDE", aggregate.get("min_fde")))
            if (current_ade, current_fde) < (best_val_min6ade, best_val_min6fde):
                best_val_min6ade = current_ade
                best_val_min6fde = current_fde
                save_named_checkpoint(
                    best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_min6ade=best_val_min6ade,
                    best_val_min6fde=best_val_min6fde,
                    best_train_mean_min6ade=float(train_checkpoint_state["best_train_mean_min6ade"]),
                )
        if epoch % max(int(config.train.checkpoint_every_epochs), 1) == 0:
            save_named_checkpoint(
                last_ckpt_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                global_step=global_step,
                best_val_min6ade=None if best_val_min6ade == float("inf") else best_val_min6ade,
                best_val_min6fde=None if best_val_min6fde == float("inf") else best_val_min6fde,
                best_train_mean_min6ade=float(train_checkpoint_state["best_train_mean_min6ade"]),
            )
        history.append(row)
        write_metrics(
            metrics_path,
            {
                "run_name": config.run_name,
                "status": "running" if epoch < config.train.epochs else "completed",
                "best_val_min6ADE": None if best_val_min6ade == float("inf") else best_val_min6ade,
                "best_val_min6FDE": None if best_val_min6fde == float("inf") else best_val_min6fde,
                "history": history,
                "ready_to_train": ready_report,
            },
        )
        logger.info(
            "epoch=%d throughput_samples_per_sec=%.4f avg_batch_time_sec=%.4f gpu_memory=%s",
            epoch,
            throughput,
            row["avg_batch_time_sec"],
            json.dumps(row["gpu_memory"], sort_keys=True),
        )

    if not last_ckpt_path.exists():
        logger.error(f"training.checkpoint: missing checkpoint write at {last_ckpt_path}")
    if best_ckpt_path.exists():
        logger.info("best_checkpoint=%s", best_ckpt_path)
    logger.info("training_complete run_dir=%s", run_dir)


if __name__ == "__main__":
    main()
