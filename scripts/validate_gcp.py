from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from kinexar.training.config.defaults import load_config
from kinexar.training.config.defaults import save_config
from kinexar.training.evaluate import run_evaluation
from kinexar.training.utils import ensure_dir
from kinexar.training.utils import setup_logging
from kinexar.training.gcp_safe import ALLOWED_COMPILE_MODES
from kinexar.training.gcp_safe import configure_runtime
from kinexar.training.gcp_safe import prepare_train_val_datasets
from kinexar.training.gcp_safe import save_runtime_bundle
from kinexar.training.gcp_safe import write_metrics
from kinexar.training.profiles import profile_config_path
from kinexar.training.profiles import resolve_profile


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["small", "ablation", "full"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--map-dir", default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-amp", type=_parse_bool, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--compile-mode", choices=ALLOWED_COMPILE_MODES, default="off")
    parser.add_argument("--split", default="val", choices=["train", "val", "debug"])
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    profile = resolve_profile(args.family)
    config_path = Path(args.config) if args.config else profile_config_path(profile)
    config = load_config(config_path)
    config.processed_root = str(Path(args.processed_root).expanduser())
    config.output_root = str(Path(args.output_root).expanduser())
    config.run_name = str(args.run_name)
    config.train.compile_mode = str(args.compile_mode)
    config.train.use_compile = str(args.compile_mode) != "off"
    if args.map_dir:
        config.map_dir = str(Path(args.map_dir).expanduser())
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

    run_dir = ensure_dir(Path(config.output_root) / config.run_name)
    logger = setup_logging(run_dir, filename="train.log")
    configure_runtime(config, logger)
    save_runtime_bundle(run_dir, config)
    save_config(config, run_dir / "resolved_config.json")
    prepare_train_val_datasets(config)
    result = run_evaluation(config, checkpoint_path=checkpoint_path, split=args.split)
    write_metrics(
        run_dir / "metrics.json",
        {
            "run_name": config.run_name,
            "status": "validated",
            "split": args.split,
            "validation": result["aggregate"],
            "checkpoint": str(checkpoint_path),
        },
    )


if __name__ == "__main__":
    main()
