from __future__ import annotations

import argparse
import json
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
from kinexar.training.utils import ensure_dir
from kinexar.training.utils import setup_logging
from kinexar.training.gcp_safe import ALLOWED_COMPILE_MODES
from kinexar.training.gcp_safe import configure_runtime
from kinexar.training.gcp_safe import prepare_train_val_datasets
from kinexar.training.gcp_safe import run_one_batch_smoke
from kinexar.training.gcp_safe import save_runtime_bundle
from kinexar.training.profiles import profile_config_path
from kinexar.training.profiles import resolve_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["small", "ablation", "full"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--map-dir", default=None)
    parser.add_argument("--run-dir", default="runs/runtime_check")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compile-mode", choices=ALLOWED_COMPILE_MODES, default="off")
    args = parser.parse_args()

    profile = resolve_profile(args.family)
    config_path = Path(args.config) if args.config else profile_config_path(profile)
    config = load_config(config_path)
    config.processed_root = str(Path(args.processed_root).expanduser())
    config.output_root = str(Path(args.run_dir).expanduser().parent)
    config.run_name = Path(args.run_dir).expanduser().name
    config.train.batch_size = int(args.batch_size)
    config.train.num_workers = int(args.num_workers)
    config.train.compile_mode = str(args.compile_mode)
    config.train.use_compile = str(args.compile_mode) != "off"
    if args.map_dir:
        config.map_dir = str(Path(args.map_dir).expanduser())
    if args.device:
        config.train.device = str(args.device)

    run_dir = ensure_dir(Path(args.run_dir).expanduser())
    logger = setup_logging(run_dir, filename="train.log")
    configure_runtime(config, logger)
    env = save_runtime_bundle(run_dir, config)
    save_config(config, run_dir / "resolved_config.json")
    print(json.dumps(env, indent=2, sort_keys=True), flush=True)
    train_dataset, val_dataset = prepare_train_val_datasets(config)
    smoke = run_one_batch_smoke(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        run_dir=run_dir,
        logger=logger,
    )
    print(json.dumps({"status": "ok", "smoke": smoke}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
