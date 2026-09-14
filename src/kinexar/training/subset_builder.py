from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinexar.dataset.data.dataset import TrajectoryDataset

from kinexar.training.config.defaults import load_config
from kinexar.training.config.defaults import SubsetConfig
from kinexar.training.utils import preload_rows
from kinexar.training.utils import write_json


@dataclass(frozen=True)
class SubsetIndices:
    train: list[int]
    val: list[int]
    debug: list[int]


def _sample_rows_by_scenario(
    dataset: TrajectoryDataset,
    cfg: SubsetConfig,
) -> SubsetIndices:
    rows = list(dataset.rows)
    if not rows:
        raise ValueError("processed AV2 dataset is empty")
    by_scenario: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_scenario.setdefault(str(row["scenario_id"]), []).append(index)
    scenario_ids = sorted(by_scenario)
    rng = random.Random(cfg.seed)
    rng.shuffle(scenario_ids)

    train: list[int] = []
    val: list[int] = []
    used_scenarios: set[str] = set()

    for scenario_id in scenario_ids:
        indices = list(by_scenario[scenario_id])
        rng.shuffle(indices)
        if len(train) < cfg.train_size:
            remaining = cfg.train_size - len(train)
            take = indices[:remaining]
            train.extend(take)
            used_scenarios.add(scenario_id)
            continue
        if cfg.val_scenario_disjoint and scenario_id in used_scenarios:
            continue
        if len(val) < cfg.val_size:
            remaining = cfg.val_size - len(val)
            take = indices[:remaining]
            val.extend(take)
        if len(train) >= cfg.train_size and len(val) >= cfg.val_size:
            break

    if len(train) < cfg.train_size:
        raise ValueError(f"insufficient train rows: need {cfg.train_size}, found {len(train)}")
    if len(val) < cfg.val_size:
        raise ValueError(f"insufficient validation rows: need {cfg.val_size}, found {len(val)}")

    debug_source = list(train)
    rng.shuffle(debug_source)
    debug = debug_source[: min(cfg.debug_size, len(debug_source))]
    return SubsetIndices(train=sorted(train), val=sorted(val), debug=sorted(debug))


def build_subset_indices(dataset: TrajectoryDataset, cfg: SubsetConfig) -> SubsetIndices:
    if getattr(dataset, "mode", "") != "processed":
        raise ValueError("controlled generalisation experiment requires processed AV2 data")
    return _sample_rows_by_scenario(dataset, cfg)


def subset_metadata(dataset: TrajectoryDataset, indices: list[int], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for experiment_index, dataset_index in enumerate(indices):
        row = dataset.rows[dataset_index]
        out.append(
            {
                "split": split,
                "experiment_index": experiment_index,
                "dataset_index": dataset_index,
                "sample_id": row["sample_id"],
                "scenario_id": row["scenario_id"],
                "file_path": row["file_path"],
                "start_t": int(row["start_t"]),
                "output_shard": row["output_shard"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/small_modal_1500_500_seed7.json")
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--output-dir", default="outputs/controlled_generalisation/subset_builder")
    parser.add_argument("--bundle-dir", default=None, help="Optional directory to save preloaded train/val/debug row bundles")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.processed_root:
        config.processed_root = args.processed_root
    dataset = TrajectoryDataset(config.processed_root, "av2")
    subsets = build_subset_indices(dataset, config.subset)
    train_meta = subset_metadata(dataset, subsets.train, "train")
    val_meta = subset_metadata(dataset, subsets.val, "val")
    debug_meta = subset_metadata(dataset, subsets.debug, "debug")
    write_json(f"{args.output_dir}/train.json", train_meta)
    write_json(f"{args.output_dir}/val.json", val_meta)
    write_json(f"{args.output_dir}/debug.json", debug_meta)
    if args.bundle_dir:
        import torch

        bundle_dir = Path(args.bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        torch.save(preload_rows(dataset, train_meta), bundle_dir / "train_rows.pt")
        torch.save(preload_rows(dataset, val_meta), bundle_dir / "val_rows.pt")
        torch.save(preload_rows(dataset, debug_meta), bundle_dir / "debug_rows.pt")
        write_json(bundle_dir / "train.json", train_meta)
        write_json(bundle_dir / "val.json", val_meta)
        write_json(bundle_dir / "debug.json", debug_meta)


if __name__ == "__main__":
    main()
