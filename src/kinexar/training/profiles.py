from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE_ROOT = REPO_ROOT.parent / "datasetpredmodel"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "neurips"


@dataclass(frozen=True)
class TrainingProfile:
    family: str
    config_name: str
    dataset_dir_name: str
    output_subdir: str
    run_prefix: str
    train_size: int | None
    val_size: int | None
    debug_size: int
    seed: int


PROFILES: dict[str, TrainingProfile] = {
    "small": TrainingProfile(
        family="small",
        config_name="small_modal_1500_500_seed7.json",
        dataset_dir_name="small_modal_1500_500_seed7",
        output_subdir="small_modal",
        run_prefix="kinexar_small_modal",
        train_size=1500,
        val_size=500,
        debug_size=32,
        seed=7,
    ),
    "ablation": TrainingProfile(
        family="ablation",
        config_name="ablation_15000_2000_seed7.json",
        dataset_dir_name="ablation_15000_2000_seed7",
        output_subdir="ablation",
        run_prefix="kinexar_ablation",
        train_size=15000,
        val_size=2000,
        debug_size=32,
        seed=7,
    ),
    "full": TrainingProfile(
        family="full",
        config_name="full_train.json",
        dataset_dir_name="full_train",
        output_subdir="full",
        run_prefix="kinexar_full",
        train_size=None,
        val_size=None,
        debug_size=0,
        seed=7,
    ),
}

ALIASES = {
    "modal": "small",
    "small-modal": "small",
    "small_modal": "small",
    "oab": "ablation",
    "ofull": "full",
}


def resolve_profile(name: str) -> TrainingProfile:
    key = ALIASES.get(name, name)
    if key not in PROFILES:
        raise KeyError(f"unknown training family: {name}")
    return PROFILES[key]


def profile_config_path(profile: TrainingProfile) -> Path:
    return REPO_ROOT / "configs" / "experiments" / profile.config_name


def archive_root_from_env() -> Path:
    return Path(os.environ.get("KINEXAR_ARCHIVE_SOURCE_ROOT", str(DEFAULT_ARCHIVE_ROOT))).expanduser()


def output_root_for(profile: TrainingProfile) -> Path:
    return DEFAULT_OUTPUT_ROOT / profile.output_subdir


def dataset_root_for(base_root: str | Path, profile: TrainingProfile) -> Path:
    return Path(base_root).expanduser() / profile.dataset_dir_name


def processed_root_for(base_root: str | Path, profile: TrainingProfile) -> Path:
    return dataset_root_for(base_root, profile) / "full"
