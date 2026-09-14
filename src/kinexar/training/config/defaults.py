from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
from typing import Any


DEFAULT_STRUCTURED_EXPERTS = (
    "cv",
    "ca",
    "ctra",
    "jerk",
    "jerk_beta",
    "jerk_turn",
)


@dataclass
class AblationConfig:
    variant: str = "full_kinexar"
    decoder_family: str = "structured_bank"
    expert_count: int = 6
    num_experts: int = 6
    expert_names: list[str] = field(default_factory=lambda: list(DEFAULT_STRUCTURED_EXPERTS))
    use_cvae_latent: bool = True
    use_physics_losses: bool = True
    shared_decoder: bool = False
    encoder_mode: str = "full"
    no_latent: bool = False
    single_expert: bool = False
    weak_encoder: bool = False
    use_residual_refinement: bool = False
    refinement_lambda: float = 0.35

    identical_kinematic_name: str = "ca"
    one_generator_name: str = "ctra"
    use_transformer_coeff_decoder: bool = True
    use_mlp_coeff_decoder: bool = False
    use_structured_latent: bool = True
    use_flat_latent: bool = False
    use_query_self_attention: bool = True
    use_true_jerk_turn: bool = True
    use_expert_latent_proj: bool = True
    use_explicit_map_tokens: bool = True
    use_learned_coeff_scales: bool = True
    use_strong_mode_scorer: bool = True
    latent_group_dim: int = 16
    decoder_layers: int = 3
    decoder_heads: int = 8
    free_bits: float = 0.10
    warmup_steps: int = 15000
    use_cfcp: bool = True
    use_cfcp_ownership: bool = True
    use_cfcp_leakage: bool = True
    use_cfcp_stability: bool = True
    cfcp_every_n_steps: int = 2
    cfcp_sigma: float = 0.15
    lambda_ownership: float = 0.10
    lambda_leakage: float = 0.10
    lambda_stability: float = 0.02
    compute_cri_metrics: bool = True
    save_ablation_artifacts: bool = True


@dataclass
class SubsetConfig:
    train_size: int = 256
    val_size: int = 64
    debug_size: int = 32
    seed: int = 7
    val_scenario_disjoint: bool = True


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 64
    log_train_metrics: bool = True
    learning_rate: float = 7e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    log_every_steps: int = 1
    save_every_epochs: int = 1
    early_stop_patience_epochs: int = 10
    device: str = "cuda"
    vram_target_gb: float = 40.0
    vram_cap_gb: float = 40.0
    use_compile: bool = False
    compile_mode: str = "off"
    use_tf32: bool = True
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    checkpoint_every_epochs: int = 1
    validate_every_epochs: int = 1


@dataclass
class EvalConfig:
    visualise_samples: int = 8
    report_filename: str = "summary_report.json"
    history_filename: str = "history.json"
    history_csv_filename: str = "history.csv"
    val_metrics_filename: str = "validation_metrics.json"
    val_samples_filename: str = "validation_samples.csv"


@dataclass
class PassFailConfig:
    min_ade_rel_improvement: float = 0.05
    min_expert_delta: float = 0.05
    max_expert_usage_ratio: float = 0.90
    max_mean_offroad: float = 0.50
    max_final_offroad: float = 1.50


@dataclass
class ModalConfig:
    app_name: str = "kinexar-neurips-modal"
    data_volume_name: str = "kinexar-data"
    output_volume_name: str = "kinexar-neurips"
    gpu: str = "A10G"
    cpu: int = 6
    memory_mb: int = 10240
    timeout_seconds: int = 12 * 60 * 60
    remote_processed_root: str = "/data/processed/av2_full"
    remote_map_dir: str = "/data/maps"
    remote_output_root: str = "/outputs/controlled_generalisation"


@dataclass
class ExperimentConfig:
    experiment_name: str = "kinexar_neurips"
    run_name: str = "kinexar_baseline"
    processed_root: str = "data/processed/av2_full"
    map_dir: str = "data/maps"
    output_root: str = "outputs/neurips/default"
    seed: int = 7
    subset: SubsetConfig = field(default_factory=SubsetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    pass_fail: PassFailConfig = field(default_factory=PassFailConfig)
    modal: ModalConfig = field(default_factory=ModalConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    @property
    def run_dir(self) -> str:
        return str(Path(self.output_root) / self.run_name)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _structured_expert_names(count: int) -> list[str]:
    if count < 1 or count > len(DEFAULT_STRUCTURED_EXPERTS):
        raise ValueError(f"num_experts must be in [1, {len(DEFAULT_STRUCTURED_EXPERTS)}], got {count}")
    return list(DEFAULT_STRUCTURED_EXPERTS[:count])


def _apply_final_paper_defaults(raw: AblationConfig) -> None:
    raw.use_cvae_latent = True
    raw.use_structured_latent = True
    raw.use_flat_latent = False
    raw.use_transformer_coeff_decoder = True
    raw.use_mlp_coeff_decoder = False
    raw.use_cfcp = True
    raw.use_cfcp_ownership = True
    raw.use_cfcp_leakage = True
    raw.use_cfcp_stability = True
    raw.use_physics_losses = True
    raw.shared_decoder = False
    raw.encoder_mode = "full"
    raw.decoder_family = "structured_bank"
    raw.expert_names = list(DEFAULT_STRUCTURED_EXPERTS)
    raw.expert_count = len(raw.expert_names)
    raw.num_experts = raw.expert_count


def _apply_paper_variant(raw: AblationConfig, variant: str) -> bool:
    key = variant.lower()
    if key in {"p0", "p0_full_final_model", "p6", "p6_full_cfcp", "l3", "l3_structured_latent_full_cfcp", "d2", "d2_transformer_coefficient_decoder"}:
        _apply_final_paper_defaults(raw)
        return True
    if key in {"p1", "p1_no_cfcp"}:
        _apply_final_paper_defaults(raw)
        raw.use_cfcp = False
        return True
    if key in {"p2", "p2_ownership_only"}:
        _apply_final_paper_defaults(raw)
        raw.use_cfcp = True
        raw.use_cfcp_ownership = True
        raw.use_cfcp_leakage = False
        raw.use_cfcp_stability = False
        return True
    if key in {"p3", "p3_leakage_only"}:
        _apply_final_paper_defaults(raw)
        raw.use_cfcp = True
        raw.use_cfcp_ownership = False
        raw.use_cfcp_leakage = True
        raw.use_cfcp_stability = False
        return True
    if key in {"p4", "p4_stability_only"}:
        _apply_final_paper_defaults(raw)
        raw.use_cfcp = True
        raw.use_cfcp_ownership = False
        raw.use_cfcp_leakage = False
        raw.use_cfcp_stability = True
        return True
    if key in {"p5", "p5_ownership_leakage"}:
        _apply_final_paper_defaults(raw)
        raw.use_cfcp = True
        raw.use_cfcp_ownership = True
        raw.use_cfcp_leakage = True
        raw.use_cfcp_stability = False
        return True
    if key in {"l1", "l1_flat_latent"}:
        _apply_final_paper_defaults(raw)
        raw.use_structured_latent = False
        raw.use_flat_latent = True
        return True
    if key in {"l2", "l2_structured_latent_no_cfcp"}:
        _apply_final_paper_defaults(raw)
        raw.use_cfcp = False
        return True
    if key in {"d1", "d1_mlp_coefficient_decoder"}:
        _apply_final_paper_defaults(raw)
        raw.use_transformer_coeff_decoder = False
        raw.use_mlp_coeff_decoder = True
        return True
    return False


def resolve_ablation_config(config: ExperimentConfig) -> AblationConfig:
    raw = AblationConfig(**asdict(config.ablation))
    variant = raw.variant.strip() or "full_kinexar"
    requested_num_experts = int(config.ablation.num_experts)
    raw.variant = variant
    if _apply_paper_variant(raw, variant):
        pass
    elif variant == "full_kinexar":
        raw.decoder_family = "structured_bank"
        raw.expert_names = list(DEFAULT_STRUCTURED_EXPERTS)
        raw.expert_count = len(raw.expert_names)
        raw.num_experts = raw.expert_count
        raw.use_cvae_latent = True
        raw.use_physics_losses = True
        raw.shared_decoder = False
        raw.encoder_mode = "full"
    elif variant == "homogeneous_6head":
        raw.decoder_family = "homogeneous_mlp"
        raw.expert_count = 6
        raw.num_experts = raw.expert_count
        raw.expert_names = [f"hom_head_{idx + 1}" for idx in range(raw.expert_count)]
        raw.use_cvae_latent = True
        raw.use_physics_losses = False
        raw.shared_decoder = False
        raw.encoder_mode = "full"
    elif variant == "core_transformer_cvae_only":
        raw.decoder_family = "homogeneous_transformer"
        raw.expert_count = 6
        raw.num_experts = raw.expert_count
        raw.expert_names = [f"tf_head_{idx + 1}" for idx in range(raw.expert_count)]
        raw.use_cvae_latent = True
        raw.use_physics_losses = False
        raw.shared_decoder = False
        raw.encoder_mode = "full"
    elif variant == "no_physics_experts":
        raw.decoder_family = "homogeneous_transformer"
        raw.expert_count = 6
        raw.num_experts = raw.expert_count
        raw.expert_names = [f"tf_head_{idx + 1}" for idx in range(raw.expert_count)]
        raw.use_cvae_latent = True
        raw.use_physics_losses = False
        raw.shared_decoder = False
        raw.encoder_mode = "full"
    elif variant == "single_shared_decoder":
        raw.decoder_family = "homogeneous_mlp"
        raw.expert_count = 1
        raw.num_experts = raw.expert_count
        raw.expert_names = ["shared_head"]
        raw.use_cvae_latent = True
        raw.use_physics_losses = False
        raw.shared_decoder = True
        raw.encoder_mode = "full"
    elif variant == "identical_kinematic_heads":
        raw.decoder_family = "structured_bank"
        raw.expert_names = [raw.identical_kinematic_name] * 6
        raw.expert_count = len(raw.expert_names)
        raw.num_experts = raw.expert_count
        raw.use_cvae_latent = True
        raw.use_physics_losses = True
        raw.shared_decoder = False
        raw.encoder_mode = "full"
    elif variant == "three_generator":
        raw.decoder_family = "structured_bank"
        raw.expert_names = list(DEFAULT_STRUCTURED_EXPERTS[:3])
        raw.expert_count = len(raw.expert_names)
        raw.num_experts = raw.expert_count
        raw.use_cvae_latent = True
        raw.use_physics_losses = True
        raw.shared_decoder = False
        raw.encoder_mode = "full"
    elif variant == "one_generator":
        raw.decoder_family = "structured_bank"
        raw.expert_names = [raw.one_generator_name]
        raw.expert_count = 1
        raw.num_experts = raw.expert_count
        raw.use_cvae_latent = True
        raw.use_physics_losses = True
        raw.shared_decoder = False
        raw.encoder_mode = "full"
    elif variant in {"no_cvae", "no_latent"}:
        raw.decoder_family = "structured_bank"
        raw.expert_names = list(DEFAULT_STRUCTURED_EXPERTS)
        raw.expert_count = len(raw.expert_names)
        raw.num_experts = raw.expert_count
        raw.use_cvae_latent = False
        raw.use_physics_losses = True
        raw.shared_decoder = False
        raw.encoder_mode = "full"
    elif variant == "no_physics_loss":
        raw.decoder_family = "structured_bank"
        raw.expert_names = list(DEFAULT_STRUCTURED_EXPERTS)
        raw.expert_count = len(raw.expert_names)
        raw.num_experts = raw.expert_count
        raw.use_cvae_latent = True
        raw.use_physics_losses = False
        raw.shared_decoder = False
        raw.encoder_mode = "full"
    elif variant in {"weak_encoder", "single_expert"}:
        raw.decoder_family = "structured_bank"
        raw.expert_names = list(DEFAULT_STRUCTURED_EXPERTS)
        raw.expert_count = len(raw.expert_names)
        raw.num_experts = raw.expert_count
        raw.use_cvae_latent = True
        raw.use_physics_losses = True
        raw.shared_decoder = False
        raw.encoder_mode = "full"
        if variant == "weak_encoder":
            raw.weak_encoder = True
        if variant == "single_expert":
            raw.single_expert = True
    else:
        raise ValueError(f"unknown ablation variant: {variant}")
    if requested_num_experts != 6 or variant not in {"three_generator", "one_generator", "single_shared_decoder"}:
        raw.num_experts = requested_num_experts
    raw.no_latent = bool(raw.no_latent)
    raw.single_expert = bool(raw.single_expert)
    raw.weak_encoder = bool(raw.weak_encoder)
    if raw.no_latent:
        raw.use_cvae_latent = False
    if raw.weak_encoder:
        raw.encoder_mode = "weak"
    else:
        raw.encoder_mode = "full"
    if raw.use_flat_latent and raw.use_structured_latent:
        raise ValueError("use_flat_latent and use_structured_latent are mutually exclusive")
    if raw.use_mlp_coeff_decoder and raw.use_transformer_coeff_decoder:
        raise ValueError("use_mlp_coeff_decoder and use_transformer_coeff_decoder are mutually exclusive")
    if raw.use_flat_latent and not raw.use_cvae_latent:
        raise ValueError("flat latent ablation requires use_cvae_latent=True")
    if raw.use_mlp_coeff_decoder and raw.decoder_family != "structured_bank":
        raise ValueError("mlp coefficient decoder is only supported for structured_bank decoder family")
    if not raw.use_cfcp:
        raw.use_cfcp_ownership = False
        raw.use_cfcp_leakage = False
        raw.use_cfcp_stability = False
    if raw.use_flat_latent:
        raw.use_structured_latent = False
    if raw.use_structured_latent:
        raw.use_flat_latent = False
    if raw.use_mlp_coeff_decoder:
        raw.use_transformer_coeff_decoder = False
    if raw.use_transformer_coeff_decoder:
        raw.use_mlp_coeff_decoder = False
    if raw.decoder_family == "structured_bank":
        requested_experts = int(raw.num_experts)
        if raw.single_expert:
            raw.expert_names = [raw.one_generator_name]
        else:
            if len(set(raw.expert_names)) == 1 and len(raw.expert_names) > 1:
                raw.expert_names = [raw.identical_kinematic_name] * requested_experts
            else:
                raw.expert_names = _structured_expert_names(requested_experts)
        raw.expert_count = len(raw.expert_names)
        raw.num_experts = raw.expert_count
    else:
        raw.expert_count = max(1, int(raw.num_experts))
        raw.num_experts = raw.expert_count
        if raw.shared_decoder:
            raw.expert_count = 1
            raw.num_experts = 1
            raw.expert_names = ["shared_head"]
        elif raw.decoder_family == "homogeneous_transformer":
            raw.expert_names = [f"tf_head_{idx + 1}" for idx in range(raw.expert_count)]
        else:
            raw.expert_names = [f"hom_head_{idx + 1}" for idx in range(raw.expert_count)]
    if raw.decoder_family == "structured_bank":
        raw.expert_count = len(raw.expert_names)
    return raw


def trajectory_model_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    ablation = resolve_ablation_config(config)
    return {
        "lat": 4 * int(ablation.latent_group_dim) if bool(ablation.use_flat_latent) else 128,
        "decoder_family": ablation.decoder_family,
        "expert_names": list(ablation.expert_names),
        "expert_count": int(ablation.expert_count),
        "use_cvae_latent": bool(ablation.use_cvae_latent),
        "shared_decoder": bool(ablation.shared_decoder),
        "encoder_mode": str(ablation.encoder_mode),
        "use_transformer_coeff_decoder": bool(ablation.use_transformer_coeff_decoder),
        "use_mlp_coeff_decoder": bool(ablation.use_mlp_coeff_decoder),
        "use_structured_latent": bool(ablation.use_structured_latent),
        "use_flat_latent": bool(ablation.use_flat_latent),
        "use_residual_refinement": bool(ablation.use_residual_refinement),
        "refinement_lambda": float(ablation.refinement_lambda),
        "latent_group_dim": int(ablation.latent_group_dim),
        "decoder_layers": int(ablation.decoder_layers),
        "decoder_heads": int(ablation.decoder_heads),
        "use_cfcp": bool(ablation.use_cfcp),
        "use_cfcp_ownership": bool(ablation.use_cfcp_ownership),
        "use_cfcp_leakage": bool(ablation.use_cfcp_leakage),
        "use_cfcp_stability": bool(ablation.use_cfcp_stability),
        "cfcp_every_n_steps": int(ablation.cfcp_every_n_steps),
        "cfcp_sigma": float(ablation.cfcp_sigma),
    }


def config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    merged = _merge_dict(config_to_dict(default_config()), payload)
    train_cfg = dict(merged["train"])
    train_cfg.pop("use_ar", None)
    return ExperimentConfig(
        experiment_name=merged["experiment_name"],
        run_name=merged["run_name"],
        processed_root=merged["processed_root"],
        map_dir=merged["map_dir"],
        output_root=merged["output_root"],
        seed=int(merged["seed"]),
        subset=SubsetConfig(**merged["subset"]),
        train=TrainConfig(**train_cfg),
        evaluation=EvalConfig(**merged["evaluation"]),
        pass_fail=PassFailConfig(**merged["pass_fail"]),
        modal=ModalConfig(**merged["modal"]),
        ablation=AblationConfig(**merged.get("ablation", {})),
    )


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return default_config()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return config_from_dict(payload)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(config_to_dict(config), handle, indent=2, sort_keys=True)
