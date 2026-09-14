# KINEXAR

**Kinematic Expert Factorization for Multimodal Trajectory Forecasting**

KINEXAR is a research codebase for **multi-modal future trajectory prediction** in autonomous-driving scenes using the [Argoverse 2](https://www.argoverse.org/av2.html) Motion Forecasting dataset.

The core idea: predict multiple plausible future paths for a focal agent by combining:
- A **spatial-temporal interaction encoder** (Mamba-style temporal blocks + geometric edge features)
- A **CVAE** with structured latent groups (*route*, *speed*, *interaction*, *style*)
- A bank of **physically motivated kinematic experts** (CV, CA, CTRA, Jerk variants)
- An optional **road-distance refinement** step

---

## Architecture at a Glance

```
Agent History (B, A, T, 5)  +  Road Distance Map (B, H, W)
         │                              │
         ▼                             │
 SpatialTemporalEncoder                │
  ├─ Mamba temporal blocks             │
  ├─ Geometric spatial edges           │
  └─ Cross-time flow interactions      │
         │                             │
         ▼                             │
  CVAE Posterior / Prior ◄─────── Future Target (train)
         │
         ▼ (latent z)
  Kinematic Expert Bank
  ├─ cv    – constant velocity
  ├─ ca    – constant acceleration
  ├─ ctra  – constant turn-rate & acceleration
  ├─ jerk  – jerk magnitude
  ├─ jerk_beta  – jerk + lateral
  └─ jerk_turn  – jerk + turn-rate
         │
         ▼
  Mode Scorer  ─►  Best-mode selection  ─►  Final Trajectory
```

**Parameter count (default config):** ~1.82M trainable parameters (`din=5`, `dim=128`, `lat=128`, 6 experts, full encoder, transformer coefficient decoder).

| Block | Parameters |
|---|---:|
| SpatialTemporalEncoder | 202,628 |
| Posterior CVAE Encoder | 598,528 |
| Prior | 197,376 |
| Structured MoE Decoder | 819,215 |

---

## Repository Layout

```
kinexar/
├── src/kinexar/            # Main namespaced Python package
│   ├── models/             # TrajectoryModel and all sub-components
│   │   ├── model.py        # Public facade: from kinexar.models.model import TrajectoryModel
│   │   ├── trajectory_model_impl.py
│   │   ├── spatial_temporal_encoder.py
│   │   ├── cvae/           # Encoder, Prior, MoE decoder
│   │   └── ...
│   ├── dataset/            # Data loading and preprocessing
│   │   ├── data/           # TrajectoryDataset, AV2 parquet loader
│   │   └── preprocessing/  # Graph builder, drivable area, Frenet, CTRA
│   ├── training/           # Full training and evaluation pipeline
│   │   ├── train.py        # Training loop (run_training_experiment)
│   │   ├── evaluate.py     # Evaluation entry point
│   │   ├── losses.py       # MoE / CVAE / physics / CFCP loss terms
│   │   ├── metrics.py      # ADE, FDE, minADE, miss-rate, expert usage
│   │   ├── batching.py     # Collation, padding, scale normalization
│   │   └── config/         # Dataclass configs and ablation variants
│   ├── cloud/              # GCP workflow helpers
│   └── utils/              # Utilities
├── scripts/                # CLI wrappers: preprocessing, training, GCP
├── configs/                # Experiment, ablation, and runtime JSON configs
├── data/                   # Lightweight manifests (raw data is external)
├── docker/                 # Docker image and entrypoint
├── docs/                   # Technical notes: model, preprocessing, training
└── pixi.toml               # Environment definition
```

---

## Quick Start

### 1. Install Environment

```bash
# Install pixi: https://prefix.dev/
curl -fsSL https://pixi.sh/install.sh | bash

# Create the environment (first run fetches all dependencies)
pixi install
```

### 2. Verify Installation

```bash
PYTHONPATH=src:. pixi run python -c "from kinexar.models.model import TrajectoryModel; print('OK')"
PYTHONPATH=src:. pixi run python -c "from kinexar.dataset.data.dataset import TrajectoryDataset; print('OK')"
```

### 3. Prepare Data

KINEXAR expects [Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html) scenario parquet files and their corresponding map JSON files.

```bash
# Set map directory
export KINEXAR_MAP_DIR=/path/to/av2/maps

# Build a data manifest (points to parquet files on disk)
python scripts/harvest_local_maps.py --manifest data/manifests/av2_val_manifest.json --map-dir "$KINEXAR_MAP_DIR"

# Preprocess raw parquet → shard tensors (v2 pipeline)
PYTHONPATH=src:. pixi run python scripts/preprocess_av2_v2.py \
  --manifest data/manifests/av2_manifest.json \
  --output-root data/processed/av2_full \
  --map-dir "$KINEXAR_MAP_DIR"
```

### 4. Train

```bash
PYTHONPATH=src:. pixi run python src/kinexar/training/train.py \
  --config configs/experiments/small_modal_1500_500_seed7.json \
  --device cuda
```

Or with the full training config:

```bash
PYTHONPATH=src:. pixi run python src/kinexar/training/train.py \
  --config configs/experiments/full_train.json \
  --device cuda \
  --epochs 64
```

### 5. Evaluate

```bash
PYTHONPATH=src:. pixi run python src/kinexar/training/evaluate.py \
  --config configs/experiments/full_train.json \
  --checkpoint outputs/neurips/full/kinexar_baseline/checkpoints/modelckpt.pth \
  --split val
```

---

## Model Usage

```python
from kinexar.models.model import TrajectoryModel

# Instantiate the full KINEXAR model
model = TrajectoryModel(
    din=5,              # Input feature dimension (x, y, heading, vx, vy)
    dim=128,            # Hidden dimension
    lat=128,            # Latent dimension
    future_steps=60,    # Prediction horizon
    decoder_family="structured_bank",
    expert_count=6,
)

# Forward pass (training)
out = model(
    feat,           # (B, A, T, 5) — padded agent history
    history_mask,   # (B, A, T)    — valid timestep mask
    agent_mask,     # (B, A)       — valid agent mask
    scale,          # (B,)         — per-sample coordinate scale
    future=future,  # (B, T_fut, 2) — ground-truth future (training only)
    mode="train",
)

# Key outputs
predictions = out["y_hat"]        # (B, K, T_fut, 2) absolute predictions
mode_scores = out["score"]        # (B, K) expert confidence scores
z = out["z"]                      # (B, lat) sampled latent
```

---

## Loss System

The loss is computed by `kinexar.training.losses.moe(...)`. Components:

| Term | Description |
|---|---|
| **Reconstruction** | Per-expert position loss (L2) |
| **Physics** | Expert-specific kinematic consistency (CV, CA, CTRA, jerk) |
| **Velocity** | L1 velocity alignment between experts and ground truth |
| **KL** | Gaussian posterior vs. prior with linear annealing and free-bits |
| **Mode CE** | Cross-entropy on expert selected by minimum ADE |
| **CFCP** | Counterfactual controlled-prediction terms (ownership, leakage, stability) |

---

## Ablation Variants

Controlled via `--variant` flag or `ablation.variant` in config JSON:

| Variant | Description |
|---|---|
| `full_kinexar` / `p0` | Full model (default) |
| `p1` | No CFCP regularization |
| `p2` – `p5` | CFCP component ablations |
| `l1` | Flat (unstructured) latent |
| `l2` | Structured latent without CFCP |
| `d1` | MLP coefficient decoder (vs. transformer) |
| `no_cvae` | No latent sampling |
| `weak_encoder` | Temporal-only encoder (no spatial/cross edges) |
| `homogeneous_6head` | Replace physics experts with 6 identical MLP heads |

---

## Dependencies

Managed via [Pixi](https://pixi.sh/) with `conda-forge` channel:

| Package | Constraint | Purpose |
|---|---|---|
| Python | `3.11.*` | Runtime |
| PyTorch | `>=2.4` | Neural network training |
| PyArrow | `>=18,<20` | Argoverse 2 parquet reading |
| SciPy | `>=1.14,<1.17` | Distance transforms, geometry |
| Pillow | `>=10,<13` | Map rasterization and visualization |

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `KINEXAR_MAP_DIR` | Directory containing `log_map_archive_*.json` files |
| `KINEXAR_VAL_PROCESSED_ROOT` | Override for validation processed-shard root |
| `KINEXAR_REQUIRE_MAP_CACHE` | Fail if maps are not pre-cached |
| `KINEXAR_NO_PARQUET` | Disable raw parquet access |

---

## Citation

If you use KINEXAR in your research, please cite:

```bibtex
@misc{kinexar2026,
  title  = {KINEXAR: Kinematic Expert Factorization for Multimodal Trajectory Forecasting},
  year   = {2026},
  note   = {https://github.com/aryanipb/kinexar}
}
```

---

## License

See [LICENSE](LICENSE) for details.
