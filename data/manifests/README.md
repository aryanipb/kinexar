# Data Manifests

This directory contains JSON manifest files that list Argoverse 2 scenario parquet files
used for training and validation.

## Files

| File | Records | Purpose |
|---|---:|---|
| `av2_val_manifest.json` | 24,988 | Validation set manifest (included) |
| `av2_manifest.json` | 599,724 | Full training manifest (not committed — too large) |
| `av2_manifest1.json` | 199,908 | Partial training manifest (not committed — too large) |

## Manifest Schema

Each line/record in a manifest JSON contains:

```json
{
  "scenario_id": "<uuid>",
  "file_path": "/path/to/scenario_<uuid>.parquet",
  "num_timesteps": 110,
  "start_t": 0
}
```

Validation manifest rows also include:

```json
{
  "history_steps": 50,
  "future_steps": 60,
  "window_steps": 110
}
```

## Generating Manifests

Use the preprocessing scripts to build manifests from a local AV2 dataset:

```bash
PYTHONPATH=src:. pixi run python scripts/preprocess_av2_v2.py \
  --manifest data/manifests/av2_manifest.json \
  --output-root data/processed/av2_full \
  --map-dir /path/to/maps
```

Download Argoverse 2 from: https://www.argoverse.org/av2.html
