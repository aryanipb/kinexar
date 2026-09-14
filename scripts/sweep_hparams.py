from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def _csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _csv_float_list(raw: str) -> list[float]:
    return [float(item) for item in _csv_list(raw)]


def _csv_int_list(raw: str) -> list[int]:
    return [int(item) for item in _csv_list(raw)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["small", "ablation", "full"])
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--map-dir", default=None)
    parser.add_argument("--output-root", default="runs/sweeps")
    parser.add_argument("--base-run-name", default="sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rates", default="7e-4")
    parser.add_argument("--weight-decays", default="1e-4")
    parser.add_argument("--batch-sizes", default="16")
    parser.add_argument("--warmup-steps-grid", default="15000")
    parser.add_argument("--kl-free-bits-grid", default="0.25")
    parser.add_argument("--cfcp-ownership-grid", default="0.10")
    parser.add_argument("--cfcp-leakage-grid", default="0.05")
    parser.add_argument("--cfcp-stability-grid", default="0.02")
    parser.add_argument("--compile-modes", default="off")
    parser.add_argument("--latent-group-dims", default="16")
    parser.add_argument("--decoder-layers-grid", default="3")
    parser.add_argument("--decoder-heads-grid", default="8")
    parser.add_argument("--road-refinement-grid", default="true")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    combos = itertools.product(
        _csv_float_list(args.learning_rates),
        _csv_float_list(args.weight_decays),
        _csv_int_list(args.batch_sizes),
        _csv_int_list(args.warmup_steps_grid),
        _csv_float_list(args.kl_free_bits_grid),
        _csv_float_list(args.cfcp_ownership_grid),
        _csv_float_list(args.cfcp_leakage_grid),
        _csv_float_list(args.cfcp_stability_grid),
        _csv_list(args.compile_modes),
        _csv_int_list(args.latent_group_dims),
        _csv_int_list(args.decoder_layers_grid),
        _csv_int_list(args.decoder_heads_grid),
        _csv_list(args.road_refinement_grid),
    )
    train_script = REPO_ROOT / "scripts" / "train_gcp.py"
    for index, combo in enumerate(combos, start=1):
        (
            lr,
            wd,
            batch_size,
            warmup_steps,
            free_bits,
            cfcp_ownership,
            cfcp_leakage,
            cfcp_stability,
            compile_mode,
            latent_group_dim,
            decoder_layers,
            decoder_heads,
            road_refinement,
        ) = combo
        run_name = f"{args.base_run_name}_{index:03d}"
        cmd = [
            sys.executable,
            str(train_script),
            "--family",
            args.family,
            "--processed-root",
            args.processed_root,
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--num-workers",
            str(args.num_workers),
            "--learning-rate",
            str(lr),
            "--weight-decay",
            str(wd),
            "--batch-size",
            str(batch_size),
            "--warmup-steps",
            str(warmup_steps),
            "--kl-free-bits",
            str(free_bits),
            "--cfcp-loss-weight-ownership",
            str(cfcp_ownership),
            "--cfcp-loss-weight-leakage",
            str(cfcp_leakage),
            "--cfcp-loss-weight-stability",
            str(cfcp_stability),
            "--compile-mode",
            str(compile_mode),
            "--latent-group-dim",
            str(latent_group_dim),
            "--decoder-layers",
            str(decoder_layers),
            "--decoder-heads",
            str(decoder_heads),
            "--road-refinement",
            str(road_refinement),
        ]
        if args.map_dir:
            cmd.extend(["--map-dir", args.map_dir])
        subprocess.run(cmd, check=True)
        metrics_path = output_root / run_name / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = payload.get("history", [])
        best_val_min6ade = payload.get("best_val_min6ADE")
        best_val_min6fde = payload.get("best_val_min6FDE")
        summary_rows.append(
            {
                "run_name": run_name,
                "learning_rate": lr,
                "weight_decay": wd,
                "batch_size": batch_size,
                "warmup_steps": warmup_steps,
                "kl_free_bits": free_bits,
                "cfcp_ownership": cfcp_ownership,
                "cfcp_leakage": cfcp_leakage,
                "cfcp_stability": cfcp_stability,
                "compile_mode": compile_mode,
                "latent_group_dim": latent_group_dim,
                "decoder_layers": decoder_layers,
                "decoder_heads": decoder_heads,
                "road_refinement": road_refinement,
                "best_val_min6ADE": best_val_min6ade,
                "best_val_min6FDE": best_val_min6fde,
                "epochs_ran": len(history),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            float("inf") if row["best_val_min6ADE"] is None else float(row["best_val_min6ADE"]),
            float("inf") if row["best_val_min6FDE"] is None else float(row["best_val_min6FDE"]),
        )
    )
    json_path = output_root / "sweep_summary.json"
    csv_path = output_root / "sweep_summary.csv"
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    if summary_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
