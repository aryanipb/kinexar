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

from kinexar.cloud.gcp import AblationRequest
from kinexar.cloud.gcp import DEFAULT_GCP_ZONE
from kinexar.cloud.gcp import GcpWorkflowError
from kinexar.cloud.gcp import TrainingRuntime
from kinexar.cloud.gcp import discover_runtime_config
from kinexar.cloud.gcp import run_ablation_workflow
from kinexar.cloud.gcp import run_docker_ablation_workflow
from kinexar.cloud.gcp import run_docker_full_workflow
from kinexar.cloud.gcp import run_docker_image_build_workflow
from kinexar.cloud.gcp import run_docker_validation_workflow
from kinexar.cloud.gcp import run_full_workflow
from kinexar.cloud.gcp import run_upload_workflow


def _parse_on_off(value: str) -> bool:
    raw = str(value).strip().lower()
    if raw in {"on", "true", "1", "yes"}:
        return True
    if raw in {"off", "false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected on/off, got {value!r}")


def _parse_seeds(value: str) -> tuple[int, ...]:
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one seed")
    return tuple(int(item) for item in parts)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=None)
    parser.add_argument("--zone", default=DEFAULT_GCP_ZONE)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--archive-root", default=None)
    parser.add_argument("--data-disk-name", default="kinexar-neurips-data")
    parser.add_argument("--data-disk-type", default="pd-standard")
    parser.add_argument("--data-disk-size-gb", type=int, default=1200)
    parser.add_argument("--data-disk-iops", type=int, default=None)
    parser.add_argument("--data-disk-throughput-mib", type=int, default=None)
    parser.add_argument("--archive-chunk-size-gb", type=int, default=4)
    parser.add_argument("--upload-machine-type", default="auto")
    parser.add_argument("--training-machine-type", default="g2-standard-4")
    parser.add_argument("--training-accelerator-type", default=None)
    parser.add_argument("--training-accelerator-count", type=int, default=0)
    parser.add_argument("--upload-max-run-duration", default="24h")
    parser.add_argument("--full-max-run-duration", default="72h")
    parser.add_argument("--ablation-max-run-duration", default="48h")
    parser.add_argument("--service-account", default=None)


def _runtime_from_args(args: argparse.Namespace) -> TrainingRuntime:
    return TrainingRuntime(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        vram_target_gb=float(args.vram_target_gb),
        vram_cap_gb=float(args.vram_cap_gb),
        epochs=None if args.epochs is None else int(args.epochs),
        skip_validation=bool(getattr(args, "skip_validation", False)),
    )


def _config_from_args(args: argparse.Namespace):
    return discover_runtime_config(
        repo_root=args.repo_root,
        archive_root=args.archive_root,
        project=args.project,
        zone=args.zone,
        data_disk_name=args.data_disk_name,
        data_disk_type=args.data_disk_type,
        data_disk_size_gb=args.data_disk_size_gb,
        data_disk_iops=args.data_disk_iops,
        data_disk_throughput_mib=args.data_disk_throughput_mib,
        archive_chunk_size_gb=args.archive_chunk_size_gb,
        upload_machine_type=args.upload_machine_type,
        training_machine_type=args.training_machine_type,
        training_accelerator_type=args.training_accelerator_type,
        training_accelerator_count=args.training_accelerator_count,
        upload_max_run_duration=args.upload_max_run_duration,
        full_max_run_duration=args.full_max_run_duration,
        ablation_max_run_duration=args.ablation_max_run_duration,
        service_account=args.service_account,
    )


def _add_runtime_args(parser: argparse.ArgumentParser, *, batch_size_default: int) -> None:
    parser.add_argument("--batch-size", type=int, default=batch_size_default)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--vram-target-gb", type=float, default=20.0)
    parser.add_argument("--vram-cap-gb", type=float, default=22.0)
    parser.add_argument("--skip-validation", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    upload = sub.add_parser("upload")
    _add_common(upload)
    upload.add_argument("--force", action="store_true")

    full = sub.add_parser("full")
    _add_common(full)
    full.add_argument("--run-name", default=None)
    _add_runtime_args(full, batch_size_default=32)

    docker_images = sub.add_parser("docker-images")
    _add_common(docker_images)

    docker_full = sub.add_parser("docker-full")
    _add_common(docker_full)
    docker_full.add_argument("--run-name", default=None)
    docker_full.add_argument("--map-dir", default=None)
    docker_full.add_argument("--suite", default=None)
    docker_full.add_argument("--ablation", action="store_true")
    _add_runtime_args(docker_full, batch_size_default=16)

    docker_validate = sub.add_parser("docker-validate")
    _add_common(docker_validate)
    docker_validate.add_argument("--family", required=True, choices=["ablation", "full"])
    docker_validate.add_argument("--checkpoint", required=True)
    docker_validate.add_argument("--split", default="val", choices=["train", "val", "debug"])
    docker_validate.add_argument("--processed-root", default=None)
    docker_validate.add_argument("--output-root", default=None)
    docker_validate.add_argument("--run-name", default=None)
    docker_validate.add_argument("--map-dir", default=None)
    docker_validate.add_argument("--visualize", action="store_true")
    _add_runtime_args(docker_validate, batch_size_default=16)

    val_maps_upload = sub.add_parser("val-maps-upload")
    _add_common(val_maps_upload)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = _config_from_args(args)
        if args.command == "upload":
            result = run_upload_workflow(config, force=bool(args.force))
        elif args.command == "val-maps-upload":
            from kinexar.cloud.gcp import run_val_maps_upload_workflow
            result = run_val_maps_upload_workflow(config)
        elif args.command == "full":
            result = run_full_workflow(
                config,
                runtime=_runtime_from_args(args),
                run_name=args.run_name,
            )
        elif args.command == "docker-images":
            result = run_docker_image_build_workflow(config)
        elif args.command == "docker-full":
            result = run_docker_full_workflow(
                config,
                runtime=_runtime_from_args(args),
                run_name=args.run_name,
                map_dir=args.map_dir,
                suite=args.suite,
            )
        elif args.command == "docker-validate":
            result = run_docker_validation_workflow(
                config,
                visualize=bool(args.visualize),
                family=args.family,
                checkpoint_path=args.checkpoint,
                split=args.split,
                processed_root=args.processed_root,
                output_root=args.output_root,
                run_name=args.run_name,
                runtime=_runtime_from_args(args),
                map_dir=args.map_dir,
            )
        else:
            raise AssertionError(f"unsupported command: {args.command}")
    except GcpWorkflowError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
