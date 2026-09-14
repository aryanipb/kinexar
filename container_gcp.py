from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import math
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tarfile
import tempfile
import textwrap
import time
from typing import Any
from typing import Iterable
from typing import Sequence

from kinexar.dataset.data.bundle_io import FULL_ARCHIVE
from kinexar.dataset.data.bundle_io import PARTS_ARCHIVE
from kinexar.dataset.data.bundle_io import verify_archives
from kinexar.training.profiles import archive_root_from_env
from kinexar.training.profiles import resolve_profile


REPO_BUNDLE_TOP_LEVEL = (
    ".dockerignore",
    ".gitignore",
    "README.md",
    "configs",
    "data",
    "docker",
    "docs",
    "pixi.lock",
    "pixi.toml",
    "scripts",
    "src",
    "tests",
)

DEFAULT_GCP_ZONE = "asia-south1-c"
DEFAULT_GCLOUD_CANDIDATES = (
    "gcloud",
    str(Path.home() / "google-cloud-sdk" / "bin" / "gcloud"),
    "/opt/google-cloud-cli/bin/gcloud",
)
G2_UNSUPPORTED_HYPERDISKS = {"hyperdisk-balanced"}
HYPERDISK_ML_SUPPORTED_SERIES = {
    "a2",
    "a3",
    "a4",
    "c3",
    "c4",
    "g2",
    "g4",
    "n4",
}
DOCKER_IMAGE_FAMILIES = ("full",)


class GcpWorkflowError(RuntimeError):
    pass


class GcpResourceBusyError(GcpWorkflowError):
    pass


def _progress(message: str) -> None:
    print(f"[gcp] {message}", flush=True)


def _find_gcloud_binary() -> str:
    for candidate in DEFAULT_GCLOUD_CANDIDATES:
        if "/" in candidate:
            path = Path(candidate).expanduser()
            if path.exists():
                return str(path)
        else:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
    raise GcpWorkflowError("gcloud binary not found; install Google Cloud CLI or add it to PATH")


def _machine_series(machine_type: str) -> str:
    token = machine_type.split("-", 1)[0].strip().lower()
    if not token:
        raise GcpWorkflowError(f"unable to resolve machine series from {machine_type!r}")
    return token


def _zone_to_region(zone: str) -> str:
    parts = zone.split("-")
    if len(parts) < 2:
        raise GcpWorkflowError(f"unable to derive region from zone {zone!r}")
    return "-".join(parts[:-1])


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _default_upload_machine_type(disk_type: str) -> str:
    if disk_type == "hyperdisk-ml":
        return "n4-standard-4"
    return "e2-standard-4"


def _gcloud_config_value(key: str) -> str | None:
    proc = subprocess.run(
        [_find_gcloud_binary(), "config", "get-value", key],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    if not value or value == "(unset)":
        return None
    return value


@dataclass
class GcpRuntimeConfig:
    project: str
    zone: str
    repo_root: Path
    archive_root: Path
    data_disk_name: str = "kinexar-neurips-data"
    data_disk_type: str = "pd-standard"
    data_disk_size_gb: int = 1200
    data_disk_iops: int | None = None
    data_disk_throughput_mib: int | None = None
    upload_machine_type: str = "auto"
    training_machine_type: str = "g2-standard-4"
    training_accelerator_type: str | None = None
    training_accelerator_count: int = 0
    upload_boot_disk_size_gb: int = 64
    training_boot_disk_size_gb: int = 100
    image_family: str = "common-cu129-ubuntu-2204-nvidia-580"
    image_project: str = "deeplearning-platform-release"
    upload_max_run_duration: str = "24h"
    full_max_run_duration: str = "72h"
    ablation_max_run_duration: str = "48h"
    data_device_name: str = "kinexar-data"
    mount_point: str = "/mnt/kinexar-data"
    remote_repo_root: str = "/mnt/kinexar-data/workspaces/kinexar"
    remote_archive_root: str = "/mnt/kinexar-data/archives"
    remote_dataset_root: str = "/mnt/kinexar-data/data/neurips"
    remote_output_root: str = "/mnt/kinexar-data/outputs/neurips"
    remote_pixi_home: str = "/mnt/kinexar-data/.pixi-home"
    service_account: str | None = None
    scopes: str = "cloud-platform"
    ssh_ready_timeout_s: int = 1800
    ssh_poll_interval_s: int = 10
    gpu_installer_version: str = "v1.7.1"
    preserve_data_disk: bool = True
    archive_chunk_size_gb: int = 4

    def normalized(self) -> GcpRuntimeConfig:
        upload_machine_type = self.upload_machine_type
        if not upload_machine_type or upload_machine_type == "auto":
            upload_machine_type = _default_upload_machine_type(self.data_disk_type)
        return GcpRuntimeConfig(
            project=self.project,
            zone=self.zone,
            repo_root=Path(self.repo_root).expanduser().resolve(),
            archive_root=Path(self.archive_root).expanduser().resolve(),
            data_disk_name=self.data_disk_name,
            data_disk_type=self.data_disk_type,
            data_disk_size_gb=int(self.data_disk_size_gb),
            data_disk_iops=self.data_disk_iops,
            data_disk_throughput_mib=self.data_disk_throughput_mib,
            upload_machine_type=upload_machine_type,
            training_machine_type=self.training_machine_type,
            training_accelerator_type=self.training_accelerator_type,
            training_accelerator_count=int(self.training_accelerator_count),
            upload_boot_disk_size_gb=int(self.upload_boot_disk_size_gb),
            training_boot_disk_size_gb=int(self.training_boot_disk_size_gb),
            image_family=self.image_family,
            image_project=self.image_project,
            upload_max_run_duration=self.upload_max_run_duration,
            full_max_run_duration=self.full_max_run_duration,
            ablation_max_run_duration=self.ablation_max_run_duration,
            data_device_name=self.data_device_name,
            mount_point=self.mount_point,
            remote_repo_root=self.remote_repo_root,
            remote_archive_root=self.remote_archive_root,
            remote_dataset_root=self.remote_dataset_root,
            remote_output_root=self.remote_output_root,
            remote_pixi_home=self.remote_pixi_home,
            service_account=self.service_account,
            scopes=self.scopes,
            ssh_ready_timeout_s=int(self.ssh_ready_timeout_s),
            ssh_poll_interval_s=int(self.ssh_poll_interval_s),
            gpu_installer_version=self.gpu_installer_version,
            preserve_data_disk=bool(self.preserve_data_disk),
            archive_chunk_size_gb=int(self.archive_chunk_size_gb),
        )

    @property
    def region(self) -> str:
        return _zone_to_region(self.zone)

    @property
    def remote_data_device(self) -> str:
        return f"/dev/disk/by-id/google-{self.data_device_name}"

    def processed_root_for(self, family: str) -> str:
        profile = resolve_profile(family)
        return f"{self.remote_dataset_root}/{profile.dataset_dir_name}/full"

    def output_root_for(self, family: str) -> str:
        profile = resolve_profile(family)
        return f"{self.remote_output_root}/{profile.output_subdir}"

    def archive_paths(self) -> list[Path]:
        root = self.archive_root
        return [
            root / FULL_ARCHIVE,
            root / f"{FULL_ARCHIVE}.sha256",
            root / PARTS_ARCHIVE,
            root / f"{PARTS_ARCHIVE}.sha256",
        ]


@dataclass
class TrainingRuntime:
    batch_size: int = 32
    num_workers: int = 2
    vram_target_gb: float = 20.0
    vram_cap_gb: float = 22.0
    epochs: int | None = None


@dataclass
class AblationRequest:
    no_latent: bool = False
    single_expert: bool = False
    no_refine: bool = False
    weak_encoder: bool = False
    num_experts: int = 6
    seeds: tuple[int, ...] = (7,)
    runtime: TrainingRuntime = field(default_factory=TrainingRuntime)

    def variant_label(self) -> str:
        labels: list[str] = []
        if self.no_latent:
            labels.append("no_latent")
        if self.single_expert:
            labels.append("single_expert")
        if self.no_refine:
            labels.append("no_refine")
        if self.weak_encoder:
            labels.append("weak_encoder")
        if int(self.num_experts) != 6:
            labels.append(f"experts_{int(self.num_experts)}")
        if not labels:
            labels.append("full_kinexar")
        return "_".join(labels)


def discover_runtime_config(
    *,
    repo_root: str | Path | None = None,
    archive_root: str | Path | None = None,
    project: str | None = None,
    zone: str | None = None,
    data_disk_name: str = "kinexar-neurips-data",
    data_disk_type: str = "pd-standard",
    data_disk_size_gb: int = 1200,
    data_disk_iops: int | None = None,
    data_disk_throughput_mib: int | None = None,
    upload_machine_type: str = "auto",
    training_machine_type: str = "g2-standard-4",
    training_accelerator_type: str | None = None,
    training_accelerator_count: int = 0,
    upload_max_run_duration: str = "24h",
    full_max_run_duration: str = "72h",
    ablation_max_run_duration: str = "48h",
    service_account: str | None = None,
    archive_chunk_size_gb: int = 4,
) -> GcpRuntimeConfig:
    resolved_project = project or os.environ.get("KINEXAR_GCP_PROJECT") or _gcloud_config_value("project")
    resolved_zone = zone or os.environ.get("KINEXAR_GCP_ZONE") or _gcloud_config_value("compute/zone") or DEFAULT_GCP_ZONE
    if not resolved_project:
        raise GcpWorkflowError("missing GCP project; set gcloud config project or pass --project")
    root = Path(repo_root or Path.cwd()).expanduser().resolve()
    archives = Path(archive_root or archive_root_from_env()).expanduser().resolve()
    config = GcpRuntimeConfig(
        project=str(resolved_project),
        zone=str(resolved_zone),
        repo_root=root,
        archive_root=archives,
        data_disk_name=data_disk_name,
        data_disk_type=data_disk_type,
        data_disk_size_gb=int(data_disk_size_gb),
        data_disk_iops=data_disk_iops,
        data_disk_throughput_mib=data_disk_throughput_mib,
        upload_machine_type=upload_machine_type,
        training_machine_type=training_machine_type,
        training_accelerator_type=training_accelerator_type,
        training_accelerator_count=int(training_accelerator_count),
        upload_max_run_duration=upload_max_run_duration,
        full_max_run_duration=full_max_run_duration,
        ablation_max_run_duration=ablation_max_run_duration,
        service_account=service_account,
        archive_chunk_size_gb=int(archive_chunk_size_gb),
    ).normalized()
    validate_runtime_config(config)
    return config


def validate_runtime_config(config: GcpRuntimeConfig) -> None:
    if not (config.repo_root / "pixi.toml").exists():
        raise GcpWorkflowError(f"repo root does not look valid: {config.repo_root}")
    training_series = _machine_series(config.training_machine_type)
    if training_series == "g2" and config.data_disk_type in G2_UNSUPPORTED_HYPERDISKS:
        raise GcpWorkflowError(
            "g2-standard-* cannot attach hyperdisk-balanced. "
            "Use pd-balanced or hyperdisk-ml on G2, or switch to a G4 machine if you require hyperdisk-balanced."
        )
    if training_series == "g4" and config.data_disk_type.startswith("pd-"):
        raise GcpWorkflowError(
            "g4 machine types cannot attach Persistent Disk data volumes. "
            "This workflow depends on the existing zonal dataset disk, so g4 is not a valid replacement for the "
            "Mumbai data-local training path."
        )
    if config.data_disk_type == "hyperdisk-ml":
        training_series = _machine_series(config.training_machine_type)
        upload_series = _machine_series(config.upload_machine_type)
        if training_series not in HYPERDISK_ML_SUPPORTED_SERIES:
            raise GcpWorkflowError(
                f"training machine {config.training_machine_type} cannot attach hyperdisk-ml; "
                f"use one of {sorted(HYPERDISK_ML_SUPPORTED_SERIES)}-series machines."
            )
        if upload_series not in HYPERDISK_ML_SUPPORTED_SERIES:
            raise GcpWorkflowError(
                f"upload machine {config.upload_machine_type} cannot attach hyperdisk-ml; "
                f"use one of {sorted(HYPERDISK_ML_SUPPORTED_SERIES)}-series machines."
            )
        if config.data_disk_iops is not None:
            raise GcpWorkflowError("hyperdisk-ml does not support provisioned IOPS; remove --data-disk-iops")
    if config.data_disk_type == "hyperdisk-throughput" and int(config.data_disk_size_gb) < 2048:
        raise GcpWorkflowError("hyperdisk-throughput requires at least 2048 GB")
    if int(config.data_disk_size_gb) < 1000:
        raise GcpWorkflowError("data_disk_size_gb is too small for the expanded dataset plus outputs")
    if int(config.archive_chunk_size_gb) < 1:
        raise GcpWorkflowError("archive_chunk_size_gb must be at least 1")
    if config.training_accelerator_count < 0:
        raise GcpWorkflowError("training_accelerator_count cannot be negative")
    if config.training_accelerator_type and int(config.training_accelerator_count) < 1:
        raise GcpWorkflowError("training_accelerator_count must be at least 1 when training_accelerator_type is set")
    if int(config.training_accelerator_count) > 0 and not config.training_accelerator_type:
        raise GcpWorkflowError("training_accelerator_type is required when training_accelerator_count is greater than 0")


def _run(
    args: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=check,
        capture_output=capture_output,
        text=text,
    )


def _gcloud_args(config: GcpRuntimeConfig, extra: Sequence[str]) -> list[str]:
    return [_find_gcloud_binary(), "--quiet", "--project", config.project, *extra]


def _gcloud(config: GcpRuntimeConfig, extra: Sequence[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return _run(_gcloud_args(config, extra), check=check, capture_output=capture_output)


def _instance_exists(config: GcpRuntimeConfig, name: str) -> bool:
    proc = _gcloud(
        config,
        ["compute", "instances", "describe", name, "--zone", config.zone],
        check=False,
        capture_output=True,
    )
    return proc.returncode == 0


def _disk_exists(config: GcpRuntimeConfig) -> bool:
    proc = _gcloud(
        config,
        ["compute", "disks", "describe", config.data_disk_name, "--zone", config.zone],
        check=False,
        capture_output=True,
    )
    return proc.returncode == 0


def _disk_users(config: GcpRuntimeConfig) -> list[str]:
    proc = _gcloud(
        config,
        [
            "compute",
            "disks",
            "describe",
            config.data_disk_name,
            "--zone",
            config.zone,
            "--format=json(users)",
        ],
        capture_output=True,
    )
    payload = json.loads(proc.stdout or "{}")
    if payload is None:
        return []
    users = payload.get("users") or []
    if not isinstance(users, list):
        return []
    return [str(user) for user in users]


def ensure_data_disk(config: GcpRuntimeConfig) -> None:
    if _disk_exists(config):
        return
    disk_attempts = [config.data_disk_type]
    if config.data_disk_type == "hyperdisk-ml":
        disk_attempts.extend(["pd-balanced", "pd-standard"])
    elif config.data_disk_type == "pd-balanced":
        disk_attempts.append("pd-standard")
    last_exc: subprocess.CalledProcessError | None = None
    for disk_type in disk_attempts:
        cmd = [
            "compute",
            "disks",
            "create",
            config.data_disk_name,
            "--zone",
            config.zone,
            "--size",
            f"{int(config.data_disk_size_gb)}GB",
            "--type",
            disk_type,
        ]
        if disk_type == config.data_disk_type:
            if config.data_disk_iops is not None:
                cmd.extend(["--provisioned-iops", str(int(config.data_disk_iops))])
            if config.data_disk_throughput_mib is not None:
                cmd.extend(["--provisioned-throughput", str(int(config.data_disk_throughput_mib))])
        try:
            _gcloud(config, cmd, capture_output=True)
            config.data_disk_type = disk_type
            if disk_type != "hyperdisk-ml":
                config.data_disk_iops = None
                config.data_disk_throughput_mib = None
            return
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "") + "\n" + (exc.stdout or "")
            last_exc = exc
            if disk_type == "hyperdisk-ml" and "HDML_TOTAL_GB" in stderr:
                continue
            if disk_type == "pd-balanced" and "SSD_TOTAL_GB" in stderr:
                continue
            raise
    if last_exc is not None:
        raise last_exc


def _wait_for_data_disk_free(
    config: GcpRuntimeConfig,
    *,
    timeout_s: int | None = None,
) -> None:
    timeout = int(timeout_s or config.ssh_ready_timeout_s)
    deadline = time.time() + float(timeout)
    last_users: list[str] = []
    while time.time() < deadline:
        users = _disk_users(config)
        if not users:
            return
        last_users = users
        joined = ", ".join(users)
        _progress(f"waiting for data disk {config.data_disk_name} to become free; currently attached to: {joined}")
        time.sleep(float(config.ssh_poll_interval_s))
    joined = ", ".join(last_users) if last_users else "unknown instance"
    raise GcpResourceBusyError(
        f"data disk {config.data_disk_name} is still attached after waiting {timeout}s: {joined}"
    )


def _delete_instance(config: GcpRuntimeConfig, name: str) -> None:
    if not _instance_exists(config, name):
        return
    cmd = ["compute", "instances", "delete", name, "--zone", config.zone]
    if config.preserve_data_disk:
        cmd.extend(["--keep-disks", "data"])
    _gcloud(config, cmd, check=False)


def _wait_for_ssh(config: GcpRuntimeConfig, name: str) -> None:
    deadline = time.time() + float(config.ssh_ready_timeout_s)
    last_error = ""
    while time.time() < deadline:
        proc = _gcloud(
            config,
            [
                "compute",
                "ssh",
                name,
                "--zone",
                config.zone,
                "--command",
                "true",
                "--ssh-flag",
                "-o ConnectTimeout=5",
                "--ssh-flag",
                "-o StrictHostKeyChecking=no",
                "--ssh-flag",
                "-o UserKnownHostsFile=/dev/null",
            ],
            check=False,
            capture_output=True,
        )
        if proc.returncode == 0:
            return
        last_error = (proc.stderr or proc.stdout or "").strip()
        time.sleep(float(config.ssh_poll_interval_s))
    raise GcpWorkflowError(f"timed out waiting for SSH on {name}: {last_error}")


def _scp(config: GcpRuntimeConfig, local_path: Path, instance_name: str, remote_path: str) -> None:
    _gcloud(
        config,
        [
            "compute",
            "scp",
            str(local_path),
            f"{instance_name}:{remote_path}",
            "--zone",
            config.zone,
            "--scp-flag",
            "-o StrictHostKeyChecking=no",
            "--scp-flag",
            "-o UserKnownHostsFile=/dev/null",
        ],
    )


def _read_expected_sha(sha_path: Path) -> str:
    raw = sha_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise GcpWorkflowError(f"empty checksum file: {sha_path}")
    return raw.split()[0]


def _ssh(config: GcpRuntimeConfig, instance_name: str, command: str, *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return _gcloud(
        config,
        [
            "compute",
            "ssh",
            instance_name,
            "--zone",
            config.zone,
            "--command",
            command,
            "--ssh-flag",
            "-o StrictHostKeyChecking=no",
            "--ssh-flag",
            "-o UserKnownHostsFile=/dev/null",
        ],
        check=check,
        capture_output=capture_output,
    )


def _remote_stat_size(config: GcpRuntimeConfig, instance_name: str, remote_path: str) -> int | None:
    proc = _ssh(
        config,
        instance_name,
        f"if [ -f {shlex.quote(remote_path)} ]; then stat -c %s {shlex.quote(remote_path)}; fi",
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    raw = proc.stdout.strip()
    if not raw:
        return None
    return int(raw)


def _copy_if_needed(config: GcpRuntimeConfig, instance_name: str, local_path: Path, remote_path: str) -> bool:
    remote_size = _remote_stat_size(config, instance_name, remote_path)
    local_size = local_path.stat().st_size
    if remote_size == local_size:
        return False
    _scp(config, local_path, instance_name, remote_path)
    return True


def _remote_path_exists(config: GcpRuntimeConfig, instance_name: str, remote_path: str) -> bool:
    proc = _ssh(
        config,
        instance_name,
        f"test -e {shlex.quote(remote_path)}",
        check=False,
        capture_output=True,
    )
    return proc.returncode == 0


def _stage_chunk(local_path: Path, offset: int, length: int) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix=f"{local_path.stem}_chunk_", suffix=".part", delete=False)
    chunk_path = Path(handle.name)
    try:
        with local_path.open("rb") as src, chunk_path.open("wb") as dst:
            src.seek(offset)
            remaining = int(length)
            while remaining > 0:
                block = src.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise GcpWorkflowError(f"unexpected EOF while chunking {local_path}")
                dst.write(block)
                remaining -= len(block)
    finally:
        handle.close()
    return chunk_path


def _upload_large_archive_if_needed(config: GcpRuntimeConfig, instance_name: str, local_path: Path, remote_path: str) -> bool:
    local_size = local_path.stat().st_size
    expected_sha = _read_expected_sha(local_path.with_name(f"{local_path.name}.sha256"))
    remote_sha_path = f"{remote_path}.sha256"
    remote_done_flag = f"{remote_path}.upload_complete"
    verify_cmd = (
        f"if [ -f {shlex.quote(remote_path)} ] && [ -f {shlex.quote(remote_done_flag)} ]; then "
        f"actual=$(sha256sum {shlex.quote(remote_path)} | awk '{{print $1}}'); "
        f"test \"$actual\" = {shlex.quote(expected_sha)}; "
        f"else "
        f"exit 1; "
        f"fi"
    )
    verify_proc = _ssh(config, instance_name, verify_cmd, check=False, capture_output=True)
    if verify_proc.returncode == 0:
        return False

    chunk_size = max(int(config.archive_chunk_size_gb), 1) * 1024**3
    chunk_count = int(math.ceil(local_size / chunk_size))
    remote_parts_dir = f"{remote_path}.parts"
    _ssh(config, instance_name, f"mkdir -p {shlex.quote(remote_parts_dir)}")
    for chunk_index in range(chunk_count):
        offset = chunk_index * chunk_size
        length = min(chunk_size, local_size - offset)
        remote_chunk = f"{remote_parts_dir}/{Path(remote_path).name}.part-{chunk_index:04d}"
        remote_size = _remote_stat_size(config, instance_name, remote_chunk)
        if remote_size == length:
            continue
        chunk_path = _stage_chunk(local_path, offset, length)
        try:
            _scp(config, chunk_path, instance_name, remote_chunk)
        finally:
            chunk_path.unlink(missing_ok=True)
    assemble_lines = [
        "set -euo pipefail",
        f"remote_path={shlex.quote(remote_path)}",
        f"remote_sha_path={shlex.quote(remote_sha_path)}",
        f"remote_done_flag={shlex.quote(remote_done_flag)}",
        f"remote_parts_dir={shlex.quote(remote_parts_dir)}",
        f"expected_sha={shlex.quote(expected_sha)}",
        "tmp_path=\"${remote_path}.assembling\"",
        "rm -f \"$tmp_path\" \"$remote_done_flag\"",
        "for part in \"${remote_parts_dir}\"/*.part-*; do",
        "  cat \"$part\" >> \"$tmp_path\"",
        "done",
        "actual_sha=\"$(sha256sum \"$tmp_path\" | awk '{print $1}')\"",
        "if [ \"$actual_sha\" != \"$expected_sha\" ]; then",
        "  echo \"sha mismatch for ${remote_path}: expected ${expected_sha} got ${actual_sha}\" >&2",
        "  exit 1",
        "fi",
        "mv \"$tmp_path\" \"$remote_path\"",
        "printf '%s  %s\\n' \"$expected_sha\" \"$(basename \"$remote_path\")\" > \"$remote_sha_path\"",
        "touch \"$remote_done_flag\"",
        "rm -rf \"$remote_parts_dir\"",
    ]
    _ssh(config, instance_name, "bash -lc " + shlex.quote("\n".join(assemble_lines)))
    post_verify_proc = _ssh(config, instance_name, verify_cmd, check=False, capture_output=True)
    if post_verify_proc.returncode != 0:
        raise GcpWorkflowError(f"remote archive verification failed after upload: {remote_path}")
    return True


def _build_repo_bundle(repo_root: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="kinexar_repo_", suffix=".tgz", delete=False)
    bundle_path = Path(handle.name)
    handle.close()
    with tarfile.open(bundle_path, mode="w:gz") as tar:
        for item in REPO_BUNDLE_TOP_LEVEL:
            path = repo_root / item
            if path.exists():
                tar.add(path, arcname=item)
    return bundle_path


def _remote_helper_script(config: GcpRuntimeConfig) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        export KINEXAR_MOUNT_POINT={shlex.quote(config.mount_point)}
        export KINEXAR_REMOTE_REPO_ROOT={shlex.quote(config.remote_repo_root)}
        export KINEXAR_REMOTE_ARCHIVE_ROOT={shlex.quote(config.remote_archive_root)}
        export KINEXAR_REMOTE_DATASET_ROOT={shlex.quote(config.remote_dataset_root)}
        export KINEXAR_REMOTE_OUTPUT_ROOT={shlex.quote(config.remote_output_root)}
        export KINEXAR_PIXI_HOME={shlex.quote(config.remote_pixi_home)}
        export KINEXAR_DATA_DEVICE={shlex.quote(config.remote_data_device)}
        export KINEXAR_GPU_INSTALLER_VERSION={shlex.quote(config.gpu_installer_version)}
        export USER="${{USER:-$(id -un)}}"

        kinexar_export_env() {{
          export PIXI_HOME="${{KINEXAR_PIXI_HOME}}"
          export XDG_CACHE_HOME="${{KINEXAR_PIXI_HOME}}/xdg-cache"
          export PATH="${{KINEXAR_PIXI_HOME}}/bin:${{HOME}}/.pixi/bin:${{PATH}}"
        }}

        kinexar_mount_data_disk() {{
          sudo mkdir -p "${{KINEXAR_MOUNT_POINT}}"
          if ! sudo blkid "${{KINEXAR_DATA_DEVICE}}" >/dev/null 2>&1; then
            sudo mkfs.ext4 -F -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "${{KINEXAR_DATA_DEVICE}}"
          fi
          local uuid
          uuid="$(sudo blkid -s UUID -o value "${{KINEXAR_DATA_DEVICE}}")"
          if ! grep -q "${{uuid}}" /etc/fstab; then
            echo "UUID=${{uuid}} ${{KINEXAR_MOUNT_POINT}} ext4 defaults,discard,nofail 0 2" | sudo tee -a /etc/fstab >/dev/null
          fi
          if ! mountpoint -q "${{KINEXAR_MOUNT_POINT}}"; then
            sudo mount "${{KINEXAR_MOUNT_POINT}}"
          fi
          sudo chown "${{USER}}:${{USER}}" "${{KINEXAR_MOUNT_POINT}}"
          mkdir -p "${{KINEXAR_REMOTE_ARCHIVE_ROOT}}" "${{KINEXAR_REMOTE_DATASET_ROOT}}" "${{KINEXAR_REMOTE_OUTPUT_ROOT}}" "${{KINEXAR_PIXI_HOME}}"
          mkdir -p "$(dirname "${{KINEXAR_REMOTE_REPO_ROOT}}")"
          sudo chown -R "${{USER}}:${{USER}}" "${{KINEXAR_PIXI_HOME}}" || true
        }}

        kinexar_install_packages() {{
          local marker="/var/lib/kinexar/bootstrap.done"
          if sudo test -f "${{marker}}"; then
            return 0
          fi
          sudo apt-get update -y
          sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git gpg jq python3 python3-pip python3-venv build-essential zstd
          sudo mkdir -p /var/lib/kinexar
          sudo touch "${{marker}}"
        }}

        kinexar_install_pixi() {{
          kinexar_export_env
          if ! command -v pixi >/dev/null 2>&1; then
            curl -fsSL https://pixi.sh/install.sh | sh
          fi
          command -v pixi >/dev/null 2>&1
        }}

        kinexar_bootstrap_base() {{
          kinexar_mount_data_disk
          kinexar_install_packages
        }}

        kinexar_bootstrap_repo() {{
          kinexar_bootstrap_base
          kinexar_install_pixi
        }}

        kinexar_refresh_repo() {{
          kinexar_bootstrap_repo
          kinexar_export_env
          mkdir -p "${{KINEXAR_REMOTE_REPO_ROOT}}"
          sudo chown -R "${{USER}}:${{USER}}" "${{KINEXAR_REMOTE_REPO_ROOT}}" || true
          find "${{KINEXAR_REMOTE_REPO_ROOT}}" -mindepth 1 -maxdepth 1 ! -name '.pixi' -exec rm -rf {{}} +
          tar -xzf /tmp/kinexar_repo_bundle.tgz -C "${{KINEXAR_REMOTE_REPO_ROOT}}"
          cd "${{KINEXAR_REMOTE_REPO_ROOT}}"
          pixi install --locked
        }}

        kinexar_prepare_layout() {{
          kinexar_refresh_repo
          cd "${{KINEXAR_REMOTE_REPO_ROOT}}"
          pixi run python scripts/manage_archives.py prepare-training-layout --archive-root "${{KINEXAR_REMOTE_ARCHIVE_ROOT}}" --output-root "${{KINEXAR_REMOTE_DATASET_ROOT}}" "$@"
        }}

        kinexar_gpu_ready() {{
          command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1
        }}

        kinexar_install_gpu() {{
          kinexar_bootstrap_base
          if kinexar_gpu_ready; then
            return 0
          fi
          cd /tmp
          if [ ! -f /tmp/cuda_installer.pyz ]; then
            curl -fSsL -O "https://storage.googleapis.com/compute-gpu-installation-us/installer/${{KINEXAR_GPU_INSTALLER_VERSION}}/cuda_installer.pyz"
          fi
          sudo python3 /tmp/cuda_installer.pyz install_driver --installation-mode=repo --installation-branch=prod
        }}

        kinexar_install_docker() {{
          local marker="/var/lib/kinexar/docker.done"
          kinexar_bootstrap_base
          if sudo test -f "${{marker}}"; then
            sudo systemctl enable --now docker >/dev/null 2>&1 || true
            return 0
          fi
          sudo apt-get update -y
          sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
          sudo systemctl enable --now docker
          sudo touch "${{marker}}"
        }}

        kinexar_install_nvidia_container_toolkit() {{
          local marker="/var/lib/kinexar/nvidia-container-toolkit.done"
          kinexar_install_docker
          if ! kinexar_gpu_ready; then
            echo "nvidia driver is not ready yet" >&2
            return 1
          fi
          if ! command -v nvidia-ctk >/dev/null 2>&1; then
            local distro
            distro="$(
              . /etc/os-release
              echo "${{ID}}${{VERSION_ID}}"
            )"
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
              | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -fsSL "https://nvidia.github.io/libnvidia-container/${{distro}}/libnvidia-container.list" \
              | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
              | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
            sudo apt-get update -y
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
          fi
          sudo nvidia-ctk runtime configure --runtime=docker
          sudo systemctl restart docker
          sudo touch "${{marker}}"
        }}

        kinexar_install_docker_stack() {{
          kinexar_install_docker
          if kinexar_gpu_ready; then
            kinexar_install_nvidia_container_toolkit
          fi
        }}

        if [[ "${{BASH_SOURCE[0]}}" == "${{0}}" ]]; then
          cmd="${{1:-}}"
          shift || true
          case "${{cmd}}" in
            bootstrap) kinexar_bootstrap_base "$@" ;;
            refresh-repo) kinexar_refresh_repo "$@" ;;
            prepare-layout) kinexar_prepare_layout "$@" ;;
            gpu-ready) kinexar_gpu_ready "$@" ;;
            install-gpu) kinexar_install_gpu "$@" ;;
            install-docker) kinexar_install_docker "$@" ;;
            install-docker-stack) kinexar_install_docker_stack "$@" ;;
            install-nvidia-container-toolkit) kinexar_install_nvidia_container_toolkit "$@" ;;
            *)
              echo "unsupported subcommand: ${{cmd}}" >&2
              exit 2
              ;;
          esac
        fi
        """
    )


def _write_temp_script(prefix: str, content: str) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".sh", delete=False)
    path = Path(handle.name)
    handle.write(content.encode("utf-8"))
    handle.flush()
    handle.close()
    path.chmod(0o755)
    return path


def _stage_remote_helper(config: GcpRuntimeConfig, instance_name: str, repo_bundle_path: Path | None = None) -> None:
    helper_path = _write_temp_script("kinexar_gcp_helper_", _remote_helper_script(config))
    try:
        _scp(config, helper_path, instance_name, "/tmp/kinexar_remote_helper.sh")
    finally:
        helper_path.unlink(missing_ok=True)
    _ssh(config, instance_name, "chmod +x /tmp/kinexar_remote_helper.sh && bash /tmp/kinexar_remote_helper.sh bootstrap")
    if repo_bundle_path is not None:
        _scp(config, repo_bundle_path, instance_name, "/tmp/kinexar_repo_bundle.tgz")
        _ssh(config, instance_name, "bash /tmp/kinexar_remote_helper.sh refresh-repo")


def _run_remote_script(config: GcpRuntimeConfig, instance_name: str, script_content: str) -> subprocess.CompletedProcess[str]:
    local_script = _write_temp_script("kinexar_gcp_job_", script_content)
    try:
        _scp(config, local_script, instance_name, "/tmp/kinexar_remote_job.sh")
    finally:
        local_script.unlink(missing_ok=True)
    return _ssh(config, instance_name, "chmod +x /tmp/kinexar_remote_job.sh && bash /tmp/kinexar_remote_job.sh", check=True)


def _start_remote_service(
    config: GcpRuntimeConfig,
    instance_name: str,
    *,
    unit_name: str,
    script_name: str,
    script_content: str,
) -> str:
    remote_script = f"/tmp/{script_name}"
    local_script = _write_temp_script(script_name.replace("/", "_"), script_content)
    try:
        _scp(config, local_script, instance_name, remote_script)
    finally:
        local_script.unlink(missing_ok=True)
    service_name = f"{unit_name}.service"
    launch_cmd = (
        f"chmod +x {shlex.quote(remote_script)} && "
        f"sudo systemctl stop {shlex.quote(service_name)} >/dev/null 2>&1 || true && "
        f"sudo systemctl reset-failed {shlex.quote(service_name)} >/dev/null 2>&1 || true && "
        f"sudo systemd-run --unit {shlex.quote(unit_name)} "
        f"--property=Restart=no "
        f"--property=Type=simple "
        f"--collect "
        f"bash {shlex.quote(remote_script)}"
    )
    _ssh(config, instance_name, launch_cmd)
    return service_name


def _create_instance(
    config: GcpRuntimeConfig,
    *,
    instance_name: str,
    machine_type: str,
    boot_disk_size_gb: int,
    max_run_duration: str,
    labels: dict[str, str],
    accelerator_type: str | None = None,
    accelerator_count: int = 0,
) -> None:
    label_arg = ",".join(f"{key}={value}" for key, value in sorted(labels.items()))
    cmd = [
        "compute",
        "instances",
        "create",
        instance_name,
        "--zone",
        config.zone,
        "--machine-type",
        machine_type,
        "--boot-disk-size",
        f"{int(boot_disk_size_gb)}GB",
        "--boot-disk-type",
        "pd-balanced",
        "--image-family",
        config.image_family,
        "--image-project",
        config.image_project,
        "--scopes",
        config.scopes,
        "--max-run-duration",
        max_run_duration,
        "--instance-termination-action",
        "DELETE",
        "--labels",
        label_arg,
        "--disk",
        f"name={config.data_disk_name},device-name={config.data_device_name},mode=rw,boot=no,auto-delete=no",
    ]
    if accelerator_type and int(accelerator_count) > 0:
        cmd.extend(["--accelerator", f"type={accelerator_type},count={int(accelerator_count)}"])
    if _machine_series(machine_type) == "g2":
        cmd.extend(["--maintenance-policy", "TERMINATE", "--restart-on-failure"])
    if config.service_account:
        cmd.extend(["--service-account", config.service_account])
    _gcloud(config, cmd)


def _gpu_install_loop(config: GcpRuntimeConfig, instance_name: str) -> None:
    deadline = time.time() + float(config.ssh_ready_timeout_s)
    while time.time() < deadline:
        proc = _ssh(config, instance_name, "bash /tmp/kinexar_remote_helper.sh gpu-ready", check=False, capture_output=True)
        if proc.returncode == 0:
            return
        install_proc = _ssh(
            config,
            instance_name,
            "bash /tmp/kinexar_remote_helper.sh install-gpu",
            check=False,
            capture_output=True,
        )
        if install_proc.returncode == 0:
            time.sleep(float(config.ssh_poll_interval_s))
        else:
            time.sleep(float(config.ssh_poll_interval_s))
        _wait_for_ssh(config, instance_name)
    raise GcpWorkflowError("timed out waiting for NVIDIA driver installation to complete")


def _launcher_log_root(config: GcpRuntimeConfig, family: str) -> str:
    return f"{config.remote_output_root}/launcher_logs/{family}"


def _safe_service_name(value: str) -> str:
    lowered = value.strip().lower()
    chars = [ch if ch.isalnum() else "-" for ch in lowered]
    compact = "".join(chars).strip("-")
    while "--" in compact:
        compact = compact.replace("--", "-")
    return compact or "kinexar-job"


def _training_script(
    config: GcpRuntimeConfig,
    *,
    family: str,
    run_name: str,
    output_root: str,
    processed_root: str,
    runtime: TrainingRuntime,
    visualize: bool = False,
    extra_args: Sequence[str],
) -> str:
    log_root = _launcher_log_root(config, family)
    command = [
        "pixi",
        "run",
        "python",
        "scripts/run_neurips.py",
        "--family",
        family,
        "--processed-root",
        processed_root,
        "--output-root",
        output_root,
        "--run-name",
        run_name,
        "--device",
        "cuda",
        "--batch-size",
        str(int(runtime.batch_size)),
        "--num-workers",
        str(int(runtime.num_workers)),
        "--vram-target-gb",
        str(float(runtime.vram_target_gb)),
        "--vram-cap-gb",
        str(float(runtime.vram_cap_gb)),
    ]
    if runtime.epochs is not None:
        command.extend(["--epochs", str(int(runtime.epochs))])
    command.extend(list(extra_args))
    command_str = _shell_join(command)
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        source /tmp/kinexar_remote_helper.sh
        kinexar_mount_data_disk
        kinexar_export_env
        mkdir -p {shlex.quote(log_root)} {shlex.quote(output_root)}
        cd "${{KINEXAR_REMOTE_REPO_ROOT}}"
        log_file={shlex.quote(f"{log_root}/{run_name}.log")}
        status_file={shlex.quote(f"{log_root}/{run_name}.status.json")}
        started_at="$(date -Is)"
        set +e
        (
          echo "[${{started_at}}] family={family} run_name={run_name}"
          echo "processed_root={processed_root}"
          echo "output_root={output_root}"
          {command_str}
        ) 2>&1 | tee -a "${{log_file}}"
        exit_code="${{PIPESTATUS[0]}}"
        set -e
        finished_at="$(date -Is)"
        cat > "${{status_file}}" <<EOF
        {{
          "family": {json.dumps(family)},
          "run_name": {json.dumps(run_name)},
          "processed_root": {json.dumps(processed_root)},
          "output_root": {json.dumps(output_root)},
          "started_at": "${{started_at}}",
          "finished_at": "${{finished_at}}",
          "log_file": "${{log_file}}",
          "exit_code": ${{exit_code}}
        }}
        EOF
        exit "${{exit_code}}"
        """
    )


def _training_service_script(
    config: GcpRuntimeConfig,
    *,
    family: str,
    run_names: Sequence[str],
    output_root: str,
    processed_root: str,
    runtime: TrainingRuntime,
    visualize: bool = False,
    command_matrix: Sequence[Sequence[str]],
    aggregate_seed_output: str | None = None,
) -> str:
    if len(run_names) != len(command_matrix):
        raise ValueError("run_names and command_matrix must have the same length")
    log_root = _launcher_log_root(config, family)
    primary_run_name = run_names[0]
    primary_log = f"{log_root}/{primary_run_name}.log"
    status_file = f"{log_root}/{primary_run_name}.status.json"
    command_lines: list[str] = []
    for run_name, extra_args in zip(run_names, command_matrix, strict=True):
        command = [
            "pixi",
            "run",
            "python",
            "scripts/run_neurips.py",
            "--family",
            family,
            "--processed-root",
            processed_root,
            "--output-root",
            output_root,
            "--run-name",
            run_name,
            "--device",
            "cuda",
            "--batch-size",
            str(int(runtime.batch_size)),
            "--num-workers",
            str(int(runtime.num_workers)),
            "--vram-target-gb",
            str(float(runtime.vram_target_gb)),
            "--vram-cap-gb",
            str(float(runtime.vram_cap_gb)),
            *list(extra_args),
        ]
        if runtime.epochs is not None:
            command.extend(["--epochs", str(int(runtime.epochs))])
        command_lines.append(_shell_join(command))
    if aggregate_seed_output is not None and len(run_names) > 1:
        run_dir_args: list[str] = []
        for run_name in run_names:
            run_dir_args.extend(["--run-dir", f"{output_root}/{run_name}"])
        command_lines.append(
            _shell_join(
                [
                    "pixi",
                    "run",
                    "python",
                    "scripts/aggregate_seed_results.py",
                    *run_dir_args,
                    "--output",
                    aggregate_seed_output,
                ]
            )
        )
    workflow_body = "\n".join(command_lines)
    run_names_json = json.dumps(list(run_names))
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        LOG_ROOT={shlex.quote(log_root)}
        LOG_FILE={shlex.quote(primary_log)}
        STATUS_FILE={shlex.quote(status_file)}
        mkdir -p "${{LOG_ROOT}}" {shlex.quote(output_root)}
        exec > >(tee -a "${{LOG_FILE}}") 2>&1
        trap 'finalize $?' EXIT

        meta() {{
          curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/$1"
        }}

        delete_self() {{
          local token project zone instance url
          token="$(python3 - <<'PY'
import json, urllib.request
req = urllib.request.Request(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    headers={{"Metadata-Flavor": "Google"}},
)
print(json.load(urllib.request.urlopen(req))["access_token"])
PY
)"
          project="$(meta project/project-id)"
          zone="$(basename "$(meta instance/zone)")"
          instance="$(meta instance/name)"
          url="https://compute.googleapis.com/compute/v1/projects/${{project}}/zones/${{zone}}/instances/${{instance}}"
          echo "[$(date -Is)] requesting self-delete via ${{url}}"
          curl -fsS -X DELETE -H "Authorization: Bearer ${{token}}" "${{url}}"
        }}

        finalize() {{
          local exit_code="${{1}}"
          local finished_at
          finished_at="$(date -Is)"
          cat > "${{STATUS_FILE}}" <<JSON
{{
  "family": {json.dumps(family)},
  "run_names": {run_names_json},
  "processed_root": {json.dumps(processed_root)},
  "output_root": {json.dumps(output_root)},
  "started_at": "${{STARTED_AT}}",
  "finished_at": "${{finished_at}}",
  "log_file": "${{LOG_FILE}}",
  "exit_code": ${{exit_code}}
}}
JSON
          if delete_self; then
            echo "[$(date -Is)] self-delete request accepted"
            return "${{exit_code}}"
          fi
          echo "[$(date -Is)] self-delete request failed; skipping shutdown to allow debugging." >&2
          return "${{exit_code}}"
        }}

        echo "[$(date -Is)] starting remote {family} workflow"
        source /tmp/kinexar_remote_helper.sh
        kinexar_mount_data_disk
        kinexar_export_env
        STARTED_AT="$(date -Is)"
        cd "${{KINEXAR_REMOTE_REPO_ROOT}}"

        set +e
{workflow_body}
        exit_code=$?
        set -e
        """
    )


def _ablation_extra_args(request: AblationRequest, seed: int) -> list[str]:
    args = ["--seed", str(int(seed))]
    if request.no_latent:
        args.append("--no-latent")
    if request.single_expert:
        args.append("--single-expert")
    if request.no_refine:
        args.append("--no-refine")
    if request.weak_encoder:
        args.append("--weak-encoder")
    if not request.single_expert:
        args.extend(["--num-experts", str(int(request.num_experts))])
    return args


def create_upload_vm(config: GcpRuntimeConfig, *, instance_name: str) -> None:
    _progress(f"ensuring data disk {config.data_disk_name} ({config.data_disk_type}, {config.data_disk_size_gb} GB)")
    ensure_data_disk(config)
    _progress(f"creating upload VM {instance_name} in {config.zone} with {config.upload_machine_type}")
    _create_instance(
        config,
        instance_name=instance_name,
        machine_type=config.upload_machine_type,
        boot_disk_size_gb=config.upload_boot_disk_size_gb,
        max_run_duration=config.upload_max_run_duration,
        labels={"app": "kinexar", "role": "upload"},
    )


def create_training_vm(config: GcpRuntimeConfig, *, instance_name: str, family: str) -> None:
    _progress(f"ensuring data disk {config.data_disk_name} ({config.data_disk_type}, {config.data_disk_size_gb} GB)")
    ensure_data_disk(config)
    _wait_for_data_disk_free(config)
    max_run_duration = config.full_max_run_duration if family == "full" else config.ablation_max_run_duration
    _progress(f"creating training VM {instance_name} in {config.zone} with {config.training_machine_type}")
    _create_instance(
        config,
        instance_name=instance_name,
        machine_type=config.training_machine_type,
        boot_disk_size_gb=config.training_boot_disk_size_gb,
        max_run_duration=max_run_duration,
        labels={"app": "kinexar", "role": family},
        accelerator_type=config.training_accelerator_type,
        accelerator_count=config.training_accelerator_count,
    )


def run_upload_workflow(config: GcpRuntimeConfig, *, force: bool = False) -> dict[str, Any]:
    _progress(f"verifying local archives under {config.archive_root}")
    verify_archives(config.archive_root)
    _progress("building repo bundle for remote helper")
    repo_bundle = _build_repo_bundle(config.repo_root)
    instance_name = f"kinexar-upload-{_timestamp()}".replace("_", "-").lower()
    try:
        create_upload_vm(config, instance_name=instance_name)
        _progress(f"waiting for SSH on {instance_name}")
        _wait_for_ssh(config, instance_name)
        _progress(f"bootstrapping remote helper on {instance_name}")
        _stage_remote_helper(config, instance_name, repo_bundle_path=repo_bundle)
        copied: list[str] = []
        for path in config.archive_paths():
            remote_path = f"{config.remote_archive_root}/{path.name}"
            _progress(f"staging {path.name} -> {remote_path}")
            if path.name in {FULL_ARCHIVE, PARTS_ARCHIVE}:
                changed = _upload_large_archive_if_needed(config, instance_name, path, remote_path)
            else:
                changed = _copy_if_needed(config, instance_name, path, remote_path)
            if changed:
                copied.append(path.name)
                _progress(f"copied {path.name}")
            else:
                _progress(f"reused remote copy of {path.name}")
        force_flag = ["--force"] if force else []
        _progress("preparing extracted dataset layout on remote disk")
        _ssh(
            config,
            instance_name,
            _shell_join(["bash", "/tmp/kinexar_remote_helper.sh", "prepare-layout", *force_flag]),
        )
        _progress("upload workflow completed successfully")
        return {
            "instance_name": instance_name,
            "copied": copied,
            "archive_root": config.remote_archive_root,
            "dataset_root": config.remote_dataset_root,
        }
    finally:
        repo_bundle.unlink(missing_ok=True)
        _progress(f"deleting upload VM {instance_name}")
        _delete_instance(config, instance_name)


def run_val_maps_upload_workflow(config: GcpRuntimeConfig) -> dict[str, Any]:
    local_harvest_dir = Path("local_map_harvest")
    tar_path = Path("validation_maps.tar")
    manifest_path = Path("data/manifests/av2_val_manifest.json")

    _progress(f"harvesting local maps from {manifest_path}")
    from scripts.harvest_local_maps import harvest_maps
    harvest_maps(str(manifest_path), str(local_harvest_dir))

    _progress(f"compressing maps into {tar_path}")
    import subprocess
    subprocess.run(["tar", "-cf", str(tar_path), str(local_harvest_dir)], check=True)

    instance_name = f"kinexar-map-upload-{_timestamp()}".replace("_", "-").lower()
    try:
        create_upload_vm(config, instance_name=instance_name)
        _progress(f"waiting for SSH on {instance_name}")
        _wait_for_ssh(config, instance_name)

        _progress(f"uploading {tar_path} to {instance_name}")
        _scp(config, tar_path, instance_name, f"/tmp/{tar_path.name}")

        _progress("extracting maps to persistent data disk")
        remote_cmd = (
            f"sudo mkdir -p /mnt/kinexar-data/maps && "
            f"sudo tar -xf /tmp/{tar_path.name} -C /mnt/kinexar-data/maps --strip-components=1 && "
            f"sudo chown -R ubuntu:ubuntu /mnt/kinexar-data/maps"
        )
        _ssh(config, instance_name, remote_cmd)

        _progress("validation map upload completed successfully")
        return {
            "instance_name": instance_name,
            "local_tar": str(tar_path),
            "remote_dir": "/mnt/kinexar-data/maps",
        }
    finally:
        if tar_path.exists():
            tar_path.unlink()
        _progress(f"deleting upload VM {instance_name}")
        _delete_instance(config, instance_name)


def run_full_workflow(
    config: GcpRuntimeConfig,
    *,
    runtime: TrainingRuntime | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    repo_bundle = _build_repo_bundle(config.repo_root)
    runtime = runtime or TrainingRuntime()
    instance_name = f"kinexar-full-{_timestamp()}".replace("_", "-").lower()
    resolved_run_name = run_name or f"kinexar_full_{_timestamp()}"
    output_root = config.output_root_for("full")
    launched_remote = False
    try:
        create_training_vm(config, instance_name=instance_name, family="full")
        _wait_for_ssh(config, instance_name)
        _stage_remote_helper(config, instance_name, repo_bundle_path=repo_bundle)
        processed_root = _resolve_remote_training_processed_root(config, instance_name, family="full")
        _ssh(
            config,
            instance_name,
            f"test -f {shlex.quote(processed_root + '/train_rows.pt')}",
        )
        _gpu_install_loop(config, instance_name)
        script = _training_service_script(
            config,
            family="full",
            run_names=[resolved_run_name],
            output_root=output_root,
            processed_root=processed_root,
            runtime=runtime,
            command_matrix=[[]],
        )
        service_name = _start_remote_service(
            config,
            instance_name,
            unit_name=f"kinexar-train-{_safe_service_name(resolved_run_name)}",
            script_name="kinexar_train_selfdelete.sh",
            script_content=script,
        )
        launched_remote = True
        return {
            "instance_name": instance_name,
            "run_name": resolved_run_name,
            "processed_root": processed_root,
            "output_root": output_root,
            "service_name": service_name,
            "log_root": _launcher_log_root(config, "full"),
        }
    finally:
        repo_bundle.unlink(missing_ok=True)
        if not launched_remote:
            _delete_instance(config, instance_name)


def run_ablation_workflow(
    config: GcpRuntimeConfig,
    request: AblationRequest,
) -> dict[str, Any]:
    repo_bundle = _build_repo_bundle(config.repo_root)
    variant_label = request.variant_label()
    instance_name = f"kinexar-ablation-{_timestamp()}".replace("_", "-").lower()
    processed_root = config.processed_root_for("ablation")
    output_root = config.output_root_for("ablation")
    run_names: list[str] = []
    launched_remote = False
    try:
        create_training_vm(config, instance_name=instance_name, family="ablation")
        _wait_for_ssh(config, instance_name)
        _stage_remote_helper(config, instance_name, repo_bundle_path=repo_bundle)
        _ssh(
            config,
            instance_name,
            f"test -f {shlex.quote(processed_root + '/train_rows.pt')}",
        )
        _gpu_install_loop(config, instance_name)
        command_matrix: list[list[str]] = []
        for seed in request.seeds:
            run_name = f"kinexar_ablation_{variant_label}_seed{int(seed)}_{_timestamp()}"
            run_names.append(run_name)
            command_matrix.append(_ablation_extra_args(request, seed))
        script = _training_service_script(
            config,
            family="ablation",
            run_names=run_names,
            output_root=output_root,
            processed_root=processed_root,
            runtime=request.runtime,
            command_matrix=command_matrix,
            aggregate_seed_output=(
                f"{output_root}/seed_summaries/{variant_label}.json"
                if len(run_names) > 1
                else None
            ),
        )
        service_name = _start_remote_service(
            config,
            instance_name,
            unit_name=f"kinexar-train-{_safe_service_name(variant_label)}",
            script_name="kinexar_train_selfdelete.sh",
            script_content=script,
        )
        launched_remote = True
        return {
            "instance_name": instance_name,
            "variant_label": variant_label,
            "run_names": run_names,
            "processed_root": processed_root,
            "output_root": output_root,
            "service_name": service_name,
            "log_root": _launcher_log_root(config, "ablation"),
        }
    finally:
        repo_bundle.unlink(missing_ok=True)
        if not launched_remote:
            _delete_instance(config, instance_name)


def _validate_docker_family(family: str) -> str:
    resolved = family.strip().lower()
    if resolved not in DOCKER_IMAGE_FAMILIES:
        raise GcpWorkflowError(
            f"unsupported docker image family: {family!r}; expected one of {DOCKER_IMAGE_FAMILIES}"
        )
    return resolved


def _docker_root(config: GcpRuntimeConfig) -> str:
    return f"{config.mount_point}/docker-images"


def _docker_map_dir(config: GcpRuntimeConfig) -> str:
    return f"{config.mount_point}/data/maps"


def _docker_image_name(family: str) -> str:
    resolved = _validate_docker_family(family)
    return f"kinexar-neurips-{resolved}:latest"


def _docker_archive_basename(family: str) -> str:
    resolved = _validate_docker_family(family)
    return f"kinexar-neurips-{resolved}.tar"


def _docker_archive_path(config: GcpRuntimeConfig, family: str) -> str:
    return f"{_docker_root(config)}/{_docker_archive_basename(family)}"


def _docker_archive_sha_path(config: GcpRuntimeConfig, family: str) -> str:
    return f"{_docker_archive_path(config, family)}.sha256"


def _docker_metadata_path(config: GcpRuntimeConfig, family: str) -> str:
    resolved = _validate_docker_family(family)
    return f"{_docker_root(config)}/kinexar-neurips-{resolved}.json"


def _docker_log_root(config: GcpRuntimeConfig, family: str) -> str:
    resolved = _validate_docker_family(family)
    return f"{config.remote_output_root}/launcher_logs/docker/{resolved}"


def _docker_log_paths(
    config: GcpRuntimeConfig,
    *,
    family: str,
    mode_label: str,
    run_names: Sequence[str],
) -> tuple[str, str]:
    resolved_family = _validate_docker_family(family)
    log_root = _docker_log_root(config, resolved_family)
    primary_name = run_names[0] if run_names else f"{resolved_family}-{mode_label}"
    return (
        f"{log_root}/{primary_name}_{mode_label}.log",
        f"{log_root}/{primary_name}_{mode_label}.status.json",
    )


def _resolve_remote_training_processed_root(
    config: GcpRuntimeConfig,
    instance_name: str,
    *,
    family: str,
) -> str:
    resolved_family = family.strip().lower()
    default_root = config.processed_root_for(resolved_family)
    return default_root


def _resolve_optional_remote_map_dir(
    config: GcpRuntimeConfig,
    instance_name: str,
    *,
    requested_map_dir: str | None,
) -> str | None:
    if requested_map_dir:
        if not _remote_path_exists(config, instance_name, requested_map_dir):
            raise GcpWorkflowError(f"requested map directory does not exist on remote disk: {requested_map_dir}")
        return requested_map_dir
    default_map_dir = _docker_map_dir(config)
    if _remote_path_exists(config, instance_name, default_map_dir):
        return default_map_dir
    return None


def _checkpoint_run_dir_from_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path).expanduser()
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def _docker_default_runtime() -> TrainingRuntime:
    return TrainingRuntime(
        batch_size=16,
        num_workers=2,
        vram_target_gb=20.0,
        vram_cap_gb=22.0,
        epochs=None,
    )


def create_docker_builder_vm(config: GcpRuntimeConfig, *, instance_name: str) -> None:
    _progress(f"ensuring data disk {config.data_disk_name} ({config.data_disk_type}, {config.data_disk_size_gb} GB)")
    ensure_data_disk(config)
    _wait_for_data_disk_free(config)
    builder_boot_disk_size_gb = max(int(config.upload_boot_disk_size_gb), 100)
    _progress(f"creating docker builder VM {instance_name} in {config.zone} with {config.upload_machine_type}")
    _create_instance(
        config,
        instance_name=instance_name,
        machine_type=config.upload_machine_type,
        boot_disk_size_gb=builder_boot_disk_size_gb,
        max_run_duration=config.upload_max_run_duration,
        labels={"app": "kinexar", "role": "docker-builder"},
    )


def _docker_build_script(config: GcpRuntimeConfig, families: Sequence[str]) -> str:
    docker_root = _docker_root(config)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "source /tmp/kinexar_remote_helper.sh",
        "kinexar_install_docker",
        f"mkdir -p {shlex.quote(docker_root)}",
        f"sudo chown -R $USER:$USER {shlex.quote(docker_root)} || true",
        f"cd {shlex.quote(config.remote_repo_root)}",
    ]
    for family in ["full"]:
        resolved_family = _validate_docker_family(family)
        image_name = _docker_image_name(resolved_family)
        archive_path = _docker_archive_path(config, resolved_family)
        archive_sha_path = _docker_archive_sha_path(config, resolved_family)
        metadata_path = _docker_metadata_path(config, resolved_family)
        lines.extend(
            [
                f"echo \"[$(date -Is)] building docker image family={resolved_family} image={image_name}\"",
                _shell_join(
                    [
                        "sudo",
                        "docker",
                        "build",
                        "--pull",
                        "--file",
                        "docker/kinexar.Dockerfile",
                        "--build-arg",
                        f"KINEXAR_DEFAULT_FAMILY={resolved_family}",
                        "--tag",
                        image_name,
                        config.remote_repo_root,
                    ]
                ),
                f"rm -f {shlex.quote(archive_path)} {shlex.quote(archive_sha_path)} {shlex.quote(metadata_path)}",
                _shell_join(
                    [
                        "sudo",
                        "docker",
                        "save",
                        "--output",
                        archive_path,
                        image_name,
                    ]
                ),
                f"sudo chown $USER:$USER {shlex.quote(archive_path)}",
                (
                    f"( cd {shlex.quote(docker_root)} && "
                    f"sha256sum {shlex.quote(Path(archive_path).name)} > {shlex.quote(Path(archive_sha_path).name)} )"
                ),
                textwrap.dedent(
                    f"""\
                    cat > {shlex.quote(metadata_path)} <<EOF
                    {{
                      "family": {json.dumps(resolved_family)},
                      "image_name": {json.dumps(image_name)},
                      "archive_path": {json.dumps(archive_path)},
                      "archive_sha256_path": {json.dumps(archive_sha_path)},
                      "built_at": "$(date -Is)"
                    }}
                    EOF
                    """
                ).strip(),
            ]
        )
    return "\n".join(lines) + "\n"


def _docker_container_command(
    config: GcpRuntimeConfig,
    *,
    family: str,
    container_args: Sequence[str],
    extra_env: dict[str, str] | None = None,
) -> str:
    resolved_family = _validate_docker_family(family)
    command = [
        "sudo",
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "-v",
        f"{config.mount_point}:{config.mount_point}",
        "-v",
        "/tmp:/tmp",
        "-e",
        f"KINEXAR_FAMILY={resolved_family}",
    ]
    if extra_env:
        for key, value in extra_env.items():
            command.extend(["-e", f"{key}={value}"])
    command.extend([_docker_image_name(resolved_family), *list(container_args)])
    return _shell_join(command)


def _docker_training_args(
    *,
    processed_root: str,
    output_root: str,
    run_name: str,
    map_dir: str | None,
    runtime: TrainingRuntime,
    visualize: bool = False,
    extra_args: Sequence[str],
) -> list[str]:
    args = [
        "--processed-root",
        processed_root,
        "--output-root",
        output_root,
        "--run-name",
        run_name,
        "--device",
        "cuda",
        "--batch-size",
        str(int(runtime.batch_size)),
        "--num-workers",
        str(int(runtime.num_workers)),
        "--vram-target-gb",
        str(float(runtime.vram_target_gb)),
        "--vram-cap-gb",
        str(float(runtime.vram_cap_gb)),
    ]
    if map_dir:
        args.extend(["--map-dir", map_dir])
    if runtime.epochs is not None:
        args.extend(["--epochs", str(int(runtime.epochs))])
    args.extend(list(extra_args))
    return args


def _docker_validation_args(
    *,
    processed_root: str,
    output_root: str,
    run_name: str,
    map_dir: str | None,
    checkpoint_path: str,
    split: str,
    runtime: TrainingRuntime,
    visualize: bool = False,
) -> list[str]:
    args = [
        "--processed-root",
        processed_root,
        "--output-root",
        output_root,
        "--run-name",
        run_name,
        "--checkpoint",
        checkpoint_path,
        "--split",
        split,
        "--device",
        "cuda",
        "--batch-size",
        str(int(runtime.batch_size)),
        "--num-workers",
        str(int(runtime.num_workers)),
    ]
    if map_dir:
        args.extend(["--map-dir", map_dir])
    return args


def _docker_service_script(
    config: GcpRuntimeConfig,
    *,
    family: str,
    mode_label: str,
    run_names: Sequence[str],
    processed_root: str,
    output_root: str,
    command_lines: Sequence[str],
    extra_status: dict[str, Any] | None = None,
) -> str:
    resolved_family = _validate_docker_family(family)
    log_root = _docker_log_root(config, resolved_family)
    log_file, status_file = _docker_log_paths(
        config,
        family=resolved_family,
        mode_label=mode_label,
        run_names=run_names,
    )
    archive_path = _docker_archive_path(config, resolved_family)
    archive_sha_path = _docker_archive_sha_path(config, resolved_family)
    workflow_body = textwrap.indent("\n".join(command_lines), "    ")
    run_names_json = json.dumps(list(run_names))
    details_json = json.dumps(extra_status or {}, indent=2, sort_keys=True)
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        OWNER_USER="$(stat -c '%U' "$0")"
        OWNER_GROUP="$(stat -c '%G' "$0")"
        export USER="${{OWNER_USER}}"
        LOG_ROOT={shlex.quote(log_root)}
        LOG_FILE={shlex.quote(log_file)}
        STATUS_FILE={shlex.quote(status_file)}
        ARCHIVE_PATH={shlex.quote(archive_path)}
        ARCHIVE_SHA_PATH={shlex.quote(archive_sha_path)}
        IMAGE_NAME={shlex.quote(_docker_image_name(resolved_family))}
        sudo mkdir -p "${{LOG_ROOT}}" {shlex.quote(output_root)}
        sudo touch "${{LOG_FILE}}"
        sudo rm -f "${{STATUS_FILE}}"
        sudo chown "${{OWNER_USER}}:${{OWNER_GROUP}}" "${{LOG_ROOT}}" "${{LOG_FILE}}" {shlex.quote(output_root)}
        sudo rm -f "${{STATUS_FILE}}" "${{LOG_FILE}}"
        : > "${{LOG_FILE}}"
        exec > >(tee -a "${{LOG_FILE}}") 2>&1
        trap 'finalize $?' EXIT

        meta() {{
          curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/$1"
        }}

        delete_self() {{
          local token project zone instance url
          token="$(python3 - <<'PY'
import json, urllib.request
req = urllib.request.Request(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    headers={{"Metadata-Flavor": "Google"}},
)
print(json.load(urllib.request.urlopen(req))["access_token"])
PY
)"
          project="$(meta project/project-id)"
          zone="$(basename "$(meta instance/zone)")"
          instance="$(meta instance/name)"
          url="https://compute.googleapis.com/compute/v1/projects/${{project}}/zones/${{zone}}/instances/${{instance}}"
          echo "[$(date -Is)] requesting self-delete via ${{url}}"
          curl -fsS -X DELETE -H "Authorization: Bearer ${{token}}" "${{url}}"
        }}

        finalize() {{
          local exit_code="${{1}}"
          local finished_at
          finished_at="$(date -Is)"
          cat > "${{STATUS_FILE}}" <<JSON
{{
  "family": {json.dumps(resolved_family)},
  "mode": {json.dumps(mode_label)},
  "run_names": {run_names_json},
  "processed_root": {json.dumps(processed_root)},
  "output_root": {json.dumps(output_root)},
  "image_name": "${{IMAGE_NAME}}",
  "image_archive_path": "${{ARCHIVE_PATH}}",
  "started_at": "${{STARTED_AT}}",
  "finished_at": "${{finished_at}}",
  "log_file": "${{LOG_FILE}}",
  "exit_code": ${{exit_code}},
  "details": {details_json}
}}
JSON
          if delete_self; then
            echo "[$(date -Is)] self-delete request accepted"
            return "${{exit_code}}"
          fi
          echo "[$(date -Is)] self-delete request failed; skipping shutdown to allow debugging." >&2
          return "${{exit_code}}"
        }}

        echo "[$(date -Is)] starting docker workflow family={resolved_family} mode={mode_label}"
        source /tmp/kinexar_remote_helper.sh
        kinexar_mount_data_disk
        kinexar_install_docker
        kinexar_install_nvidia_container_toolkit
        if [ ! -f "${{ARCHIVE_PATH}}" ]; then
          echo "missing docker image archive: ${{ARCHIVE_PATH}}" >&2
          exit 1
        fi
        if [ -f "${{ARCHIVE_SHA_PATH}}" ]; then
          (
            cd "$(dirname "${{ARCHIVE_PATH}}")"
            sha256sum -c "$(basename "${{ARCHIVE_SHA_PATH}}")"
          )
        fi
        sudo docker load -i "${{ARCHIVE_PATH}}"
        STARTED_AT="$(date -Is)"
        set +e
        (
          set -e
{workflow_body}
        )
        exit_code=$?
        set -e
        """
    )


def _wait_for_remote_log_signal(
    config: GcpRuntimeConfig,
    instance_name: str,
    *,
    log_file: str,
    status_file: str,
    success_patterns: Sequence[str],
    timeout_s: int | None = None,
    tail_lines: int = 120,
) -> dict[str, Any]:
    timeout = int(timeout_s or config.ssh_ready_timeout_s)
    deadline = time.time() + float(timeout)
    last_excerpt = ""
    while time.time() < deadline:
        probe_cmd = textwrap.dedent(
            f"""\
            set -euo pipefail
            if [ -f {shlex.quote(status_file)} ]; then
              echo '__KINEXAR_STATUS_BEGIN__'
              cat {shlex.quote(status_file)}
              echo '__KINEXAR_STATUS_END__'
            fi
            if [ -f {shlex.quote(log_file)} ]; then
              echo '__KINEXAR_LOG_BEGIN__'
              tail -n {int(tail_lines)} {shlex.quote(log_file)}
              echo '__KINEXAR_LOG_END__'
            fi
            """
        ).strip()
        proc = _ssh(config, instance_name, "bash -lc " + shlex.quote(probe_cmd), check=False, capture_output=True)
        stdout = proc.stdout or ""
        status_text = ""
        log_text = ""
        if "__KINEXAR_STATUS_BEGIN__" in stdout and "__KINEXAR_STATUS_END__" in stdout:
            status_text = stdout.split("__KINEXAR_STATUS_BEGIN__", 1)[1].split("__KINEXAR_STATUS_END__", 1)[0].strip()
        if "__KINEXAR_LOG_BEGIN__" in stdout and "__KINEXAR_LOG_END__" in stdout:
            log_text = stdout.split("__KINEXAR_LOG_BEGIN__", 1)[1].split("__KINEXAR_LOG_END__", 1)[0].strip()
            last_excerpt = log_text
        if status_text:
            status_payload = json.loads(status_text)
            exit_code = status_payload.get("exit_code")
            if exit_code is not None and int(exit_code) != 0:
                excerpt = log_text or last_excerpt or "(remote log not available)"
                raise GcpWorkflowError(
                    f"remote job failed before startup verification; exit_code={exit_code}\n{excerpt}"
                )
        for pattern in success_patterns:
            if pattern in log_text:
                return {
                    "matched_pattern": pattern,
                    "log_file": log_file,
                    "status_file": status_file,
                    "log_excerpt": log_text,
                }
        time.sleep(float(config.ssh_poll_interval_s))
    excerpt = last_excerpt or "(remote log not available yet)"
    raise GcpWorkflowError(
        f"timed out waiting for remote job startup signal in {log_file}\n{excerpt}"
    )


def _clear_remote_launcher_artifacts(
    config: GcpRuntimeConfig,
    instance_name: str,
    *,
    log_file: str,
    status_file: str,
) -> None:
    _ssh(
        config,
        instance_name,
        textwrap.dedent(
            f"""\
            sudo mkdir -p {shlex.quote(str(Path(log_file).parent))} {shlex.quote(str(Path(status_file).parent))}
            sudo rm -f {shlex.quote(log_file)} {shlex.quote(status_file)}
            sudo chown -R "$USER:$USER" {shlex.quote(str(Path(log_file).parent))}
            """
        ).strip(),
    )


def run_docker_image_build_workflow(
    config: GcpRuntimeConfig,
    *,
    families: Sequence[str] = DOCKER_IMAGE_FAMILIES,
) -> dict[str, Any]:
    resolved_families = tuple(_validate_docker_family(family) for family in families)
    repo_bundle = _build_repo_bundle(config.repo_root)
    instance_name = f"kinexar-docker-build-{_timestamp()}".replace("_", "-").lower()
    try:
        create_docker_builder_vm(config, instance_name=instance_name)
        _wait_for_ssh(config, instance_name)
        _stage_remote_helper(config, instance_name, repo_bundle_path=repo_bundle)
        _progress(f"building docker images on {instance_name} into {_docker_root(config)}")
        _run_remote_script(config, instance_name, _docker_build_script(config, resolved_families))
        return {
            "instance_name": instance_name,
            "docker_root": _docker_root(config),
            "images": [
                {
                    "family": family,
                    "image_name": _docker_image_name(family),
                    "archive_path": _docker_archive_path(config, family),
                    "archive_sha256_path": _docker_archive_sha_path(config, family),
                    "metadata_path": _docker_metadata_path(config, family),
                }
                for family in resolved_families
            ],
        }
    except Exception as e:
        print(f"[gcp] ERROR: {e}")
        print(f"[gcp] VM preserved for debugging: {instance_name}")
        raise
    else:
        _delete_instance(config, instance_name)


def run_docker_full_workflow(
    config: GcpRuntimeConfig,
    *,
    runtime: TrainingRuntime | None = None,
    run_name: str | None = None,
    map_dir: str | None = None,
    visualize: bool = False,
    suite: str | None = None,
) -> dict[str, Any]:
    runtime = runtime or _docker_default_runtime()
    instance_name = f"kinexar-docker-full-{_timestamp()}".replace("_", "-").lower()
    resolved_run_name = run_name or f"kinexar_full_docker_{_timestamp()}"
    resolved_family = "full"
    output_root = config.output_root_for(resolved_family)
    validation_root = f"{config.remote_dataset_root}/full_val"
    requested_map_dir = map_dir
    log_file, status_file = _docker_log_paths(
        config,
        family=resolved_family,
        mode_label="train",
        run_names=[resolved_run_name],
    )
    launched_remote = False
    try:
        create_training_vm(config, instance_name=instance_name, family=resolved_family)
        _wait_for_ssh(config, instance_name)
        _stage_remote_helper(config, instance_name)
        processed_root = _resolve_remote_training_processed_root(config, instance_name, family=resolved_family)
        resolved_map_dir = _resolve_optional_remote_map_dir(
            config,
            instance_name,
            requested_map_dir=requested_map_dir,
        )
        _ssh(config, instance_name, f"test -f {shlex.quote(processed_root + '/train_rows.pt')}")
        _ssh(
            config,
            instance_name,
            textwrap.dedent(
                f"""\
                if [ -f {shlex.quote(processed_root + '/val_rows.pt')} ] || \
                   [ -f {shlex.quote(validation_root + '/val_rows.pt')} ] || \
                   ( [ -f {shlex.quote(validation_root + '/bundle_manifest.json')} ] && [ -d {shlex.quote(validation_root + '/val_parts')} ] ); then
                  exit 0
                fi
                echo "missing validation rows bundle under {validation_root}" >&2
                exit 1
                """
            ).strip(),
        )
        _ssh(config, instance_name, f"test -f {shlex.quote(_docker_archive_path(config, resolved_family))}")
        _gpu_install_loop(config, instance_name)
        _clear_remote_launcher_artifacts(
            config,
            instance_name,
            log_file=log_file,
            status_file=status_file,
        )
        command_lines = [
            _docker_container_command(
                config,
                family=resolved_family,
                extra_env={"KINEXAR_VAL_PROCESSED_ROOT": validation_root},
                container_args=[
                    "train",
                    *_docker_training_args(
                        processed_root=processed_root,
                        output_root=output_root,
                        run_name=resolved_run_name,
                        map_dir=resolved_map_dir,
                        runtime=runtime,
                        extra_args=(["--suite", str(suite)] if suite else []),
                    ),
                ],
            ),
            f"sudo chown -R $USER:$USER {shlex.quote(f'{output_root}/{resolved_run_name}')}",
        ]
        script = _docker_service_script(
            config,
            family=resolved_family,
            mode_label="train",
            run_names=[resolved_run_name],
            processed_root=processed_root,
            output_root=output_root,
            command_lines=command_lines,
            extra_status={"map_dir": resolved_map_dir},
        )
        service_name = _start_remote_service(
            config,
            instance_name,
            unit_name=f"kinexar-docker-{_safe_service_name(resolved_run_name)}",
            script_name="kinexar_docker_job.sh",
            script_content=script,
        )
        startup = _wait_for_remote_log_signal(
            config,
            instance_name,
            log_file=log_file,
            status_file=status_file,
            success_patterns=(
                "ready_to_train=",
                "epoch=1",
                "training_complete",
                "suite_start",
                "suite_train_start",
                "suite_complete",
            ),
        )
        excerpt = startup.get("log_excerpt", "").strip()
        if excerpt:
            _progress(f"remote training log tail from {log_file}:\n{excerpt}")
        launched_remote = True
        return {
            "instance_name": instance_name,
            "family": resolved_family,
            "mode": "train",
            "run_name": resolved_run_name,
            "suite": suite,
            "processed_root": processed_root,
            "output_root": output_root,
            "map_dir": resolved_map_dir,
            "image_archive_path": _docker_archive_path(config, resolved_family),
            "service_name": service_name,
            "log_root": _docker_log_root(config, resolved_family),
            "log_file": log_file,
            "status_file": status_file,
            "startup_verified": True,
            "startup_signal": startup["matched_pattern"],
        }
    finally:
        if not launched_remote:
            _delete_instance(config, instance_name)


def run_docker_ablation_workflow(
    config: GcpRuntimeConfig,
    request: AblationRequest,
    *,
    map_dir: str | None = None,
    visualize: bool = False,
) -> dict[str, Any]:
    runtime = request.runtime or _docker_default_runtime()
    variant_label = request.variant_label()
    resolved_family = "ablation"
    instance_name = f"kinexar-docker-ablation-{_timestamp()}".replace("_", "-").lower()
    processed_root = config.processed_root_for(resolved_family)
    output_root = config.output_root_for(resolved_family)
    resolved_map_dir = map_dir or _docker_map_dir(config)
    run_names: list[str] = []
    command_lines: list[str] = []
    for seed in request.seeds:
        run_name = f"kinexar_ablation_{variant_label}_seed{int(seed)}_{_timestamp()}"
        run_names.append(run_name)
        command_lines.append(
            _docker_container_command(
                config,
                family=resolved_family,
                container_args=[
                    "train",
                    *_docker_training_args(
                        processed_root=processed_root,
                        output_root=output_root,
                        run_name=run_name,
                        map_dir=resolved_map_dir,
                        runtime=runtime,
                        extra_args=_ablation_extra_args(request, seed),
                    ),
                ],
            )
        )
        command_lines.append(f"sudo chown -R $USER:$USER {shlex.quote(f'{output_root}/{run_name}')}")
    aggregate_output = None
    if len(run_names) > 1:
        aggregate_output = f"{output_root}/seed_summaries/{variant_label}.json"
        aggregate_args: list[str] = []
        for run_name in run_names:
            aggregate_args.extend(["--run-dir", f"{output_root}/{run_name}"])
        aggregate_command = _shell_join(
            [
                "python",
                "scripts/aggregate_seed_results.py",
                *aggregate_args,
                "--output",
                aggregate_output,
            ]
        )
        command_lines.append(
            _docker_container_command(
                config,
                family=resolved_family,
                container_args=["bash", "-lc", aggregate_command],
            )
        )
        command_lines.append(f"sudo chown -R $USER:$USER {shlex.quote(f'{output_root}/seed_summaries')}")
    script = _docker_service_script(
        config,
        family=resolved_family,
        mode_label="train",
        run_names=run_names,
        processed_root=processed_root,
        output_root=output_root,
        command_lines=command_lines,
        extra_status={
            "map_dir": resolved_map_dir,
            "variant_label": variant_label,
            "aggregate_output": aggregate_output,
        },
    )
    launched_remote = False
    try:
        create_training_vm(config, instance_name=instance_name, family=resolved_family)
        _wait_for_ssh(config, instance_name)
        _stage_remote_helper(config, instance_name)
        _ssh(config, instance_name, f"test -f {shlex.quote(processed_root + '/train_rows.pt')}")
        _ssh(config, instance_name, f"test -f {shlex.quote(_docker_archive_path(config, resolved_family))}")
        _gpu_install_loop(config, instance_name)
        _clear_remote_launcher_artifacts(
            config,
            instance_name,
            log_file=log_file,
            status_file=status_file,
        )
        service_name = _start_remote_service(
            config,
            instance_name,
            unit_name=f"kinexar-docker-{_safe_service_name(variant_label)}",
            script_name="kinexar_docker_job.sh",
            script_content=script,
        )
        launched_remote = True
        return {
            "instance_name": instance_name,
            "family": resolved_family,
            "mode": "train",
            "variant_label": variant_label,
            "run_names": run_names,
            "processed_root": processed_root,
            "output_root": output_root,
            "map_dir": resolved_map_dir,
            "image_archive_path": _docker_archive_path(config, resolved_family),
            "service_name": service_name,
            "log_root": _docker_log_root(config, resolved_family),
            "aggregate_output": aggregate_output,
        }
    finally:
        if not launched_remote:
            _delete_instance(config, instance_name)


def run_docker_validation_workflow(
    config: GcpRuntimeConfig,
    *,
    family: str,
    checkpoint_path: str,
    split: str = "val",
    processed_root: str | None = None,
    output_root: str | None = None,
    run_name: str | None = None,
    runtime: TrainingRuntime | None = None,
    map_dir: str | None = None,
    visualize: bool = False,
) -> dict[str, Any]:
    runtime = runtime or _docker_default_runtime()
    resolved_family = _validate_docker_family(family)
    checkpoint = str(Path(checkpoint_path).expanduser())
    inferred_run_dir = _checkpoint_run_dir_from_path(checkpoint)
    resolved_output_root = output_root or str(inferred_run_dir.parent)
    resolved_run_name = run_name or inferred_run_dir.name
    resolved_processed_root = processed_root or f"{config.remote_dataset_root}/full_val"
    resolved_map_dir = map_dir or _docker_map_dir(config)
    instance_name = f"kinexar-docker-validate-{_timestamp()}".replace("_", "-").lower()
    command_lines = [
        _docker_container_command(
            config,
            family=resolved_family,
            container_args=[
                "validate",
                *_docker_validation_args(
                    processed_root=resolved_processed_root,
                    output_root=resolved_output_root,
                    run_name=resolved_run_name,
                    map_dir=resolved_map_dir,
                    checkpoint_path=checkpoint,
                    split=split,
                    runtime=runtime,
                    visualize=visualize,
                ),
            ],
        ),
        f"sudo chown -R $USER:$USER {shlex.quote(f'{resolved_output_root}/{resolved_run_name}')}",
    ]
    script = _docker_service_script(
        config,
        family=resolved_family,
        mode_label="validate",
        run_names=[resolved_run_name],
        processed_root=resolved_processed_root,
        output_root=resolved_output_root,
        command_lines=command_lines,
        extra_status={
            "checkpoint_path": checkpoint,
            "split": split,
            "map_dir": resolved_map_dir,
        },
    )
    launched_remote = False
    try:
        create_training_vm(config, instance_name=instance_name, family=resolved_family)
        _wait_for_ssh(config, instance_name)
        _stage_remote_helper(config, instance_name)
        processed_root_check = textwrap.dedent(
            f"""\
            if [ -f {shlex.quote(resolved_processed_root + '/val_rows.pt')} ] || \
               ( [ -f {shlex.quote(resolved_processed_root + '/bundle_manifest.json')} ] && [ -d {shlex.quote(resolved_processed_root + '/val_parts')} ] ) || \
               [ -f {shlex.quote(resolved_processed_root + '/train_rows.pt')} ] || \
               [ -f {shlex.quote(resolved_processed_root + '/progress.sqlite3')} ]; then
              exit 0
            fi
            echo "missing compatible processed root at {resolved_processed_root}" >&2
            exit 1
            """
        ).strip()
        _ssh(config, instance_name, processed_root_check)
        _ssh(config, instance_name, f"test -f {shlex.quote(_docker_archive_path(config, resolved_family))}")
        _ssh(config, instance_name, f"test -f {shlex.quote(checkpoint)}")
        _gpu_install_loop(config, instance_name)
        _clear_remote_launcher_artifacts(
            config,
            instance_name,
            log_file=log_file,
            status_file=status_file,
        )
        service_name = _start_remote_service(
            config,
            instance_name,
            unit_name=f"kinexar-docker-validate-{_safe_service_name(resolved_run_name)}",
            script_name="kinexar_docker_job.sh",
            script_content=script,
        )
        launched_remote = True
        return {
            "instance_name": instance_name,
            "family": resolved_family,
            "mode": "validate",
            "run_name": resolved_run_name,
            "processed_root": resolved_processed_root,
            "output_root": resolved_output_root,
            "checkpoint_path": checkpoint,
            "split": split,
            "map_dir": resolved_map_dir,
            "image_archive_path": _docker_archive_path(config, resolved_family),
            "service_name": service_name,
            "log_root": _docker_log_root(config, resolved_family),
        }
    finally:
        if not launched_remote:
            _delete_instance(config, instance_name)
