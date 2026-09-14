from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from kinexar.training.utils import load_rows_bundle_dataset
from kinexar.training.utils import _dedupe_bundle_row_aliases
from kinexar.training.utils import _existing_parts
from kinexar.training.utils import write_multipart_rows_bundle
from kinexar.training.utils import write_jsonl_atomic

from kinexar.training.profiles import dataset_root_for
from kinexar.training.profiles import resolve_profile
from kinexar.dataset.data.dataset import TrajectoryDataset


FULL_ARCHIVE = "full.tar.zst"
PARTS_ARCHIVE = "full_train_parts.tar.zst"
RAW_BUNDLE_WORKERS_DEFAULT = 9
RAW_BUNDLE_RESERVE_BYTES = 2 * 1024 * 1024 * 1024

_RAW_BUNDLE_DATASET: TrajectoryDataset | None = None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _init_raw_bundle_worker(manifest_path: str) -> None:
    global _RAW_BUNDLE_DATASET
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _RAW_BUNDLE_DATASET = TrajectoryDataset(manifest_path, "av2")


def _materialize_raw_bundle_row(meta: dict[str, Any]) -> bytes:
    if _RAW_BUNDLE_DATASET is None:
        raise RuntimeError("raw bundle worker dataset not initialized")
    row = dict(_RAW_BUNDLE_DATASET[int(meta["dataset_index"])])
    row.update(dict(meta))
    return pickle.dumps(_dedupe_bundle_row_aliases(row), protocol=pickle.HIGHEST_PROTOCOL)


def _relativize_rows_manifest_paths(manifest_path: Path) -> None:
    payload = torch.load(manifest_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected rows bundle manifest payload: {manifest_path}")
    parts = payload.get("parts", [])
    if not isinstance(parts, list):
        raise ValueError(f"bad rows bundle manifest parts: {manifest_path}")
    base_dir = manifest_path.parent.resolve()
    changed = False
    for part in parts:
        if not isinstance(part, dict):
            continue
        for key in ("path", "offsets_path", "recovery_meta_path"):
            value = part.get(key)
            if not value:
                continue
            raw_path = Path(str(value)).expanduser()
            if not raw_path.is_absolute():
                continue
            path = raw_path.resolve()
            try:
                relative = path.relative_to(base_dir)
            except ValueError:
                raise ValueError(f"cannot relativize bundle path outside manifest root: {path} (manifest={manifest_path})") from None
            part[key] = str(relative)
            changed = True
    if changed:
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(manifest_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_expected_sha(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"empty checksum file: {path}")
    return raw.split()[0]


def verify_archives(archive_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in (FULL_ARCHIVE, PARTS_ARCHIVE):
        archive_path = archive_root / name
        sha_path = archive_root / f"{name}.sha256"
        if not archive_path.exists():
            raise FileNotFoundError(f"missing archive: {archive_path}")
        if not sha_path.exists():
            raise FileNotFoundError(f"missing checksum: {sha_path}")
        expected = _read_expected_sha(sha_path)
        actual = _sha256(archive_path)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {archive_path}: expected {expected}, got {actual}")
        out[name] = {
            "path": str(archive_path),
            "sha256": actual,
            "bytes": archive_path.stat().st_size,
        }
    return out


def _extract_archive(archive_path: Path, dest_root: Path, members: list[str] | None = None) -> None:
    cmd = [
        "tar",
        "-C",
        str(dest_root),
        "-I",
        "zstd -d -T0",
        "-xf",
        str(archive_path),
    ]
    if members:
        cmd.extend(members)
    subprocess.run(cmd, check=True)


def _rewrite_manifest_paths(manifest_path: Path, full_root: Path, parts_root: Path) -> None:
    payload = torch.load(manifest_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected manifest payload: {manifest_path}")
    parts = payload.get("parts", [])
    if not isinstance(parts, list):
        raise ValueError(f"bad parts manifest: {manifest_path}")
    changed = False
    for part in parts:
        if not isinstance(part, dict):
            continue
        for key in ("path", "offsets_path", "recovery_meta_path"):
            value = part.get(key)
            if not value:
                continue
            old_path = Path(str(value))
            if old_path.parent.name == "full":
                new_path = full_root / old_path.name
            elif old_path.parent.name == "full_train_parts":
                new_path = parts_root / old_path.name
            else:
                continue
            if str(new_path) != str(value):
                part[key] = str(new_path)
                changed = True
    if changed:
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(manifest_path)


def prepare_full_bundle(archive_root: Path, dataset_root: Path, force: bool = False) -> dict[str, Any]:
    verify_archives(archive_root)
    full_root = dataset_root / "full"
    parts_root = dataset_root / "full_train_parts"
    if force and dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    if force or not full_root.exists():
        _extract_archive(archive_root / FULL_ARCHIVE, dataset_root)
    if force or not parts_root.exists():
        _extract_archive(archive_root / PARTS_ARCHIVE, dataset_root)
    manifest_path = full_root / "train_rows.pt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing extracted train_rows.pt: {manifest_path}")
    _rewrite_manifest_paths(manifest_path, full_root, parts_root)
    result = {
        "dataset_root": str(dataset_root),
        "processed_root": str(full_root),
        "manifest_path": str(manifest_path),
        "parts_root": str(parts_root),
    }
    _write_json(dataset_root / "preparation_manifest.json", result)
    return result


def _select_subset_metadata(
    rows: list[dict[str, Any]],
    train_size: int,
    val_size: int,
    debug_size: int,
    seed: int,
    val_scenario_disjoint: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import random

    enriched: list[dict[str, Any]] = []
    for bundle_index, row in enumerate(rows):
        item = dict(row)
        item["bundle_index"] = bundle_index
        enriched.append(item)
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in enriched:
        by_scenario.setdefault(str(row["scenario_id"]), []).append(row)
    scenario_ids = sorted(by_scenario)
    rng = random.Random(seed)
    rng.shuffle(scenario_ids)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    used_scenarios: set[str] = set()
    for scenario_id in scenario_ids:
        current = list(by_scenario[scenario_id])
        rng.shuffle(current)
        if len(train) < train_size:
            remaining = train_size - len(train)
            take = current[:remaining]
            train.extend(take)
            if take:
                used_scenarios.add(scenario_id)
            continue
        if val_scenario_disjoint and scenario_id in used_scenarios:
            continue
        if len(val) < val_size:
            remaining = val_size - len(val)
            val.extend(current[:remaining])
        if len(train) >= train_size and len(val) >= val_size:
            break
    if len(train) < train_size:
        raise ValueError(f"insufficient train rows: need {train_size}, found {len(train)}")
    if len(val) < val_size:
        raise ValueError(f"insufficient val rows: need {val_size}, found {len(val)}")

    def _renumber(split_rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for experiment_index, row in enumerate(split_rows):
            item = dict(row)
            item["split"] = split_name
            item["experiment_index"] = experiment_index
            item["source_bundle_index"] = int(row["bundle_index"])
            out.append(item)
        return out

    rng.shuffle(train)
    debug = train[: min(debug_size, len(train))]
    return _renumber(train, "train"), _renumber(val, "val"), _renumber(debug, "debug")


def materialize_subset_bundle(
    archive_root: Path,
    dataset_root: Path,
    *,
    train_size: int,
    val_size: int,
    debug_size: int,
    seed: int,
    val_scenario_disjoint: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    verify_archives(archive_root)
    if force and dataset_root.exists():
        shutil.rmtree(dataset_root)
    bundle_root = dataset_root / "full"
    parts_root = dataset_root / "full_train_parts"
    if bundle_root.exists() and not force and (bundle_root / "train_rows.pt").exists():
        return {
            "dataset_root": str(dataset_root),
            "processed_root": str(bundle_root),
            "status": "reused",
        }
    dataset_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)
    parts_root.mkdir(parents=True, exist_ok=True)
    temp_parent = dataset_root.parent if dataset_root.parent.exists() else None
    with tempfile.TemporaryDirectory(prefix="kinexar_subset_", dir=temp_parent) as tmpdir:
        tmpdir_path = Path(tmpdir)
        meta_root = tmpdir_path / "meta"
        meta_root.mkdir(parents=True, exist_ok=True)
        _extract_archive(
            archive_root / FULL_ARCHIVE,
            meta_root,
            members=["full/train_rows.pt", "full/train.json"],
        )
        full_meta_root = meta_root / "full"
        train_rows_manifest = torch.load(full_meta_root / "train_rows.pt", map_location="cpu")
        if not isinstance(train_rows_manifest, dict):
            raise ValueError("expected multipart train_rows.pt manifest")
        all_train_meta = json.loads((full_meta_root / "train.json").read_text(encoding="utf-8"))
        train_meta, val_meta, debug_meta = _select_subset_metadata(
            all_train_meta,
            train_size=train_size,
            val_size=val_size,
            debug_size=debug_size,
            seed=seed,
            val_scenario_disjoint=val_scenario_disjoint,
        )
        parts = [dict(part) for part in train_rows_manifest.get("parts", [])]
        if not parts:
            raise ValueError("multipart train_rows.pt manifest has no parts")
        offsets: list[int] = []
        total = 0
        for part in parts:
            offsets.append(total)
            total += int(part["row_count"])

        def _resolve_part_index(index: int) -> int:
            lo = 0
            hi = len(offsets) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                start = offsets[mid]
                end = start + int(parts[mid]["row_count"])
                if start <= index < end:
                    return mid
                if index < start:
                    hi = mid - 1
                else:
                    lo = mid + 1
            raise IndexError(index)

        def _member_name(raw_path: str) -> tuple[str, str]:
            old_path = Path(str(raw_path))
            root_name = old_path.parent.name
            if root_name not in {"full", "full_train_parts"}:
                raise ValueError(f"unsupported bundle member root: {old_path}")
            return root_name, f"{root_name}/{old_path.name}"

        needed_members: dict[str, set[str]] = {"full": set(), "full_train_parts": set()}
        needed_part_indices: set[int] = set()
        for meta in train_meta + val_meta + debug_meta:
            needed_part_indices.add(_resolve_part_index(int(meta["bundle_index"])))
        for part_index in needed_part_indices:
            part = parts[part_index]
            for key in ("path", "offsets_path", "recovery_meta_path"):
                value = part.get(key)
                if not value:
                    continue
                root_name, member = _member_name(str(value))
                needed_members[root_name].add(member)

        extract_root = tmpdir_path / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)
        if needed_members["full"]:
            _extract_archive(archive_root / FULL_ARCHIVE, extract_root, members=sorted(needed_members["full"]))
        if needed_members["full_train_parts"]:
            _extract_archive(archive_root / PARTS_ARCHIVE, extract_root, members=sorted(needed_members["full_train_parts"]))

        rewritten_manifest = json.loads(json.dumps(train_rows_manifest))
        for part in rewritten_manifest.get("parts", []):
            for key in ("path", "offsets_path", "recovery_meta_path"):
                value = part.get(key)
                if not value:
                    continue
                root_name, member = _member_name(str(value))
                part[key] = str(extract_root / member)
        temp_manifest_path = tmpdir_path / "rewritten_train_rows.pt"
        torch.save(rewritten_manifest, temp_manifest_path)
        rows_dataset = load_rows_bundle_dataset(temp_manifest_path)

        def _materialize_rows(metadata_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for meta in metadata_rows:
                row = dict(rows_dataset[int(meta["bundle_index"])])
                row.update(dict(meta))
                rows.append(row)
            return rows

        train_rows = _materialize_rows(train_meta)
        val_rows = _materialize_rows(val_meta)
        debug_rows = _materialize_rows(debug_meta)
        torch.save(train_rows, bundle_root / "train_rows.pt")
        torch.save(val_rows, bundle_root / "val_rows.pt")
        torch.save(debug_rows, bundle_root / "debug_rows.pt")
        _write_json(bundle_root / "train.json", train_meta)
        _write_json(bundle_root / "val.json", val_meta)
        _write_json(bundle_root / "debug.json", debug_meta)
    result = {
        "dataset_root": str(dataset_root),
        "processed_root": str(bundle_root),
        "train_rows": train_size,
        "val_rows": val_size,
        "debug_rows": debug_size,
        "seed": seed,
    }
    _write_json(dataset_root / "preparation_manifest.json", result)
    return result


def _raw_manifest_metadata(dataset: TrajectoryDataset, split: str) -> list[dict[str, Any]]:
    if getattr(dataset, "mode", "") != "raw":
        raise ValueError("raw bundle build requires a raw manifest dataset")
    metadata: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        row = dataset.get(index)
        metadata.append(
            {
                "split": split,
                "experiment_index": index,
                "dataset_index": index,
                "sample_id": f'{row["scenario_id"]}:{int(row["start_t"]):06d}',
                "scenario_id": str(row["scenario_id"]),
                "file_path": str(row["file_path"]),
                "start_t": int(row["start_t"]),
                "output_shard": None,
            }
        )
    return metadata


def _raw_bundle_progress_row(meta: dict[str, Any], split: str, fallback_index: int) -> dict[str, Any]:
    return {
        "split": str(meta.get("split", split)),
        "experiment_index": int(meta.get("experiment_index", fallback_index)),
        "dataset_index": int(meta["dataset_index"]),
        "sample_id": str(meta["sample_id"]),
        "scenario_id": str(meta["scenario_id"]),
        "file_path": str(meta["file_path"]),
        "start_t": int(meta["start_t"]),
        "output_shard": meta.get("output_shard"),
    }


def _write_raw_bundle_progress(metadata_rows: list[dict[str, Any]], split: str, output_root: Path, done_count: int) -> None:
    completed_rows_log = output_root / f"{split}_rows.completed.jsonl"
    remaining_rows_log = output_root / f"{split}_rows.remaining.jsonl"
    completed_rows = [_raw_bundle_progress_row(meta, split, index) for index, meta in enumerate(metadata_rows[:done_count])]
    remaining_rows = [
        _raw_bundle_progress_row(meta, split, done_count + index)
        for index, meta in enumerate(metadata_rows[done_count:])
    ]
    write_jsonl_atomic(completed_rows_log, completed_rows)
    write_jsonl_atomic(remaining_rows_log, remaining_rows)


def _write_raw_bundle_manifest(
    manifest_path: Path,
    split: str,
    chunk_rows: int,
    parts: list[dict[str, Any]],
    estimated_row_bytes: float,
) -> dict[str, Any]:
    payload = {
        "format": "roadweaver.rows.bundle.v1",
        "split": split,
        "total_rows": sum(int(part["row_count"]) for part in parts),
        "part_count": len(parts),
        "chunk_rows": int(chunk_rows),
        "chunk_multiple": 64,
        "estimated_row_bytes": float(estimated_row_bytes),
        "parts": parts,
    }
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    torch.save(payload, tmp_manifest)
    tmp_manifest.replace(manifest_path)
    return payload


def _write_raw_bundle_part(
    row_blobs: list[bytes],
    *,
    part_index: int,
    split: str,
    parts_root: Path,
) -> dict[str, Any]:
    part_path = parts_root / f"{split}_rows.part_{part_index:05d}.pt"
    tmp_part = part_path.with_suffix(part_path.suffix + ".tmp")
    offsets_path = parts_root / f"{split}_rows.part_{part_index:05d}.offsets.pt"
    tmp_offsets = offsets_path.with_suffix(offsets_path.suffix + ".tmp")
    tmp_part.unlink(missing_ok=True)
    tmp_offsets.unlink(missing_ok=True)
    offsets: list[int] = []
    with tmp_part.open("wb") as handle:
        for blob in row_blobs:
            offsets.append(handle.tell())
            handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    torch.save(offsets, tmp_offsets)
    tmp_part.replace(part_path)
    tmp_offsets.replace(offsets_path)
    return {
        "index": part_index,
        "path": str(part_path),
        "offsets_path": str(offsets_path),
        "serialization": "pickle_stream",
        "row_count": len(row_blobs),
        "bytes": int(part_path.stat().st_size),
    }


def _build_raw_bundle_parallel(
    manifest_path: Path,
    metadata_rows: list[dict[str, Any]],
    *,
    output_root: Path,
    parts_root: Path,
    manifest_out: Path,
    split: str,
    chunk_rows: int,
    worker_count: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    parts = _existing_parts(split, [parts_root])
    completed_rows = sum(int(part["row_count"]) for part in parts)
    total_rows = len(metadata_rows)
    if completed_rows > total_rows:
        raise ValueError(f"existing parts exceed available metadata rows for split={split}: {completed_rows} > {total_rows}")
    _write_raw_bundle_progress(metadata_rows, split, output_root, completed_rows)

    free_bytes = shutil.disk_usage(parts_root).free
    if free_bytes <= RAW_BUNDLE_RESERVE_BYTES:
        raise RuntimeError(f"insufficient free space for raw bundle at {parts_root}")

    estimate_count = min(64, max(0, total_rows - completed_rows))
    estimated_row_bytes = 0.0
    if estimate_count:
        probe_dataset = TrajectoryDataset(str(manifest_path), "av2")
        estimate_blobs: list[bytes] = []
        for meta in metadata_rows[completed_rows : completed_rows + estimate_count]:
            row = dict(probe_dataset[int(meta["dataset_index"])])
            row.update(dict(meta))
            estimate_blobs.append(pickle.dumps(_dedupe_bundle_row_aliases(row), protocol=pickle.HIGHEST_PROTOCOL))
        if estimate_blobs:
            estimated_row_bytes = float(sum(len(blob) for blob in estimate_blobs)) / float(len(estimate_blobs))

    if parts:
        _write_raw_bundle_manifest(manifest_out, split, chunk_rows, parts, estimated_row_bytes)

    if completed_rows >= total_rows:
        return _write_raw_bundle_manifest(manifest_out, split, chunk_rows, parts, estimated_row_bytes)

    map_chunksize = max(1, min(64, chunk_rows // max(worker_count * 4, 1)))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=ctx,
        initializer=_init_raw_bundle_worker,
        initargs=(str(manifest_path),),
    ) as pool:
        while completed_rows < total_rows:
            target_rows = min(chunk_rows, total_rows - completed_rows)
            current_meta = metadata_rows[completed_rows : completed_rows + target_rows]
            row_blobs = list(pool.map(_materialize_raw_bundle_row, current_meta, chunksize=map_chunksize))
            part_record = _write_raw_bundle_part(
                row_blobs,
                part_index=len(parts),
                split=split,
                parts_root=parts_root,
            )
            part_record["start_index"] = completed_rows
            completed_rows += len(row_blobs)
            part_record["end_index"] = completed_rows - 1
            parts.append(part_record)
            _write_raw_bundle_manifest(manifest_out, split, chunk_rows, parts, estimated_row_bytes)
            _write_raw_bundle_progress(metadata_rows, split, output_root, completed_rows)
            logger.info(
                "bundle_write split=%s rows=%d/%d part=%d rows_in_part=%d workers=%d path=%s",
                split,
                completed_rows,
                total_rows,
                int(part_record["index"]),
                int(part_record["row_count"]),
                worker_count,
                part_record["path"],
            )
    return _write_raw_bundle_manifest(manifest_out, split, chunk_rows, parts, estimated_row_bytes)


def build_raw_bundle_from_manifest(
    manifest_path: Path,
    output_root: Path,
    *,
    split: str = "val",
    chunk_rows: int = 4096,
    worker_count: int = RAW_BUNDLE_WORKERS_DEFAULT,
    force: bool = False,
) -> dict[str, Any]:
    if chunk_rows <= 0 or chunk_rows % 64 != 0:
        raise ValueError("chunk_rows must be a positive multiple of 64")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    parts_root = output_root / f"{split}_parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    manifest_out = output_root / f"{split}_rows.pt"
    metadata_out = output_root / f"{split}.json"
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"kinexar.raw_bundle.{split}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_dir / f"{split}_bundle.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    dataset = TrajectoryDataset(str(manifest_path), "av2")
    metadata = _raw_manifest_metadata(dataset, split)
    _write_json(metadata_out, metadata)
    logger.info(
        "raw bundle metadata split=%s rows=%d manifest=%s workers=%d chunk_rows=%d",
        split,
        len(metadata),
        manifest_path,
        worker_count,
        chunk_rows,
    )
    bundle_manifest = _build_raw_bundle_parallel(
        manifest_path,
        metadata,
        output_root=output_root,
        parts_root=parts_root,
        manifest_out=manifest_out,
        split=split,
        chunk_rows=chunk_rows,
        worker_count=worker_count,
        logger=logger,
    )
    _relativize_rows_manifest_paths(manifest_out)
    result = {
        "manifest_path": str(manifest_out),
        "metadata_path": str(metadata_out),
        "parts_root": str(parts_root),
        "split": split,
        "total_rows": int(bundle_manifest["total_rows"]),
        "part_count": int(bundle_manifest["part_count"]),
        "chunk_rows": int(bundle_manifest["chunk_rows"]),
        "workers": int(worker_count),
    }
    _write_json(output_root / "bundle_manifest.json", result)
    return result


def prepare_training_layout(archive_root: Path, output_root: Path, force: bool = False) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    profile_results = {}
    full_profile = resolve_profile("full")
    profile_results["full"] = prepare_full_bundle(
        archive_root,
        dataset_root_for(output_root, full_profile),
        force=force,
    )
    for family in ("small", "ablation"):
        profile = resolve_profile(family)
        profile_results[family] = materialize_subset_bundle(
            archive_root,
            dataset_root_for(output_root, profile),
            train_size=int(profile.train_size or 0),
            val_size=int(profile.val_size or 0),
            debug_size=int(profile.debug_size),
            seed=int(profile.seed),
            force=force,
        )
    manifest = {
        "archive_root": str(archive_root),
        "output_root": str(output_root),
        "profiles": profile_results,
    }
    _write_json(output_root / "training_dataset_manifest.json", manifest)
    return manifest


def prepare_oracle_layout(archive_root: Path, output_root: Path, force: bool = False) -> dict[str, Any]:
    manifest = prepare_training_layout(archive_root, output_root, force=force)
    _write_json(output_root / "oracle_dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--archive-root", required=True)

    full_parser = sub.add_parser("prepare-full")
    full_parser.add_argument("--archive-root", required=True)
    full_parser.add_argument("--dataset-root", required=True)
    full_parser.add_argument("--force", action="store_true")

    subset_parser = sub.add_parser("prepare-subset")
    subset_parser.add_argument("--archive-root", required=True)
    subset_parser.add_argument("--dataset-root", required=True)
    subset_parser.add_argument("--family", required=True, choices=["small", "ablation"])
    subset_parser.add_argument("--force", action="store_true")

    raw_bundle_parser = sub.add_parser("build-raw-bundle")
    raw_bundle_parser.add_argument("--manifest-path", required=True)
    raw_bundle_parser.add_argument("--output-root", required=True)
    raw_bundle_parser.add_argument("--split", default="val")
    raw_bundle_parser.add_argument("--chunk-rows", type=int, default=4096)
    raw_bundle_parser.add_argument("--workers", type=int, default=RAW_BUNDLE_WORKERS_DEFAULT)
    raw_bundle_parser.add_argument("--force", action="store_true")

    oracle_parser = sub.add_parser("prepare-oracle")
    oracle_parser.add_argument("--archive-root", required=True)
    oracle_parser.add_argument("--output-root", required=True)
    oracle_parser.add_argument("--force", action="store_true")
    training_parser = sub.add_parser("prepare-training-layout")
    training_parser.add_argument("--archive-root", required=True)
    training_parser.add_argument("--output-root", required=True)
    training_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.cmd == "verify":
        print(json.dumps(verify_archives(Path(args.archive_root).expanduser()), indent=2), flush=True)
        return
    if args.cmd == "prepare-full":
        print(
            json.dumps(
                prepare_full_bundle(
                    Path(args.archive_root).expanduser(),
                    Path(args.dataset_root).expanduser(),
                    force=bool(args.force),
                ),
                indent=2,
            ),
            flush=True,
        )
        return
    if args.cmd == "prepare-subset":
        profile = resolve_profile(args.family)
        print(
            json.dumps(
                materialize_subset_bundle(
                    Path(args.archive_root).expanduser(),
                    Path(args.dataset_root).expanduser(),
                    train_size=int(profile.train_size or 0),
                    val_size=int(profile.val_size or 0),
                    debug_size=int(profile.debug_size),
                    seed=int(profile.seed),
                    force=bool(args.force),
                ),
                indent=2,
            ),
            flush=True,
        )
        return
    if args.cmd == "build-raw-bundle":
        print(
            json.dumps(
                build_raw_bundle_from_manifest(
                    Path(args.manifest_path).expanduser(),
                    Path(args.output_root).expanduser(),
                    split=str(args.split),
                    chunk_rows=int(args.chunk_rows),
                    worker_count=int(args.workers),
                    force=bool(args.force),
                ),
                indent=2,
            ),
            flush=True,
        )
        return
    if args.cmd == "prepare-oracle":
        print(
            json.dumps(
                prepare_oracle_layout(
                    Path(args.archive_root).expanduser(),
                    Path(args.output_root).expanduser(),
                    force=bool(args.force),
                ),
                indent=2,
            ),
            flush=True,
        )
        return
    if args.cmd == "prepare-training-layout":
        print(
            json.dumps(
                prepare_training_layout(
                    Path(args.archive_root).expanduser(),
                    Path(args.output_root).expanduser(),
                    force=bool(args.force),
                ),
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
