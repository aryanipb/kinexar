from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from TRAINING.utils import _dedupe_bundle_row_aliases
from TRAINING.utils import _existing_parts
from TRAINING.utils import write_jsonl_atomic

from DATASET.dataset import TrajectoryDataset


ROWS_BUNDLE_FORMAT = "roadweaver.rows.bundle.v1"
DEFAULT_CHUNK_ROWS = 64
DEFAULT_WORKERS = 9
RESERVE_BYTES = 2 * 1024 * 1024 * 1024

_PROCESSED_DATASET: TrajectoryDataset | None = None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validation_metadata(processed_root: str | Path) -> list[dict[str, Any]]:
    dataset = TrajectoryDataset(str(processed_root), "av2")
    if getattr(dataset, "mode", "") != "processed":
        raise ValueError(f"expected processed dataset root, got {processed_root}")
    metadata: list[dict[str, Any]] = []
    for dataset_index, row in enumerate(dataset.rows):
        metadata.append(
            {
                "split": "val",
                "experiment_index": dataset_index,
                "dataset_index": dataset_index,
                "sample_id": str(row["sample_id"]),
                "scenario_id": str(row["scenario_id"]),
                "file_path": str(row["file_path"]),
                "start_t": int(row["start_t"]),
                "output_shard": str(row["output_shard"]),
            }
        )
    return metadata


def _init_processed_worker(processed_root: str, map_root: str | None = None) -> None:
    global _PROCESSED_DATASET
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if map_root:
        os.environ["KINEXAR_MAP_DIR"] = map_root
        os.environ["ROADWEAVER_MAP_DIR"] = map_root
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _PROCESSED_DATASET = TrajectoryDataset(processed_root, "av2")


def _materialize_processed_row(meta: dict[str, Any]) -> bytes:
    if _PROCESSED_DATASET is None:
        raise RuntimeError("processed dataset worker not initialized")
    row = dict(_PROCESSED_DATASET[int(meta["dataset_index"])])
    row.update(dict(meta))
    return pickle.dumps(_dedupe_bundle_row_aliases(row), protocol=pickle.HIGHEST_PROTOCOL)


def _progress_row(meta: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    return {
        "split": str(meta.get("split", "val")),
        "experiment_index": int(meta.get("experiment_index", fallback_index)),
        "dataset_index": int(meta["dataset_index"]),
        "sample_id": str(meta["sample_id"]),
        "scenario_id": str(meta["scenario_id"]),
        "file_path": str(meta["file_path"]),
        "start_t": int(meta["start_t"]),
        "output_shard": str(meta.get("output_shard", "")),
    }


def _write_progress(metadata_rows: list[dict[str, Any]], output_root: Path, done_count: int) -> None:
    completed_rows_log = output_root / "val_rows.completed.jsonl"
    remaining_rows_log = output_root / "val_rows.remaining.jsonl"
    completed_rows = [_progress_row(meta, index) for index, meta in enumerate(metadata_rows[:done_count])]
    remaining_rows = [
        _progress_row(meta, done_count + index)
        for index, meta in enumerate(metadata_rows[done_count:])
    ]
    write_jsonl_atomic(completed_rows_log, completed_rows)
    write_jsonl_atomic(remaining_rows_log, remaining_rows)


def _write_manifest(manifest_path: Path, parts: list[dict[str, Any]], chunk_rows: int, estimated_row_bytes: float) -> dict[str, Any]:
    payload = {
        "format": ROWS_BUNDLE_FORMAT,
        "split": "val",
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


def _write_part(row_blobs: list[bytes], *, part_index: int, parts_root: Path) -> dict[str, Any]:
    part_path = parts_root / f"val_rows.part_{part_index:05d}.pt"
    tmp_part = part_path.with_suffix(part_path.suffix + ".tmp")
    offsets_path = parts_root / f"val_rows.part_{part_index:05d}.offsets.pt"
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


def _relativize_manifest_paths(manifest_path: Path) -> None:
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
                continue
            part[key] = str(relative)
            changed = True
    if changed:
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(manifest_path)


def _estimate_row_bytes(processed_root: Path, metadata_rows: list[dict[str, Any]], worker_count: int, map_root: Path | None) -> float:
    estimate_count = min(64, len(metadata_rows))
    if estimate_count == 0:
        return 0.0
    ctx = mp.get_context("spawn")
    map_chunksize = max(1, min(64, estimate_count // max(worker_count * 4, 1) or 1))
    with ProcessPoolExecutor(
        max_workers=min(worker_count, estimate_count),
        mp_context=ctx,
        initializer=_init_processed_worker,
        initargs=(str(processed_root), str(map_root) if map_root else None),
    ) as pool:
        blobs = list(pool.map(_materialize_processed_row, metadata_rows[:estimate_count], chunksize=map_chunksize))
    return float(sum(len(blob) for blob in blobs)) / float(len(blobs))


def build_validation_bundle(
    processed_root: str | Path,
    bundle_root: str | Path,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    worker_count: int = DEFAULT_WORKERS,
    map_root: str | Path | None = None,
    force: bool = False,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    if chunk_rows <= 0 or chunk_rows % 64 != 0:
        raise ValueError("chunk_rows must be a positive multiple of 64")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")

    processed_root = Path(processed_root).expanduser()
    bundle_root = Path(bundle_root).expanduser()
    map_root_path = Path(map_root).expanduser() if map_root else None
    if force and bundle_root.exists():
        shutil.rmtree(bundle_root)

    parts_root = bundle_root / "val_parts"
    log_dir = bundle_root / "logs"
    bundle_root.mkdir(parents=True, exist_ok=True)
    parts_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_root / "val_rows.pt"
    metadata_path = bundle_root / "val.json"
    bundle_manifest_path = bundle_root / "bundle_manifest.json"

    logger = logging.getLogger("PREPROCESSING.bundle")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_target = Path(log_path).expanduser() if log_path else (log_dir / "bundle.log")
    handler = logging.FileHandler(log_target, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    metadata_rows = validation_metadata(processed_root)
    _write_json(metadata_path, metadata_rows)
    parts = _existing_parts("val", [parts_root])
    completed_rows = sum(int(part["row_count"]) for part in parts)
    if completed_rows > len(metadata_rows):
        raise ValueError(f"existing parts exceed metadata rows: {completed_rows} > {len(metadata_rows)}")
    _write_progress(metadata_rows, bundle_root, completed_rows)

    free_bytes = shutil.disk_usage(parts_root).free
    if free_bytes <= RESERVE_BYTES:
        raise RuntimeError(f"insufficient free space for bundle at {parts_root}")

    estimated_row_bytes = _estimate_row_bytes(processed_root, metadata_rows[completed_rows:], worker_count, map_root_path)
    if parts:
        _write_manifest(manifest_path, parts, chunk_rows, estimated_row_bytes)

    if completed_rows < len(metadata_rows):
        map_chunksize = max(1, min(64, chunk_rows // max(worker_count * 4, 1)))
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=_init_processed_worker,
            initargs=(str(processed_root), str(map_root_path) if map_root_path else None),
        ) as pool:
            while completed_rows < len(metadata_rows):
                target_rows = min(chunk_rows, len(metadata_rows) - completed_rows)
                current_meta = metadata_rows[completed_rows : completed_rows + target_rows]
                row_blobs = list(pool.map(_materialize_processed_row, current_meta, chunksize=map_chunksize))
                part_record = _write_part(row_blobs, part_index=len(parts), parts_root=parts_root)
                part_record["start_index"] = completed_rows
                completed_rows += len(row_blobs)
                part_record["end_index"] = completed_rows - 1
                parts.append(part_record)
                _write_manifest(manifest_path, parts, chunk_rows, estimated_row_bytes)
                _write_progress(metadata_rows, bundle_root, completed_rows)
                logger.info(
                    "bundle_write split=val rows=%d/%d part=%d rows_in_part=%d workers=%d path=%s",
                    completed_rows,
                    len(metadata_rows),
                    int(part_record["index"]),
                    int(part_record["row_count"]),
                    worker_count,
                    part_record["path"],
                )

    
    if not parts:
        # If no parts were built (empty dataset), create a dummy manifest
        _write_manifest(manifest_path, [], chunk_rows, 0)
        
    _relativize_manifest_paths(manifest_path)

    result = {
        "processed_root": str(processed_root),
        "bundle_root": str(bundle_root),
        "parts_root": str(parts_root),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "rows": len(metadata_rows),
        "part_count": len(parts),
        "chunk_rows": int(chunk_rows),
        "worker_count": int(worker_count),
    }
    _write_json(bundle_manifest_path, result)
    handler.close()
    logger.removeHandler(handler)
    return result
