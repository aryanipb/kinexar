from __future__ import annotations

from dataclasses import asdict
import json
import logging
import mmap
import pickle
import random
import tempfile
from pathlib import Path
import re
import shutil
import time
from typing import Any
from typing import Iterable
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from kinexar.dataset.data.dataset import TrajectoryDataset
from kinexar.training.batching import collate_dense_batch
from kinexar.training.batching import move_batch_to_device


LOGGER_NAME = "controlled_generalisation"
ROWS_BUNDLE_FORMAT = "roadweaver.rows.bundle.v1"


def setup_logging(log_dir: str | Path, filename: str = "train_logs.txt") -> logging.Logger:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(directory / filename, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


def collate_rows(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(batch)


def worker_init(worker_id: int) -> None:
    base = torch.initial_seed() % (2**32)
    seed = base + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    dataset: torch.utils.data.Dataset[Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    *,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_dense_batch,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = max(int(prefetch_factor), 2)
        kwargs["worker_init_fn"] = worker_init
    return DataLoader(dataset, **kwargs)


def load_processed_dataset(processed_root: str) -> TrajectoryDataset:
    return TrajectoryDataset(processed_root, "av2")


class ExperimentSubsetDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class MultiPartRowsDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: dict[str, Any], *, base_dir: str | Path | None = None):
        if manifest.get("format") != ROWS_BUNDLE_FORMAT:
            raise ValueError(f"unsupported rows bundle format: {manifest.get('format')!r}")
        parts = manifest.get("parts", [])
        if not isinstance(parts, list) or not parts:
            raise ValueError("multipart rows bundle has no parts")
        self.manifest = dict(manifest)
        self.parts = [dict(part) for part in parts]
        self.base_dir = Path(base_dir).expanduser().resolve() if base_dir is not None else None
        self.offsets: list[int] = []
        total = 0
        for part in self.parts:
            row_count = int(part["row_count"])
            self.offsets.append(total)
            total += row_count
        self.total_rows = total
        self._cache_part_index = -1
        self._cache_mm = None
        self._offset_cache: dict[int, list[int]] = {}

    def __len__(self) -> int:
        return self.total_rows

    def _resolve_part_index(self, index: int) -> int:
        if index < 0 or index >= self.total_rows:
            raise IndexError(index)
        lo = 0
        hi = len(self.offsets) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self.offsets[mid]
            end = start + int(self.parts[mid]["row_count"])
            if start <= index < end:
                return mid
            if index < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(index)

    def _load_part(self, part_index: int) -> list[dict[str, Any]]:
        if self._cache_part_index == part_index:
            return self._cache_rows
        payload = torch.load(self._resolve_path(self.parts[part_index]["path"]), map_location="cpu")
        if not isinstance(payload, list):
            raise ValueError(f"multipart rows part is not a list: {self.parts[part_index]['path']}")
        self._cache_part_index = part_index
        self._cache_rows = payload
        return payload

    def _load_offsets(self, part_index: int) -> list[int]:
        if part_index not in self._offset_cache:
            payload = torch.load(self._resolve_path(self.parts[part_index]["offsets_path"]), map_location="cpu")
            if not isinstance(payload, list):
                raise ValueError(f"multipart offsets payload is not a list: {self.parts[part_index]['offsets_path']}")
            self._offset_cache[part_index] = [int(value) for value in payload]
        return self._offset_cache[part_index]

    def _resolve_path(self, raw_path: str | Path | None) -> Path:
        if raw_path is None:
            raise ValueError("missing multipart bundle path")
        path = Path(str(raw_path)).expanduser()
        if path.is_absolute() or self.base_dir is None:
            return path
        return (self.base_dir / path).resolve()

    def __getitem__(self, index: int) -> dict[str, Any]:
        part_index = self._resolve_part_index(index)
        part = self.parts[part_index]
        local_index = index - self.offsets[part_index]
        serialization = str(part.get("serialization", "torch_list"))
        if serialization == "pickle_stream":
            # Elite Memory-Mapped Loader: Zero-overhead direct pointer access
            if self._cache_part_index != part_index:
                if hasattr(self, "_cache_mm") and self._cache_mm:
                    self._cache_mm.close()
                f_path = self._resolve_path(part["path"])
                with open(f_path, "rb") as f_obj:
                    self._cache_mm = mmap.mmap(f_obj.fileno(), 0, access=mmap.ACCESS_READ)
                self._cache_part_index = part_index
            
            offsets = self._load_offsets(part_index)
            self._cache_mm.seek(offsets[local_index])
            row = pickle.load(self._cache_mm)
            return row
        rows = self._load_part(part_index)
        return rows[local_index]


def preload_rows(
    dataset: TrajectoryDataset,
    metadata_rows: Sequence[dict[str, Any]],
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    total = len(metadata_rows)
    for idx, meta in enumerate(metadata_rows, start=1):
        row = dict(dataset[int(meta["dataset_index"])])
        row.update(dict(meta))
        out.append(row)
    return out


def is_rows_bundle_manifest(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("format") == ROWS_BUNDLE_FORMAT


def load_rows_bundle(path: str | Path) -> list[dict[str, Any]] | dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def _directory_bundle_manifest(bundle_root: str | Path, split: str) -> dict[str, Any] | None:
    root = Path(bundle_root).expanduser()
    parts_root = root / f"{split}_parts"
    metadata_path = root / f"{split}.json"
    bundle_manifest_path = root / "bundle_manifest.json"
    if not parts_root.is_dir():
        return None
    if not metadata_path.exists() and not bundle_manifest_path.exists():
        return None
    parts = _existing_parts(split, [parts_root])
    if not parts:
        return None
    return {
        "format": ROWS_BUNDLE_FORMAT,
        "split": split,
        "part_count": len(parts),
        "total_rows": sum(int(part["row_count"]) for part in parts),
        "parts": parts,
    }


def has_rows_bundle_split(path: str | Path, split: str) -> bool:
    root = Path(path).expanduser()
    if root.is_file():
        return root.exists()
    if (root / f"{split}_rows.pt").exists():
        return True
    return _directory_bundle_manifest(root, split) is not None


def load_rows_bundle_dataset(path: str | Path, *, split: str | None = None) -> Dataset[dict[str, Any]]:
    bundle_path = Path(path).expanduser()
    if bundle_path.is_dir():
        resolved_split = str(split or "train")
        manifest_path = bundle_path / f"{resolved_split}_rows.pt"
        if manifest_path.exists():
            bundle_path = manifest_path
        else:
            directory_manifest = _directory_bundle_manifest(bundle_path, resolved_split)
            if directory_manifest is not None:
                return MultiPartRowsDataset(directory_manifest)
            raise FileNotFoundError(f"missing rows bundle for split={resolved_split} under {bundle_path}")
    payload = load_rows_bundle(bundle_path)
    if is_rows_bundle_manifest(payload):
        return MultiPartRowsDataset(payload, base_dir=bundle_path.parent)
    if isinstance(payload, list):
        return ExperimentSubsetDataset(payload)
    raise ValueError(f"unsupported rows bundle payload: {type(payload).__name__}")


def load_rows_bundle_rows(path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    bundle_path = Path(path).expanduser()
    if bundle_path.is_dir():
        dataset = load_rows_bundle_dataset(bundle_path, split=split)
        return [dataset[index] for index in range(len(dataset))]
    payload = load_rows_bundle(bundle_path)
    if isinstance(payload, list):
        return payload
    if is_rows_bundle_manifest(payload):
        dataset = MultiPartRowsDataset(payload, base_dir=bundle_path.parent)
        return [dataset[index] for index in range(len(dataset))]
    raise ValueError(f"unsupported rows bundle payload: {type(payload).__name__}")


def _pick_part_dir(part_dirs: Sequence[str | Path]) -> Path:
    candidates = [Path(path) for path in part_dirs]
    if not candidates:
        raise ValueError("no part directories configured")
    best = max(candidates, key=lambda path: shutil.disk_usage(path).free)
    best.mkdir(parents=True, exist_ok=True)
    return best


def _estimate_rows_bytes(rows: Sequence[dict[str, Any]]) -> int:
    if not rows:
        return 0
    with tempfile.NamedTemporaryFile(prefix="rw_rows_estimate_", suffix=".bin", dir="/tmp", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        with tmp_path.open("wb") as out:
            for row in rows:
                pickle.dump(row, out, protocol=pickle.HIGHEST_PROTOCOL)
        return int(tmp_path.stat().st_size)
    finally:
        tmp_path.unlink(missing_ok=True)


def _max_rows_for_dir(path: Path, row_bytes: float, remaining_rows: int, chunk_multiple: int, reserve_bytes: int) -> int:
    free_bytes = shutil.disk_usage(path).free
    usable_bytes = max(0, free_bytes - reserve_bytes)
    if remaining_rows <= 0:
        return 0
    if remaining_rows < chunk_multiple:
        return remaining_rows
    if row_bytes <= 0:
        return max(chunk_multiple, (remaining_rows // chunk_multiple) * chunk_multiple)
    rows = int(usable_bytes // max(row_bytes, 1.0))
    rows = min(rows, remaining_rows)
    rows = (rows // chunk_multiple) * chunk_multiple
    return rows


def _part_pattern(split: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(split)}_rows\.part_(\d{{5}})\.pt$")


def _existing_parts(split: str, part_dirs: Sequence[str | Path]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    pattern = _part_pattern(split)
    all_roots = [Path(path) for path in part_dirs]
    for raw_dir in part_dirs:
        root = Path(raw_dir)
        if not root.exists():
            continue
        for path in sorted(root.glob(f"{split}_rows.part_*.pt")):
            match = pattern.match(path.name)
            if match is None:
                continue
            index = int(match.group(1))
            offsets_path = path.with_name(f"{split}_rows.part_{index:05d}.offsets.pt")
            recovery_name = f"{split}_rows.part_{index:05d}.recovery.json"
            recovery_meta_path = path.with_name(recovery_name)
            if not recovery_meta_path.exists():
                for candidate_root in all_roots:
                    candidate = candidate_root / recovery_name
                    if candidate.exists():
                        recovery_meta_path = candidate
                        break
            if offsets_path.exists():
                offsets = torch.load(offsets_path, map_location="cpu")
                if not isinstance(offsets, list):
                    raise ValueError(f"bad offsets payload for {offsets_path}")
                row_count = len(offsets)
                part_record = {
                    "index": index,
                    "path": str(path),
                    "offsets_path": str(offsets_path),
                    "serialization": "pickle_stream",
                    "row_count": row_count,
                    "bytes": int(path.stat().st_size),
                }
            elif recovery_meta_path.exists():
                recovery = json.loads(recovery_meta_path.read_text(encoding="utf-8"))
                row_count = int(recovery["row_count"])
                part_record = {
                    "index": index,
                    "path": str(path),
                    "offsets_path": None,
                    "serialization": "pickle_stream_recovery",
                    "row_count": row_count,
                    "bytes": int(path.stat().st_size),
                    "recovery_meta_path": str(recovery_meta_path),
                }
            else:
                continue
            parts.append(part_record)
    parts.sort(key=lambda part: int(part["index"]))
    if parts and int(parts[0]["index"]) != 0:
        return []
    total_rows = 0
    contiguous_parts: list[dict[str, Any]] = []
    for expected_index, part in enumerate(parts):
        if int(part["index"]) != expected_index:
            break
        row_count = int(part["row_count"])
        part["start_index"] = total_rows
        total_rows += row_count
        part["end_index"] = total_rows - 1
        contiguous_parts.append(part)
    return contiguous_parts


def _should_rollover_part(part_dir: Path, row_bytes: float, reserve_bytes: int) -> bool:
    free_bytes = shutil.disk_usage(part_dir).free
    safety_bytes = max(int(row_bytes * 4), 512 * 1024 * 1024)
    return free_bytes <= reserve_bytes + safety_bytes


def _dedupe_bundle_row_aliases(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "feat" in out and "input" in out:
        del out["feat"]
    if "future" in out and "output" in out:
        del out["future"]
    return out


def write_multipart_rows_bundle(
    dataset: TrajectoryDataset,
    metadata_rows: Sequence[dict[str, Any]],
    manifest_path: str | Path,
    part_dirs: Sequence[str | Path],
    chunk_rows: int,
    *,
    split: str,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    chunk_multiple = 64
    reserve_bytes = 2 * 1024 * 1024 * 1024
    if chunk_rows < 0:
        raise ValueError("chunk_rows must be non-negative")
    if chunk_rows > 0 and chunk_rows % chunk_multiple != 0:
        raise ValueError(f"chunk_rows must be a multiple of {chunk_multiple}")
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for raw_dir in part_dirs:
        Path(raw_dir).mkdir(parents=True, exist_ok=True)

    parts = _existing_parts(split, part_dirs)
    total_rows = sum(int(part["row_count"]) for part in parts)
    total = len(metadata_rows)
    if total_rows > total:
        raise ValueError(f"existing parts exceed available metadata rows for split={split}: {total_rows} > {total}")
    estimate_count = min(chunk_multiple, max(0, total - total_rows))
    estimate_rows: list[dict[str, Any]] = []
    for meta in metadata_rows[total_rows : total_rows + estimate_count]:
        row = dict(dataset[int(meta["dataset_index"])])
        row.update(dict(meta))
        estimate_rows.append(_dedupe_bundle_row_aliases(row))
    estimated_bytes = _estimate_rows_bytes(estimate_rows)
    estimated_row_bytes = float(estimated_bytes) / max(len(estimate_rows), 1)
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    if tmp_manifest.exists():
        tmp_manifest.unlink()
    completed_rows_log = manifest_path.parent / f"{split}_rows.completed.jsonl"
    remaining_rows_log = manifest_path.parent / f"{split}_rows.remaining.jsonl"

    def build_progress_row(meta: dict[str, Any], fallback_index: int) -> dict[str, Any]:
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

    def write_progress_logs(done_count: int) -> None:
        completed_rows = [build_progress_row(meta, index) for index, meta in enumerate(metadata_rows[:done_count])]
        remaining_rows = [build_progress_row(meta, done_count + index) for index, meta in enumerate(metadata_rows[done_count:])]
        write_jsonl_atomic(completed_rows_log, completed_rows)
        write_jsonl_atomic(remaining_rows_log, remaining_rows)

    write_progress_logs(total_rows)
    active_part_dir = _pick_part_dir(part_dirs)
    remaining_rows = total - total_rows
    target_rows = chunk_rows
    if remaining_rows <= 0:
        target_rows = 0
    elif target_rows == 0:
        target_rows = _max_rows_for_dir(active_part_dir, estimated_row_bytes, remaining_rows, chunk_multiple, reserve_bytes)
        if target_rows == 0:
            raise RuntimeError(f"insufficient free space for split={split} at {active_part_dir}")
    elif remaining_rows >= chunk_multiple:
        target_rows = min(target_rows, (remaining_rows // chunk_multiple) * chunk_multiple)

    handle = None
    tmp_part = None
    part_path = None
    tmp_offsets = None
    offsets_path = None
    offsets: list[int] = []
    part_row_count = 0
    part_start_index = 0

    def start_part(part_index: int, part_dir: Path) -> None:
        nonlocal handle, tmp_part, part_path, tmp_offsets, offsets_path, offsets, part_row_count, part_start_index
        part_path = part_dir / f"{split}_rows.part_{part_index:05d}.pt"
        tmp_part = part_path.with_suffix(part_path.suffix + ".tmp")
        offsets_path = part_dir / f"{split}_rows.part_{part_index:05d}.offsets.pt"
        tmp_offsets = offsets_path.with_suffix(offsets_path.suffix + ".tmp")
        tmp_part.unlink(missing_ok=True)
        tmp_offsets.unlink(missing_ok=True)
        handle = tmp_part.open("wb")
        offsets = []
        part_row_count = 0
        part_start_index = total_rows

    def finish_part(part_index: int, target_rows_for_log: int) -> None:
        nonlocal handle, tmp_part, part_path, tmp_offsets, offsets_path, offsets, part_row_count
        if handle is None or part_path is None or offsets_path is None or tmp_part is None or tmp_offsets is None:
            return
        handle.flush()
        handle.close()
        torch.save(offsets, tmp_offsets)
        tmp_part.replace(part_path)
        tmp_offsets.replace(offsets_path)
        parts.append(
            {
                "index": part_index,
                "path": str(part_path),
                "offsets_path": str(offsets_path),
                "serialization": "pickle_stream",
                "row_count": part_row_count,
                "start_index": part_start_index,
                "end_index": total_rows - 1,
                "bytes": int(part_path.stat().st_size),
            }
        )
        if logger is not None:
            logger.info(
                "bundle_write split=%s rows=%d/%d part=%d rows_in_part=%d target_rows=%d path=%s",
                split,
                total_rows,
                total,
                part_index,
                part_row_count,
                target_rows_for_log,
                part_path,
            )
        write_progress_logs(total_rows)
        handle = None

    if remaining_rows > 0:
        start_part(len(parts), active_part_dir)

    for index, meta in enumerate(metadata_rows[total_rows:], start=total_rows + 1):
        row = dict(dataset[int(meta["dataset_index"])])
        row.update(dict(meta))
        row = _dedupe_bundle_row_aliases(row)
        if handle is None:
            start_part(len(parts), active_part_dir)
        if part_row_count > 0 and _should_rollover_part(active_part_dir, estimated_row_bytes, reserve_bytes):
            finish_part(len(parts), target_rows)
            remaining_rows = total - total_rows
            if remaining_rows <= 0:
                continue
            active_part_dir = _pick_part_dir(part_dirs)
            if chunk_rows == 0:
                target_rows = _max_rows_for_dir(active_part_dir, estimated_row_bytes, remaining_rows, chunk_multiple, reserve_bytes)
                if target_rows == 0:
                    raise RuntimeError(f"insufficient free space for remaining split={split} rows at {active_part_dir}")
            elif remaining_rows < chunk_multiple:
                target_rows = remaining_rows
            else:
                target_rows = min(chunk_rows, (remaining_rows // chunk_multiple) * chunk_multiple)
            start_part(len(parts), active_part_dir)
        offsets.append(handle.tell())
        pickle.dump(row, handle, protocol=pickle.HIGHEST_PROTOCOL)
        part_row_count += 1
        total_rows += 1
        should_flush = part_row_count >= target_rows
        if not should_flush and index != total:
            continue
        finish_part(len(parts), target_rows)
        remaining_rows = total - total_rows
        if remaining_rows <= 0:
            continue
        active_part_dir = _pick_part_dir(part_dirs)
        if chunk_rows == 0:
            target_rows = _max_rows_for_dir(active_part_dir, estimated_row_bytes, remaining_rows, chunk_multiple, reserve_bytes)
            if target_rows == 0:
                raise RuntimeError(f"insufficient free space for remaining split={split} rows at {active_part_dir}")
        elif remaining_rows < chunk_multiple:
            target_rows = remaining_rows
        else:
            target_rows = min(chunk_rows, (remaining_rows // chunk_multiple) * chunk_multiple)

    manifest = {
        "format": ROWS_BUNDLE_FORMAT,
        "split": split,
        "total_rows": total_rows,
        "part_count": len(parts),
        "chunk_rows": int(chunk_rows),
        "chunk_multiple": chunk_multiple,
        "estimated_row_bytes": estimated_row_bytes,
        "parts": parts,
    }
    torch.save(manifest, tmp_manifest)
    tmp_manifest.replace(manifest_path)
    write_progress_logs(total_rows)
    return manifest


def make_subset(rows: Sequence[dict[str, Any]]) -> ExperimentSubsetDataset:
    return ExperimentSubsetDataset(rows)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
    tmp.replace(target)


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with target.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            vals = []
            for key in keys:
                val = row[key]
                if isinstance(val, float):
                    vals.append(f"{val:.8f}")
                else:
                    vals.append(str(val))
            handle.write(",".join(vals) + "\n")


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def move_row_to_device(row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            nested: dict[str, Any] = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, torch.Tensor):
                    nested[nested_key] = nested_value.to(device, non_blocking=True)
                elif isinstance(nested_value, dict):
                    nested[nested_key] = {
                        inner_key: inner_value.to(device, non_blocking=True) if isinstance(inner_value, torch.Tensor) else inner_value
                        for inner_key, inner_value in nested_value.items()
                    }
                else:
                    nested[nested_key] = nested_value
            out[key] = nested
        else:
            out[key] = value
    return out


def move_batch_to_device_for_experiment(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return move_batch_to_device(batch, device)


def collate_dense_rows(batch: list[dict[str, Any]], device: torch.device | None = None) -> dict[str, Any]:
    return collate_dense_batch(batch, device=device)


def summarise_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    return dict(config)
