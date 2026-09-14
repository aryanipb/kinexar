import json
import os
import sqlite3
from pathlib import Path

import torch
from torch.utils.data import Dataset

from kinexar.dataset.data import av2_dataset
from kinexar.dataset.preprocessing.graph_builder import prebuild


MAX_NEIGHBORS = 9
MAX_AGENTS = MAX_NEIGHBORS + 1


def _last_history_xy(feat: torch.Tensor, mask: torch.Tensor, agent_idx: int) -> torch.Tensor | None:
    valid = torch.nonzero(mask[agent_idx], as_tuple=False).view(-1)
    if valid.numel() == 0:
        return None
    return feat[agent_idx, valid[-1], :2].to(torch.float32)


def _nearest_agent_indices(feat: torch.Tensor, mask: torch.Tensor, max_neighbors: int = MAX_NEIGHBORS) -> torch.Tensor:
    agent_count = int(feat.shape[0])
    if agent_count <= max_neighbors + 1:
        return torch.arange(agent_count, dtype=torch.long)
    focal_xy = _last_history_xy(feat, mask, 0)
    ranked = []
    for agent_idx in range(1, agent_count):
        xy = _last_history_xy(feat, mask, agent_idx)
        if xy is None:
            continue
        dist2 = 0.0 if focal_xy is None else float(((xy - focal_xy) ** 2).sum().item())
        ranked.append((dist2, agent_idx))
    ranked.sort(key=lambda item: (item[0], item[1]))
    keep = [0] + [agent_idx for _, agent_idx in ranked[:max_neighbors]]
    return torch.tensor(keep, dtype=torch.long)


class TrajectoryDataset(Dataset):
    """AV2 dataset wrapper for raw manifests and processed shard bundles."""

    def __init__(self, path, kind):
        if kind != "av2":
            raise ValueError(f"unsupported dataset kind: {kind}")
        self.path = path
        self.mode = "raw"
        self.idx = []
        self.rows = []
        self.shards = {}
        self.shard_pos = {}
        self._active_shard_path = None
        p = Path(path)
        if p.is_dir() and (p / "progress.sqlite3").exists() and (p / "shards").is_dir():
            self.mode = "processed"
            self.root = p
            self._init_processed()
            return
        with open(path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                txt = line.strip()
                if txt and txt not in [b"[", b"]"]:
                    self.idx.append(pos)

    def _init_processed(self):
        db = self.root / "progress.sqlite3"
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT sample_id, scenario_id, file_path, start_t, output_shard
                FROM samples
                WHERE status = 'done'
                ORDER BY ord
                """
            ).fetchall()
        finally:
            conn.close()
        self.rows = [dict(x) for x in rows]

    def __len__(self):
        if self.mode == "processed":
            return len(self.rows)
        return len(self.idx)

    def get(self, i):
        if self.mode == "processed":
            return dict(self.rows[i])
        with open(self.path, "rb") as f:
            f.seek(self.idx[i])
            line = f.readline().decode("utf-8").strip()
        if line.endswith(","):
            line = line[:-1]
        return json.loads(line)

    def _load_shard(self, path):
        if self._active_shard_path != path:
            self.shards.clear()
            self.shard_pos.clear()
            self._active_shard_path = path
        if path not in self.shards:
            dat = torch.load(path, map_location="cpu")
            self.shards[path] = dat
            self.shard_pos[path] = {sid: i for i, sid in enumerate(dat["sample_ids"])}
        return self.shards[path], self.shard_pos[path]

    def _get_processed(self, i):
        row = self.rows[i]
        shard_path = row["output_shard"]
        if shard_path.startswith("/home/") and not Path(shard_path).exists():
            modal_shard = Path("/data/shards") / os.path.basename(shard_path)
            if modal_shard.exists() or Path("/data/shards").exists():
                shard_path = str(modal_shard)

        shard, pos = self._load_shard(shard_path)
        j = pos[row["sample_id"]]
        n = int(shard["agent_count"][j])
        feat = shard["input"][j, :n]
        mask = shard["mask"][j, :n]
        keep = _nearest_agent_indices(feat, mask)
        feat = feat[keep]
        mask = mask[keep]
        sample = {
            "feat": feat,
            "future": shard["output"][j],
            "input": feat,
            "output": shard["output"][j],
            "mask": mask,
            "sample_id": shard["sample_ids"][j] if "sample_ids" in shard else row["sample_id"],
            "scenario_id": shard["scenario_ids"][j],
        }
        if "graphs" in shard and n <= MAX_AGENTS:
            sample["graph"] = shard["graphs"][j]
        else:
            sample["graph"] = prebuild(feat, mask, topk=9, lag=1, span=1)
        return sample

    def __getitem__(self, i):
        if self.mode == "processed":
            return self._get_processed(i)
        row = self.get(i)
        return av2_dataset.load(row)
