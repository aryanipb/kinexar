import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path
import tempfile

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kinexar.dataset.data import TrajectoryDataset
from kinexar.dataset.data import av2_dataset
from kinexar.dataset.preprocessing.graph_builder import prebuild


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(os.environ.get("RW_AV2_MANIFEST", str(ROOT / "data" / "manifests" / "av2_manifest.json")))
OUT = Path(os.environ.get("RW_AV2_OUT", str(ROOT / "data" / "processed" / "av2_full")))
DB = OUT / "progress.sqlite3"
SHARDS = OUT / "shards"
LOGS = OUT / "logs"
LOG = LOGS / "preprocess.log"
PID = OUT / "preprocess.pid"
DONE = OUT / "processed_manifest.jsonl"
LEFT = OUT / "remaining_manifest.jsonl"
TRASH = ROOT / "data" / "_trash_cleanup"

MAPC = ROOT / "data" / "maps_cache"
TRASH = ROOT / "data" / "_trash_cleanup"

TMP = ".tmp"
MAX_WORKERS = 32
DEF_WORKERS = 8
DEF_SHARD_SAMPLES = 64
DEF_SHARD_BYTES = 512 * 1024 * 1024
DEF_MAX_ATTEMPTS = 3
DEF_SNAPSHOT = 1

STOP = {"flag": False, "sig": None}


@dataclass
class Sample:
    sample_id: str
    scenario_id: str
    start_t: int
    input: torch.Tensor
    output: torch.Tensor
    mask: torch.Tensor
    graph: dict


@dataclass
class Group:
    gid: int
    scenario_id: str
    file_path: str
    rows: list


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def limit_threads():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def ensure():
    OUT.mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)


def connect():
    ensure()
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            sample_id TEXT PRIMARY KEY,
            scenario_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            start_t INTEGER NOT NULL,
            ord INTEGER NOT NULL UNIQUE,
            status TEXT NOT NULL,
            output_shard TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_status_ord ON samples(status, ord)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shards (
            shard_path TEXT PRIMARY KEY,
            row_count INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            first_ord INTEGER NOT NULL,
            last_ord INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            val TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def get_state(conn, key, default=None):
    row = conn.execute("SELECT val FROM state WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return row["val"]


def set_state(conn, key, val):
    conn.execute("INSERT INTO state(key, val) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET val = excluded.val", (key, str(val)))


def parse_manifest():
    rows = []
    with MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()
            if not txt or txt in ["[", "]"]:
                continue
            if txt.endswith(","):
                txt = txt[:-1]
            row = json.loads(txt)
            rows.append(row)
    rows.sort(key=lambda x: (x["file_path"], int(x["start_t"]), x["scenario_id"]))
    out = []
    for i, row in enumerate(rows):
        out.append(
            {
                "sample_id": f'{row["scenario_id"]}:{int(row["start_t"])}',
                "scenario_id": row["scenario_id"],
                "file_path": row["file_path"],
                "start_t": int(row["start_t"]),
                "ord": i,
                "history_steps": int(row.get("history_steps", 50)),
                "future_steps": int(row.get("future_steps", 40)),
                "window_steps": int(row.get("window_steps", int(row.get("history_steps", 50)) + int(row.get("future_steps", 40)))),
            }
        )
    return out


def init_samples(conn):
    row = conn.execute("SELECT COUNT(*) AS n FROM samples").fetchone()
    if row["n"] > 0:
        return
    arr = parse_manifest()
    cur = conn.cursor()
    stamp = now()
    cur.executemany(
        """
        INSERT INTO samples(sample_id, scenario_id, file_path, start_t, ord, status, output_shard, attempts, last_error, updated_at)
        VALUES(?, ?, ?, ?, ?, 'pending', NULL, 0, NULL, ?)
        """,
        [(x["sample_id"], x["scenario_id"], x["file_path"], x["start_t"], x["ord"], stamp) for x in arr],
    )
    set_state(conn, "next_shard", 0)
    set_state(conn, "last_committed_sample_id", "")
    set_state(conn, "last_error", "")
    set_state(conn, "parity_ok", "0")
    conn.commit()


def cleanup_tmp():
    ensure()
    for path in SHARDS.glob(f"*{TMP}*"):
        if path.is_file():
            path.unlink()


def reset_stale(conn, retry_failed=False):
    stamp = now()
    conn.execute(
        "UPDATE samples SET status = 'pending', updated_at = ? WHERE status = 'processing'",
        (stamp,),
    )
    if retry_failed:
        conn.execute(
            "UPDATE samples SET status = 'pending', output_shard = NULL, last_error = NULL, updated_at = ? WHERE status = 'failed'",
            (stamp,),
        )
    conn.commit()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_torch_save(obj, path):
    tmp = path.with_name(f"{path.name}{TMP}.{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def make_sample(lst, row):
    hist_steps = int(row.get("history_steps", 50))
    future_steps = int(row.get("future_steps", 40))
    window_steps = int(row.get("window_steps", hist_steps + future_steps))
    start = int(row["start_t"])
    stop = start + window_steps
    root = lst[0]["focal_track_id"]
    dat = {}
    for x in lst:
        t = x["timestep"]
        if t < start or t >= stop:
            continue
        k = x["track_id"]
        if k not in dat:
            dat[k] = {}
        dat[k][t - start] = [
            x["position_x"],
            x["position_y"],
            x["velocity_x"],
            x["velocity_y"],
        ]
    for k in list(dat):
        dat[k] = av2_dataset.ang(dat[k])
    ids = av2_dataset.select_nearest_agent_ids(dat, root, hist_steps)
    n = len(ids)
    inp = torch.zeros((n, hist_steps, 5), dtype=torch.float16)
    out = torch.zeros((future_steps, 2), dtype=torch.float16)
    msk = torch.zeros((n, hist_steps), dtype=torch.bool)
    for i, k in enumerate(ids):
        cur = dat.get(k, {})
        for t in range(hist_steps):
            if t in cur:
                inp[i, t] = torch.tensor(cur[t], dtype=torch.float16)
                msk[i, t] = True
    cur = dat.get(root, {})
    for t in range(future_steps):
        j = t + hist_steps
        if j in cur:
            out[t] = torch.tensor(cur[j][:2], dtype=torch.float16)
    return Sample(
        sample_id=row["sample_id"],
        scenario_id=row["scenario_id"],
        start_t=start,
        input=inp,
        output=out,
        mask=msk,
        graph=prebuild(inp, msk, topk=9, lag=1, span=1),
    )


def process_group(group):
    tab = pq.read_table(group.file_path, columns=av2_dataset.col, use_threads=False)
    lst = tab.to_pylist()
    out = []
    for row in group.rows:
        out.append(make_sample(lst, row))
    return out


def worker_main(conn):
    limit_threads()
    while True:
        msg = conn.recv()
        if msg["kind"] == "stop":
            conn.close()
            return
        group = msg["group"]
        try:
            out = process_group(Group(**group))
            payload = []
            for x in out:
                payload.append(
                    {
                        "sample_id": x.sample_id,
                        "scenario_id": x.scenario_id,
                        "start_t": x.start_t,
                        "input": x.input,
                        "output": x.output,
                        "mask": x.mask,
                        "graph": x.graph,
                    }
                )
            conn.send({"kind": "ok", "gid": group["gid"], "rows": payload})
        except Exception as exc:
            conn.send({"kind": "err", "gid": group["gid"], "error": f"{type(exc).__name__}: {exc}"})


class Worker:
    def __init__(self, wid, proc, conn):
        self.wid = wid
        self.proc = proc
        self.conn = conn
        self.busy = False
        self.group = None


def spawn_workers(n):
    ctx = mp.get_context("spawn")
    out = []
    for i in range(n):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=worker_main, args=(child,))
        proc.start()
        out.append(Worker(i, proc, parent))
    return out


def stop_workers(workers):
    for w in workers:
        try:
            w.conn.send({"kind": "stop"})
        except Exception:
            pass
    for w in workers:
        w.proc.join(timeout=5)
        if w.proc.is_alive():
            w.proc.terminate()
            w.proc.join(timeout=5)
        w.conn.close()


def pending_groups(conn):
    rows = conn.execute(
        """
        SELECT sample_id, scenario_id, file_path, start_t, ord
        FROM samples
        WHERE status = 'pending'
        ORDER BY ord
        """
    ).fetchall()
    out = []
    cur = None
    gid = 0
    for row in rows:
        item = {
            "sample_id": row["sample_id"],
            "scenario_id": row["scenario_id"],
            "file_path": row["file_path"],
            "start_t": int(row["start_t"]),
            "ord": int(row["ord"]),
        }
        key = (item["scenario_id"], item["file_path"])
        if cur is None or key != cur[0]:
            cur = [key, []]
            out.append(cur)
        cur[1].append(item)
    ans = []
    for _, rows in out:
        ans.append(Group(gid=gid, scenario_id=rows[0]["scenario_id"], file_path=rows[0]["file_path"], rows=rows))
        gid += 1
    return ans


def mark_processing(conn, group):
    stamp = now()
    ids = [x["sample_id"] for x in group.rows]
    conn.executemany(
        """
        UPDATE samples
        SET status = 'processing', attempts = attempts + 1, last_error = NULL, updated_at = ?
        WHERE sample_id = ?
        """,
        [(stamp, x) for x in ids],
    )
    set_state(conn, "current_scenario_id", group.scenario_id)
    set_state(conn, "current_file_path", group.file_path)
    conn.commit()


def mark_group_error(conn, group, err, max_attempts):
    rows = conn.execute(
        "SELECT sample_id, attempts FROM samples WHERE sample_id IN ({}) ORDER BY ord".format(",".join("?" for _ in group.rows)),
        [x["sample_id"] for x in group.rows],
    ).fetchall()
    stamp = now()
    vals = []
    for row in rows:
        status = "failed" if int(row["attempts"]) >= int(max_attempts) else "pending"
        vals.append((status, err, stamp, row["sample_id"]))
    conn.executemany(
        "UPDATE samples SET status = ?, last_error = ?, updated_at = ? WHERE sample_id = ?",
        vals,
    )
    set_state(conn, "last_error", err)
    conn.commit()


def sample_bytes(sample):
    total = 0
    total += sample.input.numel() * sample.input.element_size()
    total += sample.output.numel() * sample.output.element_size()
    total += sample.mask.numel() * sample.mask.element_size()
    total += len(sample.sample_id) + len(sample.scenario_id)
    return total


def pack_shard(arr):
    n = len(arr)
    max_agents = max(int(x.input.shape[0]) for x in arr)
    hist_steps = int(arr[0].input.shape[1])
    future_steps = int(arr[0].output.shape[0])
    inp = torch.zeros((n, max_agents, hist_steps, 5), dtype=torch.float16)
    out = torch.zeros((n, future_steps, 2), dtype=torch.float16)
    msk = torch.zeros((n, max_agents, hist_steps), dtype=torch.bool)
    agents = torch.zeros((n,), dtype=torch.long)
    sids = []
    scns = []
    starts = []
    graphs = []
    for i, x in enumerate(arr):
        a = int(x.input.shape[0])
        inp[i, :a] = x.input
        out[i] = x.output
        msk[i, :a] = x.mask
        agents[i] = a
        sids.append(x.sample_id)
        scns.append(x.scenario_id)
        starts.append(int(x.start_t))
        graphs.append(x.graph)
    return {
        "input": inp,
        "output": out,
        "mask": msk,
        "graphs": graphs,
        "history_steps": hist_steps,
        "future_steps": future_steps,
        "agent_count": agents,
        "sample_ids": sids,
        "scenario_ids": scns,
        "start_ts": starts,
    }


def write_snapshots(conn):
    with DONE.open("w", encoding="utf-8") as f:
        for row in conn.execute(
            """
            SELECT sample_id, scenario_id, file_path, start_t, status, output_shard, attempts, last_error, updated_at
            FROM samples
            WHERE status = 'done'
            ORDER BY ord
            """
        ):
            f.write(json.dumps(dict(row)) + "\n")
    with LEFT.open("w", encoding="utf-8") as f:
        for row in conn.execute(
            """
            SELECT sample_id, scenario_id, file_path, start_t, status, output_shard, attempts, last_error, updated_at
            FROM samples
            WHERE status != 'done'
            ORDER BY ord
            """
        ):
            f.write(json.dumps(dict(row)) + "\n")


def commit_shard(conn, arr, shard_idx, snapshot_every):
    if not arr:
        return shard_idx
    payload = pack_shard(arr)
    name = f"shard_{shard_idx:06d}.pt"
    path = SHARDS / name
    if path.exists():
        raise ValueError(f"shard exists: {path}")
    atomic_torch_save(payload, path)
    sha = sha256_file(path)
    first_ord = conn.execute("SELECT ord FROM samples WHERE sample_id = ?", (arr[0].sample_id,)).fetchone()["ord"]
    last_ord = conn.execute("SELECT ord FROM samples WHERE sample_id = ?", (arr[-1].sample_id,)).fetchone()["ord"]
    conn.execute(
        """
        INSERT OR REPLACE INTO shards(shard_path, row_count, sha256, first_ord, last_ord, updated_at)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        (str(path), len(arr), sha, int(first_ord), int(last_ord), now()),
    )
    conn.executemany(
        """
        UPDATE samples
        SET status = 'done', output_shard = ?, updated_at = ?
        WHERE sample_id = ?
        """,
        [(str(path), now(), x.sample_id) for x in arr],
    )
    shard_idx += 1
    set_state(conn, "next_shard", shard_idx)
    set_state(conn, "last_committed_sample_id", arr[-1].sample_id)
    set_state(conn, "last_error", "")
    conn.commit()
    if snapshot_every > 0 and shard_idx % snapshot_every == 0:
        write_snapshots(conn)
    return shard_idx


def validate_shard_payload(payload, expect_ids, row_count):
    need = [
        "input",
        "output",
        "mask",
        "sample_ids",
        "scenario_ids",
        "start_ts",
        "agent_count",
    ]
    if set(need) - set(payload):
        return False, "missing fields"
    if len(payload["sample_ids"]) != row_count:
        return False, "bad row count"
    if list(payload["sample_ids"]) != list(expect_ids):
        return False, "bad sample ids"
    n = row_count
    hist_steps = int(payload.get("history_steps", payload["input"].shape[2]))
    future_steps = int(payload.get("future_steps", payload["output"].shape[1]))
    if payload["input"].shape[0] != n:
        return False, "bad input dim"
    if payload["input"].shape[2] != hist_steps or payload["mask"].shape[2] != hist_steps:
        return False, "bad history dim"
    if payload["output"].shape != (n, future_steps, 2):
        return False, "bad output dim"
    if payload["mask"].shape[:1] != (n,):
        return False, "bad mask dim"
    return True, ""


def verify_shards(conn):
    rows = conn.execute(
        """
        SELECT shard_path, row_count, sha256
        FROM shards
        ORDER BY first_ord
        """
    ).fetchall()
    for row in rows:
        path = Path(row["shard_path"])
        expect = conn.execute(
            "SELECT sample_id FROM samples WHERE output_shard = ? AND status = 'done' ORDER BY ord",
            (str(path),),
        ).fetchall()
        expect_ids = [x["sample_id"] for x in expect]
        ok = True
        err = ""
        if not path.exists():
            ok = False
            err = "missing shard"
        else:
            try:
                sha = sha256_file(path)
                if sha != row["sha256"]:
                    ok = False
                    err = "sha256 mismatch"
                else:
                    payload = torch.load(path, map_location="cpu")
                    ok, err = validate_shard_payload(payload, expect_ids, int(row["row_count"]))
            except Exception as exc:
                ok = False
                err = f"{type(exc).__name__}: {exc}"
        if ok:
            continue
        path.unlink(missing_ok=True)
        conn.execute("DELETE FROM shards WHERE shard_path = ?", (str(path),))
        conn.execute(
            """
            UPDATE samples
            SET status = 'pending', output_shard = NULL, updated_at = ?, last_error = ?
            WHERE output_shard = ?
            """,
            (now(), err, str(path)),
        )
        set_state(conn, "last_error", err)
        conn.commit()


def install_signals():
    def handle(sig, _frame):
        STOP["flag"] = True
        STOP["sig"] = signal.Signals(sig).name
        print(f"received {STOP['sig']}, stopping after safe checkpoint", flush=True)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)


def parity_rows():
    rows = parse_manifest()
    if not rows:
        return []
    n = min(10, len(rows))
    out = []
    for i in range(n):
        idx = round(i * (len(rows) - 1) / max(n - 1, 1))
        out.append(rows[idx])
    return out


def tensor_sig(x):
    y = x.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(y.shape)).encode("utf-8"))
    h.update(str(y.dtype).encode("utf-8"))
    h.update(y.numpy().tobytes())
    return h.hexdigest()


def run_parity():
    print("running parity check against datasets.av2_dataset.load", flush=True)
    rows = parity_rows()
    if not rows:
        print("skipping parity check (no rows)", flush=True)
        return
    groups = []
    cur = None
    gid = 0
    for row in rows:
        key = (row["scenario_id"], row["file_path"])
        if cur is None or key != cur[0]:
            cur = [key, []]
            groups.append(cur)
        cur[1].append(row)
    built = {}
    for _, part in groups:
        grp = Group(gid=gid, scenario_id=part[0]["scenario_id"], file_path=part[0]["file_path"], rows=part)
        gid += 1
        for sample in process_group(grp):
            built[sample.sample_id] = sample
    with tempfile.TemporaryDirectory(prefix="av2_parity_", dir="/tmp") as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "shards").mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(tmp / "progress.sqlite3")
        db.execute(
            """
            CREATE TABLE samples (
                sample_id TEXT PRIMARY KEY,
                scenario_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_t INTEGER NOT NULL,
                ord INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL,
                output_shard TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        shard = tmp / "shards" / "shard_000000.pt"
        atomic_torch_save(pack_shard([built[row["sample_id"]] for row in rows]), shard)
        db.executemany(
            """
            INSERT INTO samples(sample_id, scenario_id, file_path, start_t, ord, status, output_shard, attempts, last_error, updated_at)
            VALUES(?, ?, ?, ?, ?, 'done', ?, 1, NULL, ?)
            """,
            [(row["sample_id"], row["scenario_id"], row["file_path"], int(row["start_t"]), i, str(shard), now()) for i, row in enumerate(rows)],
        )
        db.commit()
        db.close()
        ds = TrajectoryDataset(str(tmp), "av2")
        for i, row in enumerate(rows):
            ref = av2_dataset.load(row)
            cur = ds[i]
            if row["sample_id"] != built[row["sample_id"]].sample_id:
                raise ValueError(f"bad sample id order: {built[row['sample_id']].sample_id} != {row['sample_id']}")
            pairs = [
                ("input", cur["input"], ref["input"]),
                ("output", cur["output"], ref["output"]),
                ("mask", cur["mask"], ref["mask"]),
            ]
            for name, a, b in pairs:
                if a.shape != b.shape:
                    raise ValueError(f"{row['sample_id']} {name} shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")
                if a.dtype != b.dtype:
                    raise ValueError(f"{row['sample_id']} {name} dtype mismatch: {a.dtype} != {b.dtype}")
                if not torch.equal(a, b):
                    raise ValueError(f"{row['sample_id']} {name} value mismatch: {tensor_sig(a)} != {tensor_sig(b)}")
            print(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "input": list(cur["input"].shape),
                        "output": list(cur["output"].shape),
                        "mask": list(cur["mask"].shape),
                        "input_sha": tensor_sig(cur["input"]),
                        "output_sha": tensor_sig(cur["output"]),
                        "mask_sha": tensor_sig(cur["mask"].to(torch.uint8)),
                    }
                ),
                flush=True,
            )
    print("parity check passed", flush=True)


def run_foreground(args):
    limit_threads()
    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise ValueError(f"workers must be in [1, {MAX_WORKERS}]")
    ensure()
    conn = connect()
    init_samples(conn)
    cleanup_tmp()
    reset_stale(conn, retry_failed=args.retry_failed)
    verify_shards(conn)
    if get_state(conn, "parity_ok", "0") != "1":
        run_parity()
        set_state(conn, "parity_ok", "1")
        conn.commit()
    install_signals()
    PID.write_text(str(os.getpid()), encoding="utf-8")
    groups = pending_groups(conn)
    shard_idx = int(get_state(conn, "next_shard", "0"))
    if not groups:
        write_snapshots(conn)
        print("nothing to do", flush=True)
        PID.unlink(missing_ok=True)
        conn.close()
        return
    workers = spawn_workers(args.workers)
    by_conn = {w.conn: w for w in workers}
    send = 0
    want = 0
    buf = []
    buf_bytes = 0
    ready = {}
    committed = 0
    try:
        while want < len(groups):
            while not STOP["flag"] and send < len(groups):
                idle = None
                for w in workers:
                    if not w.busy:
                        idle = w
                        break
                if idle is None:
                    break
                grp = groups[send]
                mark_processing(conn, grp)
                idle.conn.send(
                    {
                        "kind": "group",
                        "group": {
                            "gid": grp.gid,
                            "scenario_id": grp.scenario_id,
                            "file_path": grp.file_path,
                            "rows": grp.rows,
                        },
                    }
                )
                idle.busy = True
                idle.group = grp
                send += 1
            active = [w.conn for w in workers if w.busy]
            if not active:
                if STOP["flag"] or send >= len(groups):
                    break
                time.sleep(0.2)
                continue
            for conn_in in wait(active, timeout=1.0):
                w = by_conn[conn_in]
                msg = conn_in.recv()
                grp = w.group
                w.busy = False
                w.group = None
                ready[grp.gid] = (grp, msg)
            while want in ready:
                grp, msg = ready.pop(want)
                if msg["kind"] == "err":
                    mark_group_error(conn, grp, msg["error"], args.max_attempts)
                else:
                    for item in msg["rows"]:
                        sample = Sample(**item)
                        buf.append(sample)
                        buf_bytes += sample_bytes(sample)
                    if len(buf) >= args.shard_samples or buf_bytes >= args.shard_bytes:
                        shard_idx = commit_shard(conn, buf, shard_idx, args.snapshot_every)
                        committed += 1
                        buf = []
                        buf_bytes = 0
                want += 1
            if STOP["flag"] and send >= len(groups) and not any(w.busy for w in workers):
                break
        if buf:
            shard_idx = commit_shard(conn, buf, shard_idx, args.snapshot_every)
            committed += 1
            buf = []
        write_snapshots(conn)
        if STOP["flag"]:
            print(f"stopped cleanly on {STOP['sig']}", flush=True)
        else:
            print("preprocessing complete", flush=True)
        print(f"committed shards this run: {committed}", flush=True)
    finally:
        stop_workers(workers)
        PID.unlink(missing_ok=True)
        conn.close()


def pid_running():
    if not PID.exists():
        return None
    try:
        pid = int(PID.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def launch(args):
    ensure()
    if pid_running() is not None:
        raise ValueError("preprocess job already running")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--workers",
        str(args.workers),
        "--shard-samples",
        str(args.shard_samples),
        "--shard-bytes",
        str(args.shard_bytes),
        "--max-attempts",
        str(args.max_attempts),
        "--snapshot-every",
        str(args.snapshot_every),
    ]
    if args.retry_failed:
        cmd.append("--retry-failed")
    env = os.environ.copy()
    env["RW_AV2_MANIFEST"] = str(MANIFEST)
    env["RW_AV2_OUT"] = str(OUT)
    with LOG.open("ab") as f:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    print(f"started pid={proc.pid}")
    print(LOG)


def sig_job(sig):
    pid = pid_running()
    if pid is None:
        raise ValueError("no running preprocess job")
    os.kill(pid, sig)
    print(f"sent {signal.Signals(sig).name} to pid={pid}")


def status():
    ensure()
    conn = connect()
    init_samples(conn)
    total = conn.execute("SELECT COUNT(*) AS n FROM samples").fetchone()["n"]
    done = conn.execute("SELECT COUNT(*) AS n FROM samples WHERE status = 'done'").fetchone()["n"]
    pending = conn.execute("SELECT COUNT(*) AS n FROM samples WHERE status = 'pending'").fetchone()["n"]
    processing = conn.execute("SELECT COUNT(*) AS n FROM samples WHERE status = 'processing'").fetchone()["n"]
    failed = conn.execute("SELECT COUNT(*) AS n FROM samples WHERE status = 'failed'").fetchone()["n"]
    cur = conn.execute(
        """
        SELECT scenario_id, file_path, sample_id
        FROM samples
        WHERE status = 'processing'
        ORDER BY ord
        LIMIT 1
        """
    ).fetchone()
    print(f"running_pid: {pid_running()}")
    print(f"total_rows: {total}")
    print(f"processed_rows: {done}")
    print(f"remaining_rows: {pending + processing}")
    print(f"failed_rows: {failed}")
    if cur is None:
        print("current_scenario: ")
        print("current_file: ")
        print("current_sample: ")
    else:
        print(f"current_scenario: {cur['scenario_id']}")
        print(f"current_file: {cur['file_path']}")
        print(f"current_sample: {cur['sample_id']}")
    print(f"current_shard_index: {get_state(conn, 'next_shard', '0')}")
    print(f"last_committed_sample_id: {get_state(conn, 'last_committed_sample_id', '')}")
    print(f"last_error: {get_state(conn, 'last_error', '')}")
    conn.close()


def tail(path, lines=40, follow=False):
    if not path.exists():
        print(f"missing log: {path}")
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = f.readlines()
        for line in data[-lines:]:
            print(line.rstrip())
        if not follow:
            return
        while True:
            line = f.readline()
            if line:
                print(line.rstrip())
                continue
            if pid_running() is None:
                return
            time.sleep(1.0)


def dir_size(path):
    total = 0
    if not path.exists():
        return total
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def storage_check():
    ensure()
    conn = connect()
    init_samples(conn)
    done = conn.execute("SELECT COUNT(*) AS n FROM samples WHERE status = 'done'").fetchone()["n"]
    rows = conn.execute("SELECT DISTINCT scenario_id FROM samples WHERE status = 'done'").fetchall()
    done_ids = sorted(x["scenario_id"] for x in rows)
    shards = sorted(SHARDS.glob("*.pt"))
    bad = []
    for path in shards[: min(5, len(shards))]:
        payload = torch.load(path, map_location="cpu")
        for key in ["distance_map", "map_origin", "map_resolution"]:
            if key in payload:
                bad.append((str(path), key))
    print(f"done_rows: {done}")
    print(f"done_scenarios: {len(done_ids)}")
    print(f"shard_files: {len(shards)}")
    print(f"shards_dir_bytes: {dir_size(SHARDS)}")
    print(f"obsolete_maps_cache_bytes: {dir_size(MAPC)}")
    print(f"obsolete_maps_global_bytes: {dir_size(MAPG)}")
    print(f"shards_embed_maps: {bool(bad)}")
    for row in bad[:10]:
        print("bad_shard_key", *row)
    conn.close()


def stamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def old_shard(path):
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return True, "unloadable"
    keys = set(payload)
    bad = {"distance_map", "map_origin", "map_resolution", "map_hash", "map_hashes", "file_path", "file_paths"}
    hit = sorted(keys & bad)
    if hit:
        return True, ",".join(hit)
    return False, ""


def cleanup_scan():
    out = []
    for path in [MAPC, MAPG]:
        if path.exists():
            out.append({"path": path, "kind": "dir", "reason": "obsolete_map_store", "bytes": dir_size(path)})
    for path in sorted(SHARDS.glob("*")):
        if path.is_dir():
            continue
        if TMP in path.name or path.suffix != ".pt":
            out.append({"path": path, "kind": "file", "reason": "temp_or_incomplete", "bytes": path.stat().st_size})
            continue
        bad, why = old_shard(path)
        if bad:
            out.append({"path": path, "kind": "file", "reason": f"obsolete_shard:{why}", "bytes": path.stat().st_size})
    for path in sorted(OUT.rglob(f"*{TMP}*")):
        if path.is_file() and path.parent != SHARDS:
            out.append({"path": path, "kind": "file", "reason": "temp_file", "bytes": path.stat().st_size})
    seen = set()
    ans = []
    for item in out:
        key = str(item["path"])
        if key in seen:
            continue
        seen.add(key)
        ans.append(item)
    return ans


def print_cleanup(items):
    total = sum(x["bytes"] for x in items)
    print(f"cleanup_candidates: {len(items)}")
    print(f"cleanup_bytes: {total}")
    for item in items[:50]:
        print(item["reason"], item["path"])
    if len(items) > 50:
        print(f"... {len(items) - 50} more")


def move_to_trash(path, root):
    rel = path.relative_to(ROOT)
    dst = root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, dst)


def cleanup_apply():
    ensure()
    items = cleanup_scan()
    print_cleanup(items)
    if not items:
        return
    ans = input("Type YES to move these paths into data/_trash_cleanup/: ").strip()
    if ans != "YES":
        print("cleanup cancelled")
        return
    root = TRASH / stamp()
    root.mkdir(parents=True, exist_ok=True)
    for item in items:
        move_to_trash(item["path"], root)
    print(root)


def cleanup_dry():
    ensure()
    print_cleanup(cleanup_scan())


def cleanup_purge():
    ensure()
    if not TRASH.exists():
        print("nothing to purge")
        return
    run_parity()
    items = [{"path": p, "kind": "dir" if p.is_dir() else "file", "reason": "trash", "bytes": dir_size(p) if p.is_dir() else p.stat().st_size} for p in sorted(TRASH.iterdir())]
    print_cleanup(items)
    ans = input("Type PURGE to permanently delete data/_trash_cleanup/: ").strip()
    if ans != "PURGE":
        print("purge cancelled")
        return
    shutil.rmtree(TRASH)
    print("trash purged")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["run", "start", "resume"]:
        cur = sub.add_parser(name)
        cur.add_argument("--workers", type=int, default=DEF_WORKERS)
        cur.add_argument("--shard-samples", type=int, default=DEF_SHARD_SAMPLES)
        cur.add_argument("--shard-bytes", type=int, default=DEF_SHARD_BYTES)
        cur.add_argument("--max-attempts", type=int, default=DEF_MAX_ATTEMPTS)
        cur.add_argument("--snapshot-every", type=int, default=DEF_SNAPSHOT)
        cur.add_argument("--retry-failed", action="store_true")

    sub.add_parser("parity")
    sub.add_parser("pause")
    sub.add_parser("stop")
    sub.add_parser("status")
    sub.add_parser("storage")
    sub.add_parser("cleanup_dry")
    sub.add_parser("cleanup_apply")
    sub.add_parser("cleanup_purge")
    logp = sub.add_parser("logs")
    logp.add_argument("--lines", type=int, default=40)
    logp.add_argument("--follow", action="store_true")
    sub.add_parser("snapshots")

    args = parser.parse_args()
    if args.cmd == "parity":
        run_parity()
        conn = connect()
        init_samples(conn)
        set_state(conn, "parity_ok", "1")
        conn.commit()
        conn.close()
        return
    if args.cmd == "run":
        run_foreground(args)
        return
    if args.cmd in ["start", "resume"]:
        launch(args)
        return
    if args.cmd == "pause":
        sig_job(signal.SIGINT)
        return
    if args.cmd == "stop":
        sig_job(signal.SIGTERM)
        return
    if args.cmd == "status":
        status()
        return
    if args.cmd == "logs":
        tail(LOG, lines=args.lines, follow=args.follow)
        return
    if args.cmd == "storage":
        storage_check()
        return
    if args.cmd == "cleanup_dry":
        cleanup_dry()
        return
    if args.cmd == "cleanup_apply":
        cleanup_apply()
        return
    if args.cmd == "cleanup_purge":
        cleanup_purge()
        return
    if args.cmd == "snapshots":
        conn = connect()
        init_samples(conn)
        write_snapshots(conn)
        conn.close()
        return


if __name__ == "__main__":
    main()
