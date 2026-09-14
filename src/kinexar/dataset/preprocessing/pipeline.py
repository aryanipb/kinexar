from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from PREPROCESSING.bundle import build_validation_bundle
from PREPROCESSING.compress import compress_validation_bundle
from PREPROCESSING.manifest import build_shared_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AV2_ROOT = ROOT / "DATA" / "val"
DEFAULT_BUILD_ROOT = ROOT / "DATA"
DEFAULT_MANIFEST = ROOT / "DATA" / "manifests" / "val_manifest.json"
DEFAULT_PROCESSED = DEFAULT_BUILD_ROOT / "processed"
DEFAULT_BUNDLE = DEFAULT_BUILD_ROOT / "bundle"
DEFAULT_ARCHIVE = DEFAULT_BUILD_ROOT / "bundle.tar.zst"
DEFAULT_MAP_ROOT = DEFAULT_BUILD_ROOT / "map_cache"
DEFAULT_STATE = DEFAULT_BUILD_ROOT / "state"
STATE_FILE = "state.json"
PID_FILE = "pipeline.pid"
LOG_FILE = "pipeline.log"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_state(state_dir: Path) -> dict[str, object]:
    path = state_dir / STATE_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(state_dir: Path, **kwargs: object) -> None:
    ensure_dir(state_dir)
    current = read_state(state_dir)
    current.update(kwargs)
    current["updated_at"] = now()
    (state_dir / STATE_FILE).write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pid_running(state_dir: Path) -> int | None:
    pid_path = state_dir / PID_FILE
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def preprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["RW_AV2_V2_MANIFEST"] = str(args.manifest_path)
    env["RW_AV2_V2_OUT"] = str(args.processed_root)
    env["RW_AV2_V2_AV2_ROOT"] = str(args.av2_root)
    env["RW_AV2_V2_MAX_WORKERS"] = str(args.workers)
    pythonpath_parts = [str(ROOT), str(ROOT / "src")]
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    if current_pythonpath:
        pythonpath_parts.append(current_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return env


def run_preprocess(args: argparse.Namespace, state_dir: Path) -> None:
    write_state(state_dir, status="running", phase="preprocess", processed_root=str(args.processed_root))
    cmd = [
        sys.executable,
        str(ROOT / "PREPROCESSING" / "preprocess_av2_v2.py"),
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
    subprocess.run(cmd, cwd=str(ROOT), env=preprocess_env(args), check=True)


def run_pipeline(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_root).expanduser()
    ensure_dir(state_dir)
    (state_dir / PID_FILE).write_text(str(os.getpid()), encoding="utf-8")
    write_state(
        state_dir,
        status="running",
        phase="init",
        av2_root=str(args.av2_root),
        manifest_path=str(args.manifest_path),
        processed_root=str(args.processed_root),
        bundle_root=str(args.bundle_root),
        archive_path=str(args.archive_path),
        map_root=str(args.map_root),
    )
    try:
        write_state(state_dir, phase="manifest")
        manifest_path = build_shared_manifest(
            args.av2_root,
            args.manifest_path,
            history_steps=args.history_steps,
            future_steps=args.future_steps,
            force=args.force_manifest,
        )
        write_state(state_dir, manifest_ready=True, manifest_path=str(manifest_path))

        run_preprocess(args, state_dir)
        write_state(state_dir, preprocess_ready=True)

        write_state(state_dir, phase="bundle")
        bundle_result = build_validation_bundle(
            args.processed_root,
            args.bundle_root,
            chunk_rows=getattr(args, "bundle_chunk_rows", 1000),
            worker_count=args.bundle_workers,
            map_root=args.map_root,
            force=args.force_bundle,
            log_path=args.bundle_log_path,
        )
        write_state(state_dir, bundle_ready=True, bundle_result=bundle_result)

        write_state(state_dir, phase="compress")
        compress_result = compress_validation_bundle(
            args.bundle_root,
            args.archive_path,
            force=args.force_compress,
        )
        write_state(state_dir, phase="harvest_maps")
        harvest_maps(args.manifest_path, Path(args.av2_root), ROOT / "DATA" / "maps")

        write_state(state_dir, status="done", phase="done", compress_result=compress_result)
    except KeyboardInterrupt:
        write_state(state_dir, status="paused", phase=read_state(state_dir).get("phase", "unknown"))
        raise
    except Exception as exc:
        write_state(
            state_dir,
            status="failed",
            phase=read_state(state_dir).get("phase", "unknown"),
            last_error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        (state_dir / PID_FILE).unlink(missing_ok=True)

def harvest_maps(manifest_path: str | Path, av2_root: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if isinstance(manifest, list):
        rows = manifest
    elif isinstance(manifest, dict):
        rows = manifest.get("rows", [])
    else:
        rows = []

    unique_maps = set()
    for row in rows:
        sid = row.get("scenario_id")
        if not sid and "file_path" in row:
             sid = Path(row["file_path"]).stem.replace("scenario_", "", 1)
        
        if sid:
            # Look for map inside the provided av2_root where the reviewer downloaded data
            map_path = av2_root / sid / f"log_map_archive_{sid}.json"
            if map_path.exists():
                unique_maps.add(map_path)

    print(f"Harvesting {len(unique_maps)} map JSONs to {target_dir}...")
    import shutil
    for m in unique_maps:
        dest = target_dir / m.name
        if not dest.exists():
            shutil.copy2(m, dest)


def launch(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_root).expanduser()
    ensure_dir(state_dir)
    if pid_running(state_dir) is not None:
        raise ValueError("preprocessing1 pipeline already running")
    cmd = [
        sys.executable,
        "-m",
        "PREPROCESSING.pipeline",
        "run",
        "--av2-root",
        str(args.av2_root),
        "--manifest-path",
        str(args.manifest_path),
        "--processed-root",
        str(args.processed_root),
        "--bundle-root",
        str(args.bundle_root),
        "--archive-path",
        str(args.archive_path),
        "--map-root",
        str(args.map_root),
        "--state-root",
        str(args.state_root),
        "--history-steps",
        str(args.history_steps),
        "--future-steps",
        str(args.future_steps),
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
        "--bundle-chunk-rows",
        str(args.bundle_chunk_rows),
        "--bundle-workers",
        str(args.bundle_workers),
    ]
    if args.retry_failed:
        cmd.append("--retry-failed")
    if args.force_manifest:
        cmd.append("--force-manifest")
    if args.force_bundle:
        cmd.append("--force-bundle")
    if args.force_compress:
        cmd.append("--force-compress")
    if args.bundle_log_path:
        cmd.extend(["--bundle-log-path", str(args.bundle_log_path)])
    env = os.environ.copy()
    pythonpath_parts = [str(ROOT), str(ROOT / "src")]
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    if current_pythonpath:
        pythonpath_parts.append(current_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    log_path = state_dir / LOG_FILE
    with log_path.open("ab") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"started pid={proc.pid}")
    print(log_path)


def status(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_root).expanduser()
    state = read_state(state_dir)
    print(f"running_pid: {pid_running(state_dir)}")
    print(f"status: {state.get('status', '')}")
    print(f"phase: {state.get('phase', '')}")
    print(f"updated_at: {state.get('updated_at', '')}")
    print(f"av2_root: {args.av2_root}")
    print(f"manifest_path: {args.manifest_path}")
    print(f"processed_root: {args.processed_root}")
    print(f"bundle_root: {args.bundle_root}")
    print(f"archive_path: {args.archive_path}")
    print(f"map_root: {args.map_root}")
    print(f"pipeline_log: {state_dir / LOG_FILE}")
    print(f"preprocess_log: {Path(args.processed_root) / 'logs' / 'preprocess.log'}")
    print(f"bundle_log: {args.bundle_log_path or (Path(args.bundle_root) / 'logs' / 'bundle.log')}")
    if state.get("last_error"):
        print(f"last_error: {state['last_error']}")


def tail_log(path: Path, *, lines: int = 40, follow: bool = False, state_dir: Path | None = None) -> None:
    if not path.exists():
        print(f"missing log: {path}")
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        data = handle.readlines()
        for line in data[-lines:]:
            print(line.rstrip())
        if not follow:
            return
        while True:
            line = handle.readline()
            if line:
                print(line.rstrip())
                continue
            if state_dir is not None and pid_running(state_dir) is None:
                return
            time.sleep(1.0)


def signal_job(args: argparse.Namespace, sig: int) -> None:
    state_dir = Path(args.state_root).expanduser()
    pid = pid_running(state_dir)
    if pid is None:
        raise ValueError("no running preprocessing1 pipeline")
    os.killpg(pid, sig)
    print(f"sent {signal.Signals(sig).name} to process_group={pid}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(cur: argparse.ArgumentParser) -> None:
        cur.add_argument("--av2-root", default=str(DEFAULT_AV2_ROOT))
        cur.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
        cur.add_argument("--processed-root", default=str(DEFAULT_PROCESSED))
        cur.add_argument("--bundle-root", default=str(DEFAULT_BUNDLE))
        cur.add_argument("--archive-path", default=str(DEFAULT_ARCHIVE))
        cur.add_argument("--map-root", default=str(DEFAULT_MAP_ROOT))
        cur.add_argument("--state-root", default=str(DEFAULT_STATE))
        cur.add_argument("--history-steps", type=int, default=50)
        cur.add_argument("--future-steps", type=int, default=60)
        cur.add_argument("--workers", type=int, default=8)
        cur.add_argument("--shard-samples", type=int, default=64)
        cur.add_argument("--shard-bytes", type=int, default=512 * 1024 * 1024)
        cur.add_argument("--max-attempts", type=int, default=3)
        cur.add_argument("--snapshot-every", type=int, default=1)
        cur.add_argument("--bundle-workers", type=int, default=8)
        cur.add_argument("--bundle-chunk-rows", type=int, default=960)
        cur.add_argument("--bundle-log-path", default=None)
        cur.add_argument("--retry-failed", action="store_true")
        cur.add_argument("--force-manifest", action="store_true")
        cur.add_argument("--force-bundle", action="store_true")
        cur.add_argument("--force-compress", action="store_true")

    for name in ("run", "start", "resume", "status", "stop", "pause", "logs", "preprocess-logs", "bundle-logs"):
        cur = sub.add_parser(name)
        add_common(cur)
        if name in {"logs", "preprocess-logs", "bundle-logs"}:
            cur.add_argument("--lines", type=int, default=40)
            cur.add_argument("--follow", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "run":
        run_pipeline(args)
        return
    if args.cmd in {"start", "resume"}:
        launch(args)
        return
    if args.cmd == "status":
        status(args)
        return
    if args.cmd == "pause":
        signal_job(args, signal.SIGINT)
        return
    if args.cmd == "stop":
        signal_job(args, signal.SIGTERM)
        return
    state_dir = Path(args.state_root).expanduser()
    if args.cmd == "logs":
        tail_log(state_dir / LOG_FILE, lines=args.lines, follow=args.follow, state_dir=state_dir)
        return
    if args.cmd == "preprocess-logs":
        tail_log(Path(args.processed_root) / "logs" / "preprocess.log", lines=args.lines, follow=args.follow, state_dir=state_dir)
        return
    if args.cmd == "bundle-logs":
        bundle_log = Path(args.bundle_log_path).expanduser() if args.bundle_log_path else Path(args.bundle_root) / "logs" / "bundle.log"
        tail_log(bundle_log, lines=args.lines, follow=args.follow, state_dir=state_dir)
        return


if __name__ == "__main__":
    main()
