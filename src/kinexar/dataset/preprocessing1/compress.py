from __future__ import annotations

import json
from pathlib import Path
import subprocess


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    proc = subprocess.run(
        ["sha256sum", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip().split()[0]


def compress_validation_bundle(
    bundle_root: str | Path,
    archive_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, object]:
    bundle_root = Path(bundle_root).expanduser()
    archive_path = Path(archive_path).expanduser()
    if not bundle_root.exists() or not (bundle_root / "val_rows.pt").exists() or not (bundle_root / "val_parts").exists():
        raise FileNotFoundError(f"missing validation bundle outputs under {bundle_root}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and archive_path.exists():
        result = {
            "archive_path": str(archive_path),
            "status": "reused",
        }
        _write_json(archive_path.parent / "compression_manifest.json", result)
        return result
    archive_path.unlink(missing_ok=True)
    subprocess.run(
        ["tar", "-C", str(bundle_root.parent), "-I", "zstd -1 -T0", "-cf", str(archive_path), bundle_root.name],
        check=True,
    )
    sha = _sha256_file(archive_path)
    (archive_path.parent / f"{archive_path.name}.sha256").write_text(f"{sha}  {archive_path}\n", encoding="utf-8")
    result = {
        "archive_path": str(archive_path),
        "bytes": int(archive_path.stat().st_size),
    }
    _write_json(archive_path.parent / "compression_manifest.json", result)
    return result
