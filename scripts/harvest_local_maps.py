import json
import shutil
import os
import argparse
from pathlib import Path

def harvest_maps(manifest_path, target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading manifest from {manifest_path}...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    # Handle both list and dict manifest formats
    if isinstance(manifest, list):
        rows = manifest
    elif isinstance(manifest, dict):
        rows = manifest.get("rows", [])
    else:
        raise ValueError(f"Unsupported manifest format: {type(manifest)}")
        
    unique_maps = set()
    print(f"Scanning {len(rows)} rows for map references...")
    
    for row in rows:
        if "file_path" in row:
            p = Path(row["file_path"])
            # In Argoverse 2, map is usually log_map_archive_<scenario_id>.json
            # It lives in the same folder as the parquet.
            sid = row.get("scenario_id") or p.stem.replace("scenario_", "", 1)
            map_path = p.with_name(f"log_map_archive_{sid}.json")
            if map_path.exists():
                unique_maps.add(map_path)
    
    if not unique_maps:
        print("WARNING: No map files found! Please check if the raw AV2 dataset is located at the paths specified in the manifest.")
        return

    print(f"Found {len(unique_maps)} unique map files. Starting copy...")
    
    count = 0
    for m in unique_maps:
        dest = target_dir / m.name
        if not dest.exists():
            shutil.copy2(m, dest)
            count += 1
            if count % 1000 == 0:
                print(f"Copied {count} maps...")
                
    print(f"Done! Successfully harvested {len(unique_maps)} maps into {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/av2_val_manifest.json")
    parser.add_argument("--output", default="local_map_harvest")
    args = parser.parse_args()
    harvest_maps(args.manifest, args.output)
