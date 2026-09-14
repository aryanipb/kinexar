import json
import os
from pathlib import Path

MANIFEST_PATH = "data/manifests/av2_manifest1.json"
NEW_ROOT = "/home/aryan/work/startup/datasetpredmodel/train"

def relocate():
    print(f"Loading manifest from {MANIFEST_PATH}...")
    with open(MANIFEST_PATH, "r") as f:
        rows = []
        for line in f:
            line = line.strip()
            if not line or line in ["[", "]"]:
                continue
            if line.endswith(","):
                line = line[:-1]
            rows.append(json.loads(line))
    
    print(f"Total samples: {len(rows)}")
    found_count = 0
    missing_count = 0
    
    for row in rows:
        scenario_id = row["scenario_id"]
        filename = f"scenario_{scenario_id}.parquet"
        
        # Check in local root
        local_path = Path(NEW_ROOT) / scenario_id / filename
        if local_path.exists():
            row["file_path"] = str(local_path)
            found_count += 1
        else:
            if missing_count < 10:
                print(f"Missing: {scenario_id} at {local_path}")
            missing_count += 1

    print(f"Relocated {found_count} samples. Missing {missing_count}.")
    
    if missing_count == 0:
        with open(MANIFEST_PATH, "w") as f:
            f.write("[\n")
            for i, row in enumerate(rows):
                comma = "," if i < len(rows) - 1 else ""
                f.write(json.dumps(row) + comma + "\n")
            f.write("]\n")
        print("Manifest updated successfully.")
    else:
        print("Manifest NOT updated due to missing files. Verify the local path.")

if __name__ == "__main__":
    relocate()
