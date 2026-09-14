from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = []
    for run_dir_str in args.run_dir:
        run_dir = Path(run_dir_str).expanduser()
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with metrics_file.open("r", encoding="utf-8") as f:
                results.append(json.load(f))
        else:
            print(f"Warning: metrics.json not found in {run_dir}")

    if not results:
        print("Error: No metrics found to aggregate")
        return

    # Basic aggregation: just collect them for now, or average if they have common keys
    summary = {
        "num_seeds": len(results),
        "results": results,
    }

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregated results saved to {output_path}")


if __name__ == "__main__":
    main()
