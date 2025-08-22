#!/usr/bin/env python3
"""
Copy a single NAB CSV and labels into ./data/ from an existing NAB checkout.
Usage:
  python scripts/fetch_nab_file.py --nab-root ../NAB --rel-path realKnownCause/machine_temperature_system_failure.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nab-root", type=Path, required=True, help="Path to NAB checkout")
    ap.add_argument("--rel-path", type=str, required=True, help="Relative path within NAB/data/ to a CSV")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    src_csv = args.nab_root / "data" / args.rel_path
    src_labels = args.nab_root / "labels" / "combined_windows.json"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if src_csv.exists():
        shutil.copy2(src_csv, args.out_dir / Path(args.rel_path).name)
        if src_labels.exists():
            shutil.copy2(src_labels, args.out_dir / "combined_windows.json")
        print(f"Copied {src_csv} and labels (if present) to {args.out_dir}")
    else:
        print(f"CSV not found: {src_csv}")

if __name__ == "__main__":
    main()
