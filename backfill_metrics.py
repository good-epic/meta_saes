#!/usr/bin/env python3
"""
Backfill metrics.json from training.log files for completed runs.

This script parses training.log files and creates metrics.json in the same
format that train_meta_sae.py now outputs, so all runs have consistent metrics.

Usage:
    python backfill_metrics.py --input_dir outputs/grid_search_XXXXX
    python backfill_metrics.py --input_dir outputs/grid_search_XXXXX --dry_run
"""

import argparse
import json
import re
from pathlib import Path


def parse_training_log(log_file: Path) -> dict:
    """Extract final metrics from a training.log file."""
    metrics = {
        "joint": None,
        "solo": None,
        "sequential_meta": None,
        "config": None,
    }

    if not log_file.exists():
        return metrics

    try:
        content = log_file.read_text()

        # Find the last line of joint training (has "Decomp=")
        joint_pattern = r'Training SAE \+ Meta SAE.*?Loss=([0-9.]+).*?L0=([0-9.]+).*?L2=([0-9.]+).*?Decomp=([0-9.]+)'
        joint_matches = re.findall(joint_pattern, content)
        if joint_matches:
            last_match = joint_matches[-1]
            metrics["joint"] = {
                "loss": float(last_match[0]),
                "l0": float(last_match[1]),
                "l2": float(last_match[2]),
                "decomp": float(last_match[3]),
            }

        # Find the last line of solo training (has "dead=")
        solo_pattern = r'Solo Primary SAE Training.*?loss=([0-9.]+).*?l2=([0-9.]+).*?l1=([0-9.]+).*?l0=([0-9.]+).*?dead=([0-9]+)'
        solo_matches = re.findall(solo_pattern, content)
        if solo_matches:
            last_match = solo_matches[-1]
            metrics["solo"] = {
                "loss": float(last_match[0]),
                "l2": float(last_match[1]),
                "l1": float(last_match[2]),
                "l0": float(last_match[3]),
                "dead": int(last_match[4]),
            }

        # Find the last line of meta SAE training
        meta_pattern = r'Meta SAE Training.*?loss=([0-9.]+).*?l2=([0-9.]+).*?l0=([0-9.]+)'
        meta_matches = re.findall(meta_pattern, content)
        if meta_matches:
            last_match = meta_matches[-1]
            metrics["sequential_meta"] = {
                "loss": float(last_match[0]),
                "l2": float(last_match[1]),
                "l0": float(last_match[2]),
            }

    except Exception as e:
        print(f"Error parsing {log_file}: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Backfill metrics.json from training.log files")
    parser.add_argument("--input_dir", type=str, required=True, help="Grid search output directory")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be done without writing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metrics.json files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return

    # Find all run directories
    run_dirs = [d for d in input_dir.iterdir() if d.is_dir() and (d / "training.log").exists()]
    print(f"Found {len(run_dirs)} run directories with training.log")

    created = 0
    skipped = 0
    failed = 0

    for run_dir in sorted(run_dirs):
        log_file = run_dir / "training.log"
        metrics_file = run_dir / "metrics.json"
        params_file = run_dir / "params.json"

        # Skip if metrics.json already exists (unless --overwrite)
        if metrics_file.exists() and not args.overwrite:
            skipped += 1
            continue

        # Parse log
        metrics = parse_training_log(log_file)

        # Add config from params.json if available
        if params_file.exists():
            try:
                with open(params_file) as f:
                    params = json.load(f)
                metrics["config"] = {
                    "lambda2": params.get("lambda2"),
                    "sigma_sq": params.get("sigma_sq"),
                    "dict_size": params.get("dict_size"),
                    "meta_dict_size": params.get("meta_dict_size"),
                    "primary_sae_type": params.get("primary_sae_type"),
                    "meta_sae_type": params.get("meta_sae_type"),
                    "primary_top_k": params.get("primary_top_k"),
                    "meta_top_k": params.get("meta_top_k"),
                    "num_tokens": params.get("num_tokens"),
                }
            except Exception as e:
                print(f"Warning: Could not load params.json for {run_dir.name}: {e}")

        # Check if we got any metrics
        if metrics["joint"] is None and metrics["solo"] is None:
            print(f"  No metrics found in {run_dir.name}")
            failed += 1
            continue

        if args.dry_run:
            print(f"  Would create {metrics_file}")
            print(f"    joint: {metrics['joint']}")
            print(f"    solo: {metrics['solo']}")
            print(f"    sequential_meta: {metrics['sequential_meta']}")
        else:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            created += 1

    print(f"\nSummary:")
    print(f"  Created: {created}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed (no metrics found): {failed}")

    if args.dry_run:
        print("\n(Dry run - no files were written)")


if __name__ == "__main__":
    main()
