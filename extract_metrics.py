#!/usr/bin/env python3
"""
Extract metrics from training.log files in grid search results.

Usage:
    python extract_metrics.py --input_dir outputs/grid_search_XXXXX
    python extract_metrics.py --input_dir outputs/grid_search_XXXXX --output metrics.csv
"""

import argparse
import json
import re
from pathlib import Path
import csv


def load_metrics_json(metrics_file: Path) -> dict:
    """Load metrics from metrics.json if it exists."""
    metrics = {
        "joint_loss": None,
        "joint_l0": None,
        "joint_l2": None,
        "joint_decomp": None,
        "solo_loss": None,
        "solo_l0": None,
        "solo_l2": None,
        "solo_l1": None,
        "solo_dead": None,
        "meta_loss": None,
        "meta_l0": None,
        "meta_l2": None,
    }

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file) as f:
            data = json.load(f)

        if data.get("joint"):
            metrics["joint_loss"] = data["joint"].get("loss")
            metrics["joint_l0"] = data["joint"].get("l0")
            metrics["joint_l2"] = data["joint"].get("l2")
            metrics["joint_decomp"] = data["joint"].get("decomp")

        if data.get("solo"):
            metrics["solo_loss"] = data["solo"].get("loss")
            metrics["solo_l0"] = data["solo"].get("l0")
            metrics["solo_l2"] = data["solo"].get("l2")
            metrics["solo_l1"] = data["solo"].get("l1")
            metrics["solo_dead"] = data["solo"].get("dead")

        if data.get("sequential_meta"):
            metrics["meta_loss"] = data["sequential_meta"].get("loss")
            metrics["meta_l0"] = data["sequential_meta"].get("l0")
            metrics["meta_l2"] = data["sequential_meta"].get("l2")

        return metrics
    except Exception as e:
        print(f"Error loading {metrics_file}: {e}")
        return None


def parse_training_log(log_file: Path) -> dict:
    """Extract final metrics from a training.log file."""
    metrics = {
        "joint_loss": None,
        "joint_l0": None,
        "joint_l2": None,
        "joint_decomp": None,
        "solo_loss": None,
        "solo_l0": None,
        "solo_l2": None,
        "solo_l1": None,
        "solo_dead": None,
        "meta_loss": None,
        "meta_l0": None,
        "meta_l2": None,
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
            metrics["joint_loss"] = float(last_match[0])
            metrics["joint_l0"] = float(last_match[1])
            metrics["joint_l2"] = float(last_match[2])
            metrics["joint_decomp"] = float(last_match[3])

        # Find the last line of solo training (has "dead=")
        solo_pattern = r'Solo Primary SAE Training.*?loss=([0-9.]+).*?l2=([0-9.]+).*?l1=([0-9.]+).*?l0=([0-9.]+).*?dead=([0-9]+)'
        solo_matches = re.findall(solo_pattern, content)
        if solo_matches:
            last_match = solo_matches[-1]
            metrics["solo_loss"] = float(last_match[0])
            metrics["solo_l2"] = float(last_match[1])
            metrics["solo_l1"] = float(last_match[2])
            metrics["solo_l0"] = float(last_match[3])
            metrics["solo_dead"] = int(last_match[4])

        # Find the last line of meta SAE training
        meta_pattern = r'Meta SAE Training.*?loss=([0-9.]+).*?l2=([0-9.]+).*?l0=([0-9.]+)'
        meta_matches = re.findall(meta_pattern, content)
        if meta_matches:
            last_match = meta_matches[-1]
            metrics["meta_loss"] = float(last_match[0])
            metrics["meta_l2"] = float(last_match[1])
            metrics["meta_l0"] = float(last_match[2])

    except Exception as e:
        print(f"Error parsing {log_file}: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Extract metrics from grid search logs")
    parser.add_argument("--input_dir", type=str, required=True, help="Grid search output directory")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file (default: input_dir/metrics.csv)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return

    output_file = Path(args.output) if args.output else input_dir / "metrics.csv"

    # Find all run directories
    results = []
    run_dirs = [d for d in input_dir.iterdir() if d.is_dir() and (d / "params.json").exists()]

    print(f"Found {len(run_dirs)} run directories")

    for run_dir in sorted(run_dirs):
        # Load params
        params_file = run_dir / "params.json"
        log_file = run_dir / "training.log"
        result_file = run_dir / "result.json"
        metrics_file = run_dir / "metrics.json"

        try:
            with open(params_file) as f:
                params = json.load(f)
        except:
            continue

        # Check if completed
        status = "unknown"
        if result_file.exists():
            try:
                with open(result_file) as f:
                    result = json.load(f)
                status = result.get("status", "unknown")
            except:
                pass

        # Try to load from metrics.json first, fall back to parsing log
        metrics = load_metrics_json(metrics_file)
        if metrics is None:
            metrics = parse_training_log(log_file)

        # Combine
        row = {
            "run_id": run_dir.name,
            "status": status,
            "lambda2": params.get("lambda2"),
            "sigma_sq": params.get("sigma_sq"),
            "meta_dict_size": params.get("meta_dict_size"),
            "primary_sae_type": params.get("primary_sae_type"),
            "meta_sae_type": params.get("meta_sae_type"),
            **metrics
        }
        results.append(row)

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {len(results)} rows to {output_file}")

        # Print summary
        completed = [r for r in results if r["status"] == "success" and r["joint_loss"] is not None]
        print(f"\nCompleted runs with metrics: {len(completed)}")

        if completed:
            print("\nSample of results:")
            print(f"{'lambda2':>8} {'sigma_sq':>8} {'meta_dict':>9} {'joint_loss':>10} {'joint_decomp':>12} {'solo_loss':>10}")
            print("-" * 70)
            for r in completed[:10]:
                print(f"{r['lambda2']:>8} {r['sigma_sq']:>8} {r['meta_dict_size']:>9} {r['joint_loss']:>10.4f} {r['joint_decomp']:>12.4f} {r['solo_loss']:>10.4f}")
    else:
        print("No results found")


if __name__ == "__main__":
    main()
