#!/usr/bin/env python3
"""
Analyze Grid Search Results for Meta-SAE Hyperparameter Tuning

This script aggregates results from a grid search and provides:
1. Summary statistics
2. Best hyperparameter combinations
3. Pareto frontier analysis
4. CSV export for further analysis

Usage:
    python analyze_grid_results.py --input_dir outputs/grid_search_001
    python analyze_grid_results.py --input_dir outputs/grid_search_001 --metric l2_loss
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch


def load_run_results(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load results from a single run directory."""
    params_file = run_dir / "params.json"
    result_file = run_dir / "result.json"

    if not params_file.exists() or not result_file.exists():
        return None

    try:
        with open(params_file) as f:
            params = json.load(f)
        with open(result_file) as f:
            result = json.load(f)

        if result.get("status") != "success":
            return None

        # Try to load final metrics from the saved models
        metrics = {}

        # Check for joint models
        joint_primary_path = run_dir / "joint_primary_sae.pt"
        if joint_primary_path.exists():
            try:
                checkpoint = torch.load(joint_primary_path, map_location="cpu")
                if "cfg" in checkpoint:
                    metrics["joint_cfg"] = checkpoint["cfg"]
            except:
                pass

        # Parse training log for final metrics
        log_file = run_dir / "training.log"
        if log_file.exists():
            metrics.update(parse_training_log(log_file))

        return {
            "run_id": run_dir.name,
            "params": params,
            "result": result,
            "metrics": metrics,
        }
    except Exception as e:
        print(f"Error loading {run_dir}: {e}")
        return None


def parse_training_log(log_file: Path) -> Dict[str, Any]:
    """Parse training log to extract final metrics."""
    metrics = {}

    try:
        with open(log_file) as f:
            lines = f.readlines()

        # Look for the last progress bar update with metrics
        for line in reversed(lines):
            if "Primary Loss" in line or "loss" in line.lower():
                # Try to extract metrics from tqdm output
                parts = line.split("|")
                for part in parts:
                    if ":" in part:
                        for kv in part.split(","):
                            kv = kv.strip()
                            if "=" in kv:
                                k, v = kv.split("=", 1)
                                try:
                                    metrics[k.strip()] = float(v.strip())
                                except:
                                    pass
                            elif ":" in kv:
                                k, v = kv.split(":", 1)
                                try:
                                    metrics[k.strip()] = float(v.strip())
                                except:
                                    pass
                if metrics:
                    break
    except:
        pass

    return metrics


def aggregate_results(input_dir: Path) -> List[Dict[str, Any]]:
    """Aggregate all results from the grid search."""
    results = []

    for run_dir in input_dir.iterdir():
        if run_dir.is_dir() and (run_dir / "params.json").exists():
            run_result = load_run_results(run_dir)
            if run_result:
                results.append(run_result)

    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("Grid Search Summary")
    print(f"{'='*60}")
    print(f"Total successful runs: {len(results)}")

    if not results:
        print("No successful runs found!")
        return

    # Compute average duration
    durations = [r["result"].get("duration_seconds", 0) for r in results if r["result"].get("duration_seconds")]
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"Average run duration: {avg_duration/60:.1f} minutes")

    # Count by parameter values
    print(f"\n{'='*60}")
    print("Runs by parameter value:")
    print(f"{'='*60}")

    param_counts = {}
    for r in results:
        for key in ["lambda2", "sigma_sq", "meta_dict_size", "primary_sae_type", "meta_sae_type"]:
            if key not in param_counts:
                param_counts[key] = {}
            val = r["params"].get(key)
            param_counts[key][val] = param_counts[key].get(val, 0) + 1

    for key, counts in param_counts.items():
        print(f"\n{key}:")
        for val, count in sorted(counts.items(), key=lambda x: str(x[0])):
            print(f"  {val}: {count} runs")


def find_best_runs(results: List[Dict[str, Any]], metric: str = "Primary Loss", top_k: int = 10) -> List[Dict[str, Any]]:
    """Find the best runs by a given metric (lower is better)."""
    # Filter to runs that have the metric
    runs_with_metric = []
    for r in results:
        value = r["metrics"].get(metric)
        if value is not None:
            runs_with_metric.append((value, r))

    if not runs_with_metric:
        print(f"No runs found with metric '{metric}'")
        return []

    # Sort by metric (lower is better)
    runs_with_metric.sort(key=lambda x: x[0])

    return [r for _, r in runs_with_metric[:top_k]]


def export_to_csv(results: List[Dict[str, Any]], output_file: Path):
    """Export results to CSV for analysis."""
    if not results:
        print("No results to export")
        return

    # Collect all possible columns
    param_keys = set()
    metric_keys = set()
    for r in results:
        param_keys.update(r["params"].keys())
        metric_keys.update(r["metrics"].keys())

    # Sort keys for consistent ordering
    param_keys = sorted(param_keys)
    metric_keys = sorted(metric_keys)

    with open(output_file, "w") as f:
        # Header
        headers = ["run_id", "status", "duration_seconds"] + param_keys + metric_keys
        f.write(",".join(headers) + "\n")

        # Data rows
        for r in results:
            row = [
                r["run_id"],
                r["result"].get("status", ""),
                str(r["result"].get("duration_seconds", "")),
            ]
            for key in param_keys:
                row.append(str(r["params"].get(key, "")))
            for key in metric_keys:
                row.append(str(r["metrics"].get(key, "")))
            f.write(",".join(row) + "\n")

    print(f"Exported {len(results)} results to {output_file}")


def analyze_by_hyperparameter(results: List[Dict[str, Any]], metric: str = "Primary Loss"):
    """Analyze how each hyperparameter affects the metric."""
    print(f"\n{'='*60}")
    print(f"Analysis by hyperparameter (metric: {metric})")
    print(f"{'='*60}")

    for param in ["lambda2", "sigma_sq", "meta_dict_size", "primary_sae_type", "meta_sae_type"]:
        print(f"\n{param}:")

        # Group by parameter value
        by_value = {}
        for r in results:
            val = r["params"].get(param)
            metric_val = r["metrics"].get(metric)
            if metric_val is not None:
                if val not in by_value:
                    by_value[val] = []
                by_value[val].append(metric_val)

        # Compute statistics
        for val in sorted(by_value.keys(), key=lambda x: str(x)):
            values = by_value[val]
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            print(f"  {val}: avg={avg:.4f}, min={min_val:.4f}, max={max_val:.4f} (n={len(values)})")


def main():
    parser = argparse.ArgumentParser(description="Analyze grid search results")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing grid search results")
    parser.add_argument("--metric", type=str, default="Primary Loss",
                        help="Metric to optimize (lower is better)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top runs to show")
    parser.add_argument("--export_csv", type=str, default=None,
                        help="Path to export results as CSV")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    print(f"Loading results from {input_dir}...")
    results = aggregate_results(input_dir)

    if not results:
        print("No successful runs found!")
        return

    # Print summary
    print_summary(results)

    # Analyze by hyperparameter
    analyze_by_hyperparameter(results, args.metric)

    # Find best runs
    print(f"\n{'='*60}")
    print(f"Top {args.top_k} runs by {args.metric}:")
    print(f"{'='*60}")

    best_runs = find_best_runs(results, args.metric, args.top_k)
    for i, r in enumerate(best_runs):
        metric_val = r["metrics"].get(args.metric, "N/A")
        params = r["params"]
        print(f"\n{i+1}. {r['run_id']}")
        print(f"   {args.metric}: {metric_val}")
        print(f"   lambda2={params.get('lambda2')}, sigma_sq={params.get('sigma_sq')}, "
              f"meta_dict_size={params.get('meta_dict_size')}")
        print(f"   primary_sae_type={params.get('primary_sae_type')}, "
              f"meta_sae_type={params.get('meta_sae_type')}")

    # Export if requested
    if args.export_csv:
        export_to_csv(results, Path(args.export_csv))
    else:
        # Default export
        csv_path = input_dir / "results.csv"
        export_to_csv(results, csv_path)


if __name__ == "__main__":
    main()
