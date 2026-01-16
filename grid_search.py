#!/usr/bin/env python3
"""
Grid Search Script for Meta-SAE Hyperparameter Tuning

This script runs a grid search over hyperparameters for meta-SAE training,
managing concurrent jobs to maximize GPU utilization.

Usage:
    python grid_search.py --num_workers 8 --output_dir outputs/grid_search_001
    python grid_search.py --dry_run  # Print all commands without running
    python grid_search.py --resume outputs/grid_search_001  # Resume incomplete runs

Example on RunPod A40 (48GB VRAM):
    python grid_search.py --num_workers 8 --output_dir outputs/grid_v1
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib


@dataclass
class GridConfig:
    """Configuration for the grid search."""
    # Grid parameters to sweep
    lambda2: List[float] = field(default_factory=lambda: [0.0, 0.001, 0.01, 0.1])
    sigma_sq: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0])
    meta_dict_size: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    primary_sae_type: List[str] = field(default_factory=lambda: ["batchtopk", "jumprelu"])
    meta_sae_type: List[str] = field(default_factory=lambda: ["batchtopk", "jumprelu"])

    # Fixed parameters (not part of grid)
    dict_size: int = 12288
    primary_top_k: int = 32
    meta_top_k: int = 4
    bandwidth: float = 0.001
    num_tokens: int = 100_000_000
    batch_size: int = 4096
    lr: float = 3e-4
    layer: int = 8
    site: str = "resid_pre"
    dataset_path: str = "HuggingFaceFW/fineweb"
    dataset_name: str = "sample-10BT"
    n_primary_steps: int = 10
    n_meta_steps: int = 5
    num_batches_in_buffer_joint: int = 5
    num_batches_in_buffer_sequential: int = 3
    model_batch_size: int = 256
    seq_len: int = 128


def generate_run_id(params: Dict[str, Any]) -> str:
    """Generate a unique, human-readable run ID from parameters."""
    # Create a short hash for uniqueness
    param_str = json.dumps(params, sort_keys=True)
    short_hash = hashlib.md5(param_str.encode()).hexdigest()[:6]

    # Create human-readable prefix
    parts = [
        f"l2_{params['lambda2']}",
        f"s2_{params['sigma_sq']}",
        f"md_{params['meta_dict_size']}",
        f"pt_{params['primary_sae_type'][:3]}",
        f"mt_{params['meta_sae_type'][:3]}",
    ]
    return "_".join(parts) + f"_{short_hash}"


def generate_grid(config: GridConfig) -> List[Dict[str, Any]]:
    """Generate all hyperparameter combinations from the grid config."""
    grid_params = {
        'lambda2': config.lambda2,
        'sigma_sq': config.sigma_sq,
        'meta_dict_size': config.meta_dict_size,
        'primary_sae_type': config.primary_sae_type,
        'meta_sae_type': config.meta_sae_type,
    }

    # Fixed parameters
    fixed_params = {
        'dict_size': config.dict_size,
        'primary_top_k': config.primary_top_k,
        'meta_top_k': config.meta_top_k,
        'bandwidth': config.bandwidth,
        'num_tokens': config.num_tokens,
        'batch_size': config.batch_size,
        'lr': config.lr,
        'layer': config.layer,
        'site': config.site,
        'dataset_path': config.dataset_path,
        'dataset_name': config.dataset_name,
        'n_primary_steps': config.n_primary_steps,
        'n_meta_steps': config.n_meta_steps,
        'num_batches_in_buffer_joint': config.num_batches_in_buffer_joint,
        'num_batches_in_buffer_sequential': config.num_batches_in_buffer_sequential,
        'model_batch_size': config.model_batch_size,
        'seq_len': config.seq_len,
    }

    # Generate all combinations
    keys = list(grid_params.keys())
    combinations = list(itertools.product(*[grid_params[k] for k in keys]))

    runs = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        params.update(fixed_params)
        params['run_id'] = generate_run_id(params)
        runs.append(params)

    return runs


def build_command(params: Dict[str, Any], output_dir: Path, device: str = "cuda:0") -> List[str]:
    """Build the training command for a single run."""
    run_dir = output_dir / params['run_id']

    cmd = [
        sys.executable, "train_meta_sae.py",
        "--dataset_path", params['dataset_path'],
        "--dataset_name", params['dataset_name'],
        "--layer", str(params['layer']),
        "--site", params['site'],
        "--dict_size", str(params['dict_size']),
        "--meta_dict_size", str(params['meta_dict_size']),
        "--primary_top_k", str(params['primary_top_k']),
        "--meta_top_k", str(params['meta_top_k']),
        "--primary_sae_type", params['primary_sae_type'],
        "--meta_sae_type", params['meta_sae_type'],
        "--bandwidth", str(params['bandwidth']),
        "--num_tokens", str(params['num_tokens']),
        "--batch_size", str(params['batch_size']),
        "--lr", str(params['lr']),
        "--lambda2", str(params['lambda2']),
        "--sigma_sq", str(params['sigma_sq']),
        "--n_primary_steps", str(params['n_primary_steps']),
        "--n_meta_steps", str(params['n_meta_steps']),
        "--num_batches_in_buffer_joint", str(params['num_batches_in_buffer_joint']),
        "--num_batches_in_buffer_sequential", str(params['num_batches_in_buffer_sequential']),
        "--model_batch_size", str(params['model_batch_size']),
        "--seq_len", str(params['seq_len']),
        "--device", device,
        "--joint_primary_path", str(run_dir / "joint_primary_sae.pt"),
        "--joint_meta_path", str(run_dir / "joint_meta_sae.pt"),
        "--solo_primary_path", str(run_dir / "solo_primary_sae.pt"),
        "--sequential_meta_path", str(run_dir / "sequential_meta_sae.pt"),
        "--train_joint_saes",
        "--train_sequential_saes",
    ]

    return cmd


def run_single_job(
    params: Dict[str, Any],
    output_dir: Path,
    device: str,
    job_index: int,
    total_jobs: int,
) -> Dict[str, Any]:
    """Run a single training job and return results."""
    run_id = params['run_id']
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save params
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Build command
    cmd = build_command(params, output_dir, device)

    # Log file
    log_file = run_dir / "training.log"

    result = {
        "run_id": run_id,
        "params": params,
        "device": device,
        "status": "unknown",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "error": None,
    }

    print(f"[{job_index + 1}/{total_jobs}] Starting {run_id} on {device}")

    start_time = time.time()

    try:
        # Set environment to use specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")

        with open(log_file, "w") as log:
            # Write command for reference
            log.write(f"Command: {' '.join(cmd)}\n")
            log.write(f"Started: {result['start_time']}\n")
            log.write("=" * 60 + "\n\n")
            log.flush()

            # Run with modified device (since CUDA_VISIBLE_DEVICES remaps)
            cmd_modified = [c if c != device else "cuda:0" for c in cmd]

            process = subprocess.run(
                cmd_modified,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )

        end_time = time.time()
        result["end_time"] = datetime.now().isoformat()
        result["duration_seconds"] = end_time - start_time

        if process.returncode == 0:
            result["status"] = "success"
            print(f"[{job_index + 1}/{total_jobs}] Completed {run_id} in {result['duration_seconds']:.1f}s")
        else:
            result["status"] = "failed"
            result["error"] = f"Process exited with code {process.returncode}"
            print(f"[{job_index + 1}/{total_jobs}] FAILED {run_id} (exit code {process.returncode})")

    except Exception as e:
        end_time = time.time()
        result["end_time"] = datetime.now().isoformat()
        result["duration_seconds"] = end_time - start_time
        result["status"] = "error"
        result["error"] = str(e)
        print(f"[{job_index + 1}/{total_jobs}] ERROR {run_id}: {e}")

    # Save result
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def get_completed_runs(output_dir: Path) -> set:
    """Get set of run IDs that have already completed successfully."""
    completed = set()
    if not output_dir.exists():
        return completed

    for run_dir in output_dir.iterdir():
        if run_dir.is_dir():
            result_file = run_dir / "result.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        result = json.load(f)
                    if result.get("status") == "success":
                        completed.add(run_dir.name)
                except:
                    pass

    return completed


def main():
    parser = argparse.ArgumentParser(description="Grid search for meta-SAE hyperparameters")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of concurrent training jobs")
    parser.add_argument("--output_dir", type=str, default="outputs/grid_search",
                        help="Directory for all run outputs")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already completed runs")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file overriding defaults")

    # Allow overriding grid parameters
    parser.add_argument("--lambda2", type=float, nargs="+", default=None)
    parser.add_argument("--sigma_sq", type=float, nargs="+", default=None)
    parser.add_argument("--meta_dict_size", type=int, nargs="+", default=None)
    parser.add_argument("--primary_sae_type", type=str, nargs="+", default=None)
    parser.add_argument("--meta_sae_type", type=str, nargs="+", default=None)

    # Fixed parameter overrides
    parser.add_argument("--num_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # Create config
    config = GridConfig()

    # Load from file if provided
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)

    # Apply command-line overrides
    if args.lambda2 is not None:
        config.lambda2 = args.lambda2
    if args.sigma_sq is not None:
        config.sigma_sq = args.sigma_sq
    if args.meta_dict_size is not None:
        config.meta_dict_size = args.meta_dict_size
    if args.primary_sae_type is not None:
        config.primary_sae_type = args.primary_sae_type
    if args.meta_sae_type is not None:
        config.meta_sae_type = args.meta_sae_type
    if args.num_tokens is not None:
        config.num_tokens = args.num_tokens
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # Generate all runs
    all_runs = generate_grid(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save grid config
    with open(output_dir / "grid_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Filter out completed runs if resuming
    if args.resume:
        completed = get_completed_runs(output_dir)
        runs_to_do = [r for r in all_runs if r['run_id'] not in completed]
        print(f"Resuming: {len(completed)} already completed, {len(runs_to_do)} remaining")
    else:
        runs_to_do = all_runs

    print(f"\n{'='*60}")
    print(f"Grid Search Configuration")
    print(f"{'='*60}")
    print(f"Total combinations: {len(all_runs)}")
    print(f"Runs to execute: {len(runs_to_do)}")
    print(f"Concurrent workers: {args.num_workers}")
    print(f"Output directory: {output_dir}")
    print(f"\nGrid parameters:")
    print(f"  lambda2: {config.lambda2}")
    print(f"  sigma_sq: {config.sigma_sq}")
    print(f"  meta_dict_size: {config.meta_dict_size}")
    print(f"  primary_sae_type: {config.primary_sae_type}")
    print(f"  meta_sae_type: {config.meta_sae_type}")
    print(f"\nFixed parameters:")
    print(f"  dict_size: {config.dict_size}")
    print(f"  num_tokens: {config.num_tokens:,}")
    print(f"  bandwidth: {config.bandwidth}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("DRY RUN - Commands that would be executed:\n")
        for i, params in enumerate(runs_to_do[:5]):  # Show first 5
            cmd = build_command(params, output_dir, "cuda:0")
            print(f"[{i+1}] {params['run_id']}")
            print(f"    {' '.join(cmd[:20])}...")
            print()
        if len(runs_to_do) > 5:
            print(f"... and {len(runs_to_do) - 5} more runs")
        return

    if len(runs_to_do) == 0:
        print("No runs to execute!")
        return

    # Save manifest of all runs
    with open(output_dir / "runs_manifest.json", "w") as f:
        json.dump(runs_to_do, f, indent=2)

    # Track results
    all_results = []
    start_time = time.time()

    # Run jobs with process pool
    # Note: We use CUDA_VISIBLE_DEVICES to isolate GPUs, so all jobs use "cuda:0"
    # but see different physical GPUs
    print(f"Starting {len(runs_to_do)} jobs with {args.num_workers} workers...\n")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all jobs - they'll use cuda:0 but with different CUDA_VISIBLE_DEVICES
        futures = {
            executor.submit(
                run_single_job,
                params,
                output_dir,
                "cuda:0",  # Will be remapped via CUDA_VISIBLE_DEVICES
                i,
                len(runs_to_do),
            ): params
            for i, params in enumerate(runs_to_do)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            params = futures[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Job {params['run_id']} raised exception: {e}")
                all_results.append({
                    "run_id": params['run_id'],
                    "status": "exception",
                    "error": str(e),
                })

    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r.get("status") == "success")
    failed = sum(1 for r in all_results if r.get("status") in ("failed", "error", "exception"))

    print(f"\n{'='*60}")
    print(f"Grid Search Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Successful: {successful}/{len(all_results)}")
    print(f"Failed: {failed}/{len(all_results)}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "total_runs": len(all_results),
        "successful": successful,
        "failed": failed,
        "total_time_seconds": total_time,
        "results": all_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
