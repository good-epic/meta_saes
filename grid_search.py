#!/usr/bin/env python3
"""
Grid Search Script for Meta-SAE Hyperparameter Tuning

This script runs a grid search over hyperparameters for meta-SAE training,
managing concurrent jobs to maximize GPU utilization.

Usage:
    # Basic run with new defaults (k=64, matched L0)
    python grid_search.py --num_workers 8 --output_dir outputs/grid_search_v2

    # Specify grid parameters via CLI
    python grid_search.py --lambda2 0.0 0.01 0.1 1.0 --sigma_sq 0.1

    # Use GPT2-medium instead of small
    python grid_search.py --model_name gpt2-medium --layer 12

    # Find JumpReLU threshold for target L0 before running grid
    python grid_search.py --target_l0 64 --calibrate_jumprelu

    # Or specify JumpReLU initial threshold directly
    python grid_search.py --jumprelu_init_threshold -0.5

    # Dry run to see commands
    python grid_search.py --dry_run

    # Resume incomplete runs
    python grid_search.py --resume --output_dir outputs/grid_search_v2
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
    lambda2: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.1, 1.0])
    sigma_sq: List[float] = field(default_factory=lambda: [0.1, 1.0])
    meta_dict_size: List[int] = field(default_factory=lambda: [1800])
    # SAE architecture - when using matched mode, both primary and meta use the same type
    sae_type: List[str] = field(default_factory=lambda: ["batchtopk", "jumprelu"])
    # Set to True to match primary and meta architectures (recommended)
    # Set to False to allow all combinations (primary_sae_type × meta_sae_type)
    match_architectures: bool = True
    # These are only used if match_architectures=False
    primary_sae_type: List[str] = field(default_factory=lambda: ["batchtopk", "jumprelu"])
    meta_sae_type: List[str] = field(default_factory=lambda: ["batchtopk", "jumprelu"])

    # Fixed parameters (not part of grid)
    model_name: str = "gpt2-large"
    dict_size: int = 20480  # 16x residual stream (1280)
    primary_top_k: int = 64
    meta_top_k: int = 8
    bandwidth: float = 0.0001  # Steeper JumpReLU
    num_tokens: int = 100_000_000
    batch_size: int = 4096
    lr: float = 3e-4
    layer: int = 20  # ~56% through GPT-2 Large (36 layers)
    site: str = "resid_pre"
    dataset_path: str = "HuggingFaceFW/fineweb"
    dataset_name: str = "sample-10BT"
    n_primary_steps: int = 10
    n_meta_steps: int = 5
    num_batches_in_buffer_joint: int = 5
    num_batches_in_buffer_sequential: int = 3
    model_batch_size: int = 256
    seq_len: int = 128

    # JumpReLU sparsity control (two modes):
    # Fixed mode: set l0_coeff directly
    l0_coeff: Optional[float] = None

    # Dynamic mode: set target_l0, coefficient adapts during training
    target_l0: Optional[int] = 64  # Target L0 for adaptive sparsity (match BatchTopK)
    l0_coeff_start: float = 1e-5  # Initial coefficient for dynamic mode
    l0_coeff_lr: float = 1e-4  # Learning rate for coefficient updates

    # Initial threshold value (affects starting L0 before adaptation)
    jumprelu_init_threshold: Optional[float] = None


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
    if config.match_architectures:
        # Matched mode: primary and meta use the same architecture
        grid_params = {
            'lambda2': config.lambda2,
            'sigma_sq': config.sigma_sq,
            'meta_dict_size': config.meta_dict_size,
            'sae_type': config.sae_type,  # Single param for both
        }
    else:
        # Cross-product mode: all combinations of primary and meta
        grid_params = {
            'lambda2': config.lambda2,
            'sigma_sq': config.sigma_sq,
            'meta_dict_size': config.meta_dict_size,
            'primary_sae_type': config.primary_sae_type,
            'meta_sae_type': config.meta_sae_type,
        }

    # Fixed parameters
    fixed_params = {
        'model_name': config.model_name,
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

    if config.jumprelu_init_threshold is not None:
        fixed_params['jumprelu_init_threshold'] = config.jumprelu_init_threshold

    # JumpReLU sparsity control
    if config.l0_coeff is not None:
        fixed_params['l0_coeff'] = config.l0_coeff
    if config.target_l0 is not None:
        fixed_params['target_l0'] = config.target_l0
        fixed_params['l0_coeff_start'] = config.l0_coeff_start
        fixed_params['l0_coeff_lr'] = config.l0_coeff_lr

    # Generate all combinations
    keys = list(grid_params.keys())
    combinations = list(itertools.product(*[grid_params[k] for k in keys]))

    runs = []
    for combo in combinations:
        params = dict(zip(keys, combo))

        # In matched mode, expand 'sae_type' to both primary and meta
        if config.match_architectures and 'sae_type' in params:
            params['primary_sae_type'] = params['sae_type']
            params['meta_sae_type'] = params['sae_type']
            del params['sae_type']

        params.update(fixed_params)
        params['run_id'] = generate_run_id(params)
        runs.append(params)

    return runs


def build_command(params: Dict[str, Any], output_dir: Path, device: str = "cuda:0") -> List[str]:
    """Build the training command for a single run."""
    run_dir = output_dir / params['run_id']

    cmd = [
        sys.executable, "train_meta_sae.py",
        "--model_name", params['model_name'],
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

    # Add JumpReLU parameters if specified
    if 'jumprelu_init_threshold' in params and params['jumprelu_init_threshold'] is not None:
        cmd.extend(["--jumprelu_init_threshold", str(params['jumprelu_init_threshold'])])

    # JumpReLU sparsity control
    if 'l0_coeff' in params and params['l0_coeff'] is not None:
        cmd.extend(["--l0_coeff", str(params['l0_coeff'])])
    if 'target_l0' in params and params['target_l0'] is not None:
        cmd.extend(["--target_l0", str(params['target_l0'])])
        cmd.extend(["--l0_coeff_start", str(params['l0_coeff_start'])])
        cmd.extend(["--l0_coeff_lr", str(params['l0_coeff_lr'])])

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


def calibrate_jumprelu_threshold(config: GridConfig, device: str = "cuda:0") -> float:
    """
    Run a short calibration to find JumpReLU init threshold for target L0.

    Returns the log_threshold value that achieves approximately target_l0.
    """
    if config.target_l0 is None:
        raise ValueError("target_l0 must be set for calibration")

    print(f"\n{'='*60}")
    print(f"Calibrating JumpReLU threshold for target L0 = {config.target_l0}")
    print(f"{'='*60}\n")

    # This would run a short training and binary search
    # For now, provide guidance on manual calibration
    print("JumpReLU threshold calibration:")
    print(f"  - Target L0: {config.target_l0}")
    print(f"  - Bandwidth: {config.bandwidth}")
    print(f"  - Model: {config.model_name}")
    print()
    print("To calibrate manually:")
    print("  1. Run a short training with JumpReLU")
    print("  2. Observe the L0 value")
    print("  3. Adjust --jumprelu_init_threshold:")
    print("     - Higher threshold (e.g., 0.5) → lower L0")
    print("     - Lower threshold (e.g., -1.0) → higher L0")
    print()
    print("Typical values for L0 ≈ 64:")
    print("  - GPT2-small: try --jumprelu_init_threshold -0.5 to 0.5")
    print()

    # TODO: Implement automatic binary search calibration
    raise NotImplementedError(
        "Automatic calibration not yet implemented. "
        "Please specify --jumprelu_init_threshold directly based on manual testing."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for meta-SAE hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults
  python grid_search.py --output_dir outputs/grid_v2

  # Customize grid parameters
  python grid_search.py --lambda2 0.0 0.1 1.0 --sigma_sq 0.1

  # Use different model
  python grid_search.py --model_name gpt2-medium --layer 12

  # Specify JumpReLU threshold
  python grid_search.py --jumprelu_init_threshold 0.0
        """
    )

    # Execution options
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of concurrent training jobs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for outputs (default: auto-generated with timestamp)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already completed runs")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file overriding defaults")

    # Grid parameters (lists)
    parser.add_argument("--lambda2", type=float, nargs="+", default=None,
                        help="Composability penalty weights (default: [0.0, 0.01, 0.1, 1.0])")
    parser.add_argument("--sigma_sq", type=float, nargs="+", default=None,
                        help="Penalty variance values (default: [0.1])")
    parser.add_argument("--meta_dict_size", type=int, nargs="+", default=None,
                        help="Meta SAE dictionary sizes (default: [1024])")
    parser.add_argument("--sae_type", type=str, nargs="+", default=None,
                        choices=["batchtopk", "jumprelu"],
                        help="SAE architecture types - used for both primary and meta (default: [batchtopk, jumprelu])")
    parser.add_argument("--no_match_architectures", action="store_true",
                        help="Allow mismatched primary/meta architectures (creates 4x more runs)")
    # These are only used with --no_match_architectures
    parser.add_argument("--primary_sae_type", type=str, nargs="+", default=None,
                        choices=["batchtopk", "jumprelu"],
                        help="Primary SAE types (only with --no_match_architectures)")
    parser.add_argument("--meta_sae_type", type=str, nargs="+", default=None,
                        choices=["batchtopk", "jumprelu"],
                        help="Meta SAE types (only with --no_match_architectures)")

    # Fixed parameters (single values)
    parser.add_argument("--model_name", type=str, default=None,
                        choices=["gpt2-small", "gpt2-medium", "gpt2-large"],
                        help="Model to train on (default: gpt2-small)")
    parser.add_argument("--dict_size", type=int, default=None,
                        help="Primary SAE dictionary size (default: 12288)")
    parser.add_argument("--primary_top_k", type=int, default=None,
                        help="Top-K for BatchTopK / target L0 (default: 64)")
    parser.add_argument("--meta_top_k", type=int, default=None,
                        help="Meta SAE Top-K (default: 8)")
    parser.add_argument("--bandwidth", type=float, default=None,
                        help="JumpReLU bandwidth - lower = steeper (default: 0.0001)")
    parser.add_argument("--num_tokens", type=int, default=None,
                        help="Training tokens (default: 100M)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="SAE batch size (default: 4096)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Model layer to hook (default: 8)")
    parser.add_argument("--site", type=str, default=None,
                        help="Hook site (default: resid_pre)")
    parser.add_argument("--n_primary_steps", type=int, default=None,
                        help="Primary SAE steps per alternation (default: 10)")
    parser.add_argument("--n_meta_steps", type=int, default=None,
                        help="Meta SAE steps per alternation (default: 5)")

    # JumpReLU sparsity control
    parser.add_argument("--jumprelu_init_threshold", type=float, default=None,
                        help="Initial threshold value for JumpReLU features (default: 0.001)")

    # Fixed mode: set l0_coeff directly
    parser.add_argument("--l0_coeff", type=float, default=None,
                        help="Fixed L0 sparsity coefficient (if set, uses fixed mode)")

    # Dynamic mode: coefficient adapts to achieve target_l0
    parser.add_argument("--target_l0", type=int, default=None,
                        help="Target L0 sparsity for dynamic coefficient adaptation")
    parser.add_argument("--l0_coeff_start", type=float, default=None,
                        help="Initial L0 coefficient for dynamic mode (default: 1e-5)")
    parser.add_argument("--l0_coeff_lr", type=float, default=None,
                        help="Learning rate for L0 coefficient updates (default: 1e-4)")

    # Legacy option (not needed with dynamic mode)
    parser.add_argument("--calibrate_jumprelu", action="store_true",
                        help="[Deprecated] Use --target_l0 instead for dynamic L0 targeting")

    args = parser.parse_args()

    # Create config with defaults
    config = GridConfig()

    # Load from file if provided
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)

    # Apply command-line overrides for grid parameters
    if args.lambda2 is not None:
        config.lambda2 = args.lambda2
    if args.sigma_sq is not None:
        config.sigma_sq = args.sigma_sq
    if args.meta_dict_size is not None:
        config.meta_dict_size = args.meta_dict_size
    if args.sae_type is not None:
        config.sae_type = args.sae_type
    if args.no_match_architectures:
        config.match_architectures = False
    if args.primary_sae_type is not None:
        config.primary_sae_type = args.primary_sae_type
    if args.meta_sae_type is not None:
        config.meta_sae_type = args.meta_sae_type

    # Apply command-line overrides for fixed parameters
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.dict_size is not None:
        config.dict_size = args.dict_size
    if args.primary_top_k is not None:
        config.primary_top_k = args.primary_top_k
    if args.meta_top_k is not None:
        config.meta_top_k = args.meta_top_k
    if args.bandwidth is not None:
        config.bandwidth = args.bandwidth
    if args.num_tokens is not None:
        config.num_tokens = args.num_tokens
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.layer is not None:
        config.layer = args.layer
    if args.site is not None:
        config.site = args.site
    if args.n_primary_steps is not None:
        config.n_primary_steps = args.n_primary_steps
    if args.n_meta_steps is not None:
        config.n_meta_steps = args.n_meta_steps
    if args.jumprelu_init_threshold is not None:
        config.jumprelu_init_threshold = args.jumprelu_init_threshold
    if args.l0_coeff is not None:
        config.l0_coeff = args.l0_coeff
    if args.target_l0 is not None:
        config.target_l0 = args.target_l0
    if args.l0_coeff_start is not None:
        config.l0_coeff_start = args.l0_coeff_start
    if args.l0_coeff_lr is not None:
        config.l0_coeff_lr = args.l0_coeff_lr

    # Handle JumpReLU calibration
    if args.calibrate_jumprelu:
        threshold = calibrate_jumprelu_threshold(config)
        config.jumprelu_init_threshold = threshold
        print(f"Calibrated JumpReLU threshold: {threshold}")

    # Generate all runs
    all_runs = generate_grid(config)

    # Auto-generate output dir with timestamp if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/grid_search_{timestamp}")
    else:
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
    print(f"\nModel settings:")
    print(f"  model_name: {config.model_name}")
    print(f"  layer: {config.layer}")
    print(f"  site: {config.site}")
    print(f"\nSAE settings:")
    print(f"  dict_size: {config.dict_size}")
    print(f"  primary_top_k: {config.primary_top_k}")
    print(f"  meta_top_k: {config.meta_top_k}")
    print(f"  bandwidth: {config.bandwidth}")
    if config.jumprelu_init_threshold is not None:
        print(f"  jumprelu_init_threshold: {config.jumprelu_init_threshold}")
    if config.target_l0 is not None:
        print(f"  target_l0: {config.target_l0} (dynamic mode)")
        print(f"  l0_coeff_start: {config.l0_coeff_start}")
        print(f"  l0_coeff_lr: {config.l0_coeff_lr}")
    elif config.l0_coeff is not None:
        print(f"  l0_coeff: {config.l0_coeff} (fixed mode)")
    print(f"\nGrid parameters:")
    print(f"  lambda2: {config.lambda2}")
    print(f"  sigma_sq: {config.sigma_sq}")
    print(f"  meta_dict_size: {config.meta_dict_size}")
    if config.match_architectures:
        print(f"  sae_type: {config.sae_type} (matched primary & meta)")
    else:
        print(f"  primary_sae_type: {config.primary_sae_type}")
        print(f"  meta_sae_type: {config.meta_sae_type}")
    print(f"\nTraining settings:")
    print(f"  num_tokens: {config.num_tokens:,}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  lr: {config.lr}")
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
    print(f"Starting {len(runs_to_do)} jobs with {args.num_workers} workers...\n")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                run_single_job,
                params,
                output_dir,
                "cuda:0",
                i,
                len(runs_to_do),
            ): params
            for i, params in enumerate(runs_to_do)
        }

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
