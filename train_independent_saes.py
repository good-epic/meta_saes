"""
Train multiple independent primary SAEs with different random seeds.
Used to establish baseline variance for decoder comparison.
Supports training both JumpReLU and BatchTopK SAEs.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import numpy as np

from BatchTopK.config import get_default_cfg
from BatchTopK.activation_store import ActivationsStore
from meta_sae_extension import get_sae_class, train_primary_sae_solo
from utils import load_model_and_set_sizes


def train_sae_runs(model, base_cfg, sae_type, num_runs, output_dir, cumulative_documents, args):
    """Train multiple independent SAEs of a given type."""

    results = []
    type_dir = output_dir / sae_type
    type_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# Training {num_runs} independent {sae_type.upper()} SAEs")
    print(f"{'#'*60}")

    for run_idx in range(num_runs):
        # Generate random seed (for model init, not data)
        seed = random.randint(0, 2**32 - 1)

        print(f"\n{'='*60}")
        print(f"[{sae_type.upper()}] RUN {run_idx + 1}/{num_runs} - Seed: {seed}")
        print(f"Skipping {cumulative_documents:,} documents (previously used)")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Set seeds for model initialization
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create config for this run
        cfg = base_cfg.copy()
        cfg["sae_type"] = sae_type
        cfg["skip_documents"] = cumulative_documents

        # SAE-type specific settings
        if sae_type == "batchtopk":
            cfg["top_k"] = args.primary_top_k
            # BatchTopK doesn't need L0 coefficient control
            cfg["target_l0"] = None
        else:  # jumprelu
            cfg["target_l0"] = args.target_l0
            cfg["bandwidth"] = args.bandwidth
            cfg["jumprelu_init_threshold"] = args.jumprelu_init_threshold
            cfg["l0_coeff_start"] = args.l0_coeff_start
            cfg["l0_stability_threshold"] = args.l0_stability_threshold
            cfg["l0_stability_window"] = args.l0_stability_window
            cfg["l0_adjustment_factor"] = args.l0_adjustment_factor

        start_time = time.time()

        # Create fresh activation store (will skip previously used documents)
        print(f"Creating activation store...")
        sys.stdout.flush()
        activation_store = ActivationsStore(model, cfg)

        # Create fresh SAE
        print(f"Creating {sae_type} SAE (dict_size={cfg['dict_size']})...")
        sys.stdout.flush()
        sae_cls = get_sae_class(sae_type)
        sae = sae_cls(cfg)

        # Train (train_primary_sae_solo has its own tqdm progress bar)
        print(f"Starting training...")
        sys.stdout.flush()
        metrics = train_primary_sae_solo(sae, activation_store, cfg)

        elapsed = time.time() - start_time

        # Track documents used in this run
        docs_this_run = activation_store.documents_processed - cumulative_documents
        cumulative_documents = activation_store.documents_processed

        # Store result
        result = {
            "run_idx": run_idx,
            "sae_type": sae_type,
            "seed": seed,
            "l2": metrics["l2"],
            "l0": metrics["l0"],
            "dead": metrics.get("dead", 0),
            "elapsed_min": elapsed / 60,
            "documents_used": docs_this_run,
            "documents_cumulative": cumulative_documents,
        }
        results.append(result)

        print(f"\n[{sae_type.upper()} Run {run_idx + 1}/{num_runs}] Completed in {elapsed/60:.1f} min")
        print(f"  L2: {metrics['l2']:.6f}")
        print(f"  L0: {metrics['l0']:.1f}")
        print(f"  Dead features: {metrics.get('dead', 'N/A')}")
        print(f"  Documents used: {docs_this_run:,} (cumulative: {cumulative_documents:,})")
        sys.stdout.flush()

        # Save
        run_dir = type_dir / f"run_{run_idx:02d}_seed_{seed}"
        run_dir.mkdir(exist_ok=True)

        save_path = run_dir / "primary_sae.pt"
        torch.save({
            "state_dict": sae.state_dict(),
            "cfg": cfg,
            "seed": seed,
            "run_idx": run_idx,
            "sae_type": sae_type,
            "metrics": metrics,
        }, save_path)
        print(f"  Saved to: {save_path}")
        sys.stdout.flush()

        # Cleanup
        del activation_store, sae
        torch.cuda.empty_cache()

    return results, cumulative_documents


def print_type_summary(results, sae_type):
    """Print summary statistics for one SAE type."""
    if not results:
        return

    print(f"\n{'-'*60}")
    print(f"{sae_type.upper()} Summary ({len(results)} runs)")
    print(f"{'-'*60}")
    print(f"{'Run':<6} {'Seed':<12} {'L2':<12} {'L0':<8} {'Dead':<8} {'Docs':<12} {'Time(min)':<10}")
    print("-" * 70)

    l2_values = []
    l0_values = []
    for r in results:
        print(f"{r['run_idx']:<6} {r['seed']:<12} {r['l2']:<12.6f} {r['l0']:<8.1f} {r['dead']:<8} {r['documents_used']:<12,} {r['elapsed_min']:<10.1f}")
        l2_values.append(r["l2"])
        l0_values.append(r["l0"])

    print("-" * 70)
    print(f"{'Mean':<6} {'':<12} {np.mean(l2_values):<12.6f} {np.mean(l0_values):<8.1f}")
    print(f"{'Std':<6} {'':<12} {np.std(l2_values):<12.6f} {np.std(l0_values):<8.1f}")
    print(f"{'Min':<6} {'':<12} {np.min(l2_values):<12.6f} {np.min(l0_values):<8.1f}")
    print(f"{'Max':<6} {'':<12} {np.max(l2_values):<12.6f} {np.max(l0_values):<8.1f}")

    return {
        "l2_mean": float(np.mean(l2_values)),
        "l2_std": float(np.std(l2_values)),
        "l0_mean": float(np.mean(l0_values)),
        "l0_std": float(np.std(l0_values)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train independent SAEs with random seeds")
    parser.add_argument("--num_runs", type=int, default=4, help="Number of independent SAE fits per type")
    parser.add_argument("--output_dir", type=str, default="outputs/independent_saes", help="Output directory")

    # SAE types to train
    parser.add_argument("--sae_types", type=str, nargs="+", default=["jumprelu", "batchtopk"],
                        choices=["batchtopk", "jumprelu"],
                        help="SAE types to train (default: both jumprelu and batchtopk)")

    # Match grid search config for GPT-2 Large
    parser.add_argument("--model_name", type=str, default="gpt2-large")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--site", type=str, default="resid_pre")
    parser.add_argument("--dict_size", type=int, default=20480)
    parser.add_argument("--num_tokens", type=int, default=100_000_000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--model_batch_size", type=int, default=256)
    parser.add_argument("--num_batches_in_buffer", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")

    # BatchTopK config
    parser.add_argument("--primary_top_k", type=int, default=64, help="Top-K for BatchTopK SAE (also target L0)")

    # JumpReLU config
    parser.add_argument("--target_l0", type=int, default=64, help="Target L0 for JumpReLU")
    parser.add_argument("--bandwidth", type=float, default=0.001)
    parser.add_argument("--jumprelu_init_threshold", type=float, default=0.001)

    # L0 coefficient control (wait-for-stability approach) - JumpReLU only
    parser.add_argument("--l0_coeff_start", type=float, default=1e-5)
    parser.add_argument("--l0_stability_threshold", type=float, default=0.02, help="Relative change below this = stable")
    parser.add_argument("--l0_stability_window", type=int, default=500, help="Steps to check stability over")
    parser.add_argument("--l0_adjustment_factor", type=float, default=0.1, help="How much to adjust when stable")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {args.num_runs} independent SAEs for each type: {args.sae_types}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model_name}, Layer: {args.layer}")
    print(f"Dict size: {args.dict_size}, Tokens per run: {args.num_tokens:,}")
    print("="*60)

    # Build base config (shared across SAE types)
    base_cfg = get_default_cfg()
    base_cfg.update({
        "model_name": args.model_name,
        "layer": args.layer,
        "site": args.site,
        "dict_size": args.dict_size,
        "num_tokens": args.num_tokens,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "model_batch_size": args.model_batch_size,
        "num_batches_in_buffer": args.num_batches_in_buffer,
        "device": args.device,
        "dataset_path": "HuggingFaceFW/fineweb",
        "dataset_name": "sample-10BT",
    })

    # Load model once (reuse across all runs)
    print("Loading model...")
    model = load_model_and_set_sizes(base_cfg)
    print(f"Model loaded. act_size={base_cfg['act_size']}")

    # Track cumulative documents to skip (so each run uses fresh data)
    cumulative_documents = 0
    total_start_time = time.time()

    # Store results per SAE type
    all_results = {}
    all_stats = {}

    # Train each SAE type
    for sae_type in args.sae_types:
        results, cumulative_documents = train_sae_runs(
            model, base_cfg, sae_type, args.num_runs,
            output_dir, cumulative_documents, args
        )
        all_results[sae_type] = results

    total_elapsed = time.time() - total_start_time

    # Print summaries
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Total documents used: {cumulative_documents:,}")
    print(f"SAE types trained: {args.sae_types}")
    print(f"Runs per type: {args.num_runs}")

    for sae_type in args.sae_types:
        stats = print_type_summary(all_results[sae_type], sae_type)
        if stats:
            all_stats[sae_type] = stats

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    # Save summary JSON
    summary = {
        "num_runs_per_type": args.num_runs,
        "sae_types": args.sae_types,
        "total_time_min": total_elapsed / 60,
        "total_documents": cumulative_documents,
        "config": {
            "model_name": args.model_name,
            "layer": args.layer,
            "dict_size": args.dict_size,
            "num_tokens": args.num_tokens,
            "primary_top_k": args.primary_top_k,
            "target_l0": args.target_l0,
        },
        "results_by_type": all_results,
        "stats_by_type": all_stats,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
