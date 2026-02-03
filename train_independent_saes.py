"""
Train multiple independent primary SAEs with different random seeds.
Used to establish baseline variance for decoder comparison.
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


def main():
    parser = argparse.ArgumentParser(description="Train independent SAEs with random seeds")
    parser.add_argument("--num_runs", type=int, default=4, help="Number of independent SAE fits")
    parser.add_argument("--output_dir", type=str, default="outputs/independent_saes", help="Output directory")

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

    # SAE type config
    parser.add_argument("--sae_type", type=str, default="jumprelu", choices=["batchtopk", "jumprelu"])
    parser.add_argument("--primary_top_k", type=int, default=64)
    parser.add_argument("--target_l0", type=int, default=64)
    parser.add_argument("--bandwidth", type=float, default=0.001)
    parser.add_argument("--jumprelu_init_threshold", type=float, default=0.001)

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {args.num_runs} independent SAEs")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model_name}, Layer: {args.layer}")
    print(f"SAE type: {args.sae_type}, dict_size: {args.dict_size}")
    print(f"Tokens per run: {args.num_tokens:,}")
    print("="*60)

    # Build base config (same as grid search)
    cfg = get_default_cfg()
    cfg.update({
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
        "sae_type": args.sae_type,
        "top_k": args.primary_top_k,
        "target_l0": args.target_l0,
        "bandwidth": args.bandwidth,
        "jumprelu_init_threshold": args.jumprelu_init_threshold,
        "dataset_path": "HuggingFaceFW/fineweb",
        "dataset_name": "sample-10BT",
    })

    # Load model once (reuse across runs)
    print("Loading model...")
    model = load_model_and_set_sizes(cfg)
    print(f"Model loaded. act_size={cfg['act_size']}")

    # Store results for summary
    all_results = []
    total_start_time = time.time()

    # Track cumulative documents to skip (so each run uses fresh data)
    cumulative_documents = 0

    # Train each independent SAE
    for run_idx in range(args.num_runs):
        # Generate random seed (for model init, not data)
        seed = random.randint(0, 2**32 - 1)

        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{args.num_runs} - Seed: {seed}")
        print(f"Skipping {cumulative_documents:,} documents (previously used)")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Set seeds for model initialization
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Skip documents already used by previous runs
        cfg["skip_documents"] = cumulative_documents

        start_time = time.time()

        # Create fresh activation store (will skip previously used documents)
        print(f"Creating activation store...")
        sys.stdout.flush()
        activation_store = ActivationsStore(model, cfg)

        # Create fresh SAE
        print(f"Creating {cfg['sae_type']} SAE (dict_size={cfg['dict_size']})...")
        sys.stdout.flush()
        sae_cls = get_sae_class(cfg["sae_type"])
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
            "seed": seed,
            "l2": metrics["l2"],
            "l0": metrics["l0"],
            "dead": metrics.get("dead", 0),
            "elapsed_min": elapsed / 60,
            "documents_used": docs_this_run,
            "documents_cumulative": cumulative_documents,
        }
        all_results.append(result)

        print(f"\n[Run {run_idx + 1}/{args.num_runs}] Completed in {elapsed/60:.1f} min")
        print(f"  L2: {metrics['l2']:.6f}")
        print(f"  L0: {metrics['l0']:.1f}")
        print(f"  Dead features: {metrics.get('dead', 'N/A')}")
        print(f"  Documents used: {docs_this_run:,} (cumulative: {cumulative_documents:,})")
        sys.stdout.flush()

        # Save
        run_dir = output_dir / f"run_{run_idx:02d}_seed_{seed}"
        run_dir.mkdir(exist_ok=True)

        save_path = run_dir / "primary_sae.pt"
        torch.save({
            "state_dict": sae.state_dict(),
            "cfg": cfg,
            "seed": seed,
            "run_idx": run_idx,
            "metrics": metrics,
        }, save_path)
        print(f"  Saved to: {save_path}")
        sys.stdout.flush()

        # Cleanup
        del activation_store, sae
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start_time

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.num_runs} Independent SAE Fits")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Total documents used: {cumulative_documents:,}")
    print(f"\n{'Run':<6} {'Seed':<12} {'L2':<12} {'L0':<8} {'Dead':<8} {'Docs':<12} {'Time(min)':<10}")
    print("-" * 70)

    l2_values = []
    l0_values = []
    for r in all_results:
        print(f"{r['run_idx']:<6} {r['seed']:<12} {r['l2']:<12.6f} {r['l0']:<8.1f} {r['dead']:<8} {r['documents_used']:<12,} {r['elapsed_min']:<10.1f}")
        l2_values.append(r["l2"])
        l0_values.append(r["l0"])

    print("-" * 70)
    print(f"{'Mean':<6} {'':<12} {np.mean(l2_values):<12.6f} {np.mean(l0_values):<8.1f}")
    print(f"{'Std':<6} {'':<12} {np.std(l2_values):<12.6f} {np.std(l0_values):<8.1f}")
    print(f"{'Min':<6} {'':<12} {np.min(l2_values):<12.6f} {np.min(l0_values):<8.1f}")
    print(f"{'Max':<6} {'':<12} {np.max(l2_values):<12.6f} {np.max(l0_values):<8.1f}")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")

    # Save summary JSON
    summary = {
        "num_runs": args.num_runs,
        "total_time_min": total_elapsed / 60,
        "config": {
            "model_name": args.model_name,
            "layer": args.layer,
            "dict_size": args.dict_size,
            "sae_type": args.sae_type,
            "num_tokens": args.num_tokens,
        },
        "runs": all_results,
        "stats": {
            "l2_mean": float(np.mean(l2_values)),
            "l2_std": float(np.std(l2_values)),
            "l0_mean": float(np.mean(l0_values)),
            "l0_std": float(np.std(l0_values)),
        }
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
