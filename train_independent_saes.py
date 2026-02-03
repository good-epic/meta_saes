"""
Train multiple independent primary SAEs with different random seeds.
Used to establish baseline variance for decoder comparison.
"""

import argparse
import random
import time
from pathlib import Path

import torch
import numpy as np

from BatchTopK.sae import BatchTopKSAE
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

    # Train each independent SAE
    for run_idx in range(args.num_runs):
        # Generate random seed
        seed = random.randint(0, 2**32 - 1)

        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{args.num_runs} - Seed: {seed}")
        print(f"{'='*60}")

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        start_time = time.time()

        # Create fresh activation store (will use new random sampling)
        activation_store = ActivationsStore(model, cfg)

        # Create fresh SAE
        sae_cls = get_sae_class(cfg["sae_type"])
        sae = sae_cls(cfg)

        # Train
        print(f"Training {cfg['sae_type']} SAE...")
        metrics = train_primary_sae_solo(sae, activation_store, cfg)

        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed/60:.1f} minutes")
        print(f"Final metrics: L2={metrics['l2']:.6f}, L0={metrics['l0']:.1f}")

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
        print(f"Saved to {save_path}")

        # Cleanup
        del activation_store, sae
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"COMPLETED: {args.num_runs} independent SAE fits")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
