#%%
"""
Command‑line script for training a primary BatchTopK SAE with a
meta‑SAE using Hugging Face streaming datasets.

This script reads a configuration, loads a transformer model via
``transformer_lens``, constructs an activation store, instantiates a
primary SAE with a decomposability penalty and a meta SAE, and
trains them alternately using the logic defined in
``meta_sae_extension.train_sae_with_meta``.

Example usage (command line)::

    python train_meta_sae.py \
        --dataset_path "wikitext" \
        --layer 8 \
        --site resid_pre \
        --dict_size 12288 \
        --meta_dict_size 4096 \
        --primary_top_k 32 \
        --meta_top_k 4 \
        --lambda2 1e-3 \
        --sigma_sq 0.1 \
        --n_primary_steps 100 \
        --n_meta_steps 10

Example usage (interactive mode in Cursor with #%% blocks)::

    # Interactive mode automatically uses smaller test-friendly defaults
    args = parse_args()  
    
    # Then run the training pipeline
    cfg, meta_cfg, penalty_cfg = build_configs(args)
    # ... continue with training

Note that this script assumes you have installed ``transformer_lens``
and ``datasets`` with streaming enabled (``pip install transformer_lens datasets``).
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Any
import numpy as np

import torch
import gc # Added for detailed memory debugging
import sys

from BatchTopK.sae import BatchTopKSAE  # primary SAE class
from BatchTopK.config import get_default_cfg  # default configuration generator
from BatchTopK.activation_store import ActivationsStore  # provides model activations
import random
from meta_sae_extension import (
    MetaSAEWrapper,
    BatchTopKSAEWithPenalty,
    train_sae_with_meta,
    train_primary_sae_solo,
    train_meta_sae_on_frozen_primary,
)
import importlib
assessment = importlib.reload(__import__("assessment"))
assess_sae_thresholds = assessment.assess_sae_thresholds
assess_l0_with_thresholds = assessment.assess_l0_with_thresholds

# Import shared utilities
from utils import build_common_arg_parser, parse_args_common, load_model_and_set_sizes, print_gpu_memory_usage, print_configs

#%%
def parse_args() -> argparse.Namespace:
    parser = build_common_arg_parser("Train a primary SAE with a meta SAE penalty.")
    
    # Use shared argument parsing logic
    args = parse_args_common(parser)
    
    # Set training flags for interactive mode
    if len([arg for arg in sys.argv if not arg.startswith('--f=')]) == 1 or any('ipykernel' in arg for arg in sys.argv):
        args.train_joint_saes = True
        args.train_sequential_saes = True
    
    return args


def build_configs(args: argparse.Namespace) -> (Dict[str, Any], Dict[str, Any], Dict[str, Any]):
    """Create configuration dictionaries for the primary SAE, meta SAE, and penalty."""
    # Start from repo defaults, then copy ALL args into cfg
    cfg = get_default_cfg()
    cfg.update(vars(args))
    # Explicitly set top_k for primary
    cfg["top_k"] = args.primary_top_k
    # Ensure buffer for initial phase uses joint setting by default; phases will override as needed
    cfg["num_batches_in_buffer"] = args.num_batches_in_buffer_joint

    # Meta configuration (copy everything, then override size/sparsity)
    meta_cfg = cfg.copy()
    meta_cfg["dict_size"] = args.meta_dict_size
    meta_cfg["top_k"] = args.meta_top_k

    # Penalty configuration from args
    penalty_cfg = {
        "lambda2": args.lambda2,
        "sigma_sq": args.sigma_sq,
        "n_primary_steps": args.n_primary_steps,
        "n_meta_steps": args.n_meta_steps,
        "start_with_primary": True,
    }
    return cfg, meta_cfg, penalty_cfg


#%%
###### Check CUDA availability and GPU info ######
##################################################
if torch.cuda.is_available():
    print(f"🚀 CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ CUDA not available. Training will use CPU (much slower).")

# For interactive mode, parse_args() automatically uses test-friendly defaults
# For command line, pass arguments normally:
# python train_meta_sae.py --dict_size=4096 --batch_size=2048 etc.

args = parse_args()
cfg, meta_cfg, penalty_cfg = build_configs(args)
# Load model and update act_size in configs
model = load_model_and_set_sizes(cfg)
print_gpu_memory_usage()

#%%
###### Ensure meta_cfg uses the updated act_size ######
#######################################################
meta_cfg["act_size"] = cfg["act_size"]
meta_cfg["model_name"] = cfg["model_name"]
meta_cfg["hook_point"] = cfg["hook_point"]
meta_cfg["layer"] = cfg["layer"]
meta_cfg["site"] = cfg["site"]

print_configs(cfg, meta_cfg, penalty_cfg)

#%%

print("🚀 Starting comprehensive training pipeline...")
print("Will train 3 SAE variants for comparison:")
print("1. Joint training: Primary + Meta SAE together")
print("2. Solo training: Primary SAE without meta penalty") 
print("3. Sequential training: Meta SAE on frozen solo primary")
print("="*60)

if args.train_joint_saes:
    # ===== 1. JOINT TRAINING =====
    # Configure for joint training
    cfg["num_batches_in_buffer"] = args.num_batches_in_buffer_joint
    
    # Create activation store for primary SAE
    activation_store = ActivationsStore(model, cfg)
    print(f"🔍 Joint training model_batch_size: {cfg['model_batch_size']}")
    # Instantiate meta SAE and primary SAE with penalty
    meta_sae = MetaSAEWrapper(BatchTopKSAE, meta_cfg)
    primary_sae = BatchTopKSAEWithPenalty(cfg, meta_sae, penalty_cfg)
    print_gpu_memory_usage()

    print("\n" + "="*60)
    print("PHASE 1: JOINT TRAINING (Primary + Meta SAE together)")
    print("="*60)
    if torch.cuda.is_available():
        print(f"🔍 Memory at start of joint training - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")
    train_sae_with_meta(primary_sae, meta_sae, activation_store, model, cfg, meta_cfg, penalty_cfg)

    # Memory cleanup after joint training
    print("\n🧹 Cleaning up memory after joint training...")
    if torch.cuda.is_available():
        print(f"   Memory before cleanup - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")

    # Clear the training activation store
    del activation_store

    # Clear gradients from joint training models
    for param in primary_sae.parameters():
        if param.grad is not None:
            param.grad = None
    for param in meta_sae.parameters():
        if param.grad is not None:
            param.grad = None

    # Clear model cache
    if hasattr(model, 'reset_hooks'):
        model.reset_hooks()
    if hasattr(model, 'cache'):
        model.cache = {}

    # Force garbage collection and CUDA cache cleanup
    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print(f"   Memory after cleanup - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")

    print("💾 Saving jointly trained models...")
    torch.save({'state_dict': primary_sae.state_dict(),
                'cfg': cfg,
                'meta_cfg': meta_cfg,
                'penalty_cfg': penalty_cfg},
               args.joint_primary_path)
    torch.save({'state_dict': meta_sae.state_dict(),
                'meta_cfg': meta_cfg},
               args.joint_meta_path) 



if args.train_sequential_saes:
    # ===== 2. SOLO PRIMARY TRAINING =====
    print("\n" + "="*60)
    print("PHASE 2: SOLO PRIMARY SAE TRAINING (No meta penalty)")
    print("="*60)

    # Configure for sequential training (lower memory usage)
    cfg["num_batches_in_buffer"] = args.num_batches_in_buffer_sequential

    # Create fresh activation store for solo training
    print("Creating fresh activation store for solo training...")
    solo_activation_store = ActivationsStore(model, cfg)
    print(f"🔍 Solo training model_batch_size: {cfg['model_batch_size']}")

    # Create solo primary SAE (regular BatchTopKSAE without penalty)
    print("Creating solo primary SAE...")
    solo_primary_sae = BatchTopKSAE(cfg)
    if torch.cuda.is_available():
        print(f"�� Memory at start of solo training - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")
    train_primary_sae_solo(solo_primary_sae, solo_activation_store, cfg)

    # Memory cleanup after solo training
    print("\n🧹 Cleaning up memory after solo training...")
    if torch.cuda.is_available():
        print(f"   Memory before cleanup - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")

    # Clear the solo activation store
    del solo_activation_store

    # Clear gradients from solo primary SAE
    for param in solo_primary_sae.parameters():
        if param.grad is not None:
            param.grad = None

    # Clear model cache again
    if hasattr(model, 'reset_hooks'):
        model.reset_hooks()
    if hasattr(model, 'cache'):
        model.cache = {}

    # Force garbage collection and CUDA cache cleanup
    gc.collect()
    torch.cuda.empty_cache()

    print("💾 Saving solo primary SAE...")
    torch.save({'state_dict': solo_primary_sae.state_dict(), 'cfg': cfg},
               args.solo_primary_path)

    if torch.cuda.is_available():
        print(f"   Memory after solo primary training after cleanup - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")


    # ===== 3. SEQUENTIAL META TRAINING =====
    print("\n" + "="*60)
    print("PHASE 3: SEQUENTIAL META SAE TRAINING (On frozen solo primary)")
    print("="*60)

    # Create meta SAE for sequential training 
    print("Creating meta SAE for sequential training...")
    sequential_meta_sae = MetaSAEWrapper(BatchTopKSAE, meta_cfg)
    if torch.cuda.is_available():
        print(f"🔍 Memory at start of sequential training - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")
    train_meta_sae_on_frozen_primary(sequential_meta_sae, solo_primary_sae, meta_cfg, penalty_cfg)

    # Memory cleanup after sequential training
    print("\n🧹 Cleaning up memory after sequential training...")
    if torch.cuda.is_available():
        print(f"   Memory before cleanup - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")

    # Clear gradients from sequential meta SAE
    for param in sequential_meta_sae.parameters():
        if param.grad is not None:
            param.grad = None

    # Clear model cache again
    if hasattr(model, 'reset_hooks'):
        model.reset_hooks()
    if hasattr(model, 'cache'):
        model.cache = {}

    # Force garbage collection and CUDA cache cleanup
    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print(f"   Memory after cleanup - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")

    print("💾 Saving sequential meta SAE...")
    torch.save({'state_dict': sequential_meta_sae.state_dict(),
                'meta_cfg': meta_cfg,
                'penalty_cfg': penalty_cfg},
               args.sequential_meta_path)

print("\n" + "="*60)
print("🎉 ALL TRAINING PHASES COMPLETED!")
print("="*60)
print("Models trained:")
if args.train_joint_saes:
    print("  • Joint primary SAE (with meta penalty)")
    print("  • Joint meta SAE") 
if args.train_sequential_saes:
    print("  • Solo primary SAE (no meta penalty)")
    print("  • Sequential meta SAE (trained on frozen solo primary)")
if not args.train_joint_saes and not args.train_sequential_saes:
    print("  • No models were trained in this run")
print("="*60)

#%% 

print("\n🧹 Saving trained models and cleaning up memory...")
if torch.cuda.is_available():
    print(f"   Memory before cleanup - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")

# Save models that were actually trained
print("💾 Saving trained models...")
models_saved = []
if args.train_joint_saes:
    # Joint models were already saved after joint training
    models_saved.extend(["joint_primary_sae.pt", "joint_meta_sae.pt"])
if args.train_sequential_saes:
    # Sequential models were already saved after sequential training  
    models_saved.extend(["solo_primary_sae.pt", "sequential_meta_sae.pt"])

if models_saved:
    print(f"   ✅ Saved: {', '.join(models_saved)}")
else:
    print("   ⚠️ No models were trained in this run")

# Clear model cache
print("   Clearing model cache and gradients...")
if hasattr(model, 'reset_hooks'):
    model.reset_hooks()
if hasattr(model, 'cache'):
    model.cache = {}

# Clear gradients from SAEs that exist
sae_models_to_clean = []
if args.train_joint_saes:
    sae_models_to_clean.extend([primary_sae, meta_sae])
if args.train_sequential_saes:
    sae_models_to_clean.extend([solo_primary_sae, sequential_meta_sae])

for sae_model in sae_models_to_clean:
    for param in sae_model.parameters():
        if param.grad is not None:
            param.grad = None

# Clear any remaining tensors and force garbage collection
gc.collect()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    print(f"   Memory after cleanup - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")

print("All models saved and memory cleaned up.")

