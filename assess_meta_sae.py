#%%
"""
Standalone script for assessing trained SAEs (joint and sequential training variants).

This script loads pre-trained SAEs from .pt files and runs comprehensive assessments
including threshold analysis, L0 sparsity evaluation, and reconstruction quality.

Example usage (command line)::
    python assess_meta_sae.py \
        --dataset_path "vietgpt/openwebtext_en" \
        --layer 8 \
        --site resid_pre \
        --dict_size 36864 \
        --meta_dict_size 1536 \
        --primary_top_k 32 \
        --meta_top_k 4 \
        --num_assessment_batches 1000

Example usage (interactive mode in Cursor with #%% blocks)::
    # Interactive mode automatically uses smaller test-friendly defaults
    args = parse_args()  
    
    # Then run the assessment pipeline
    cfg, meta_cfg = build_configs(args)
    # ... continue with assessment
"""

from __future__ import annotations

import argparse
import json
import pickle
from typing import Dict, Any
import numpy as np

import torch
import gc
import sys

from BatchTopK.sae import BatchTopKSAE  # primary SAE class
from BatchTopK.config import get_default_cfg  # default configuration generator
from BatchTopK.activation_store import ActivationsStore  # provides model activations
from meta_sae_extension import MetaSAEWrapper, BatchTopKSAEWithPenalty

from assessment import assess_sae_thresholds, assess_l0_with_thresholds, ReconstructionAssessor
from similarity_analysis import FunctionalSimilarityAssessor
from utils import build_common_arg_parser, parse_args_common, load_model_and_set_sizes, print_gpu_memory_usage, print_configs

#%%
##### Preliminary Functions #####
#################################

def build_configs(args: argparse.Namespace) -> (Dict[str, Any], Dict[str, Any]):
    """Create configuration dictionaries for the primary SAE and meta SAE."""
    # Start from repo defaults, then copy ALL args into cfg
    cfg = get_default_cfg()
    cfg.update(vars(args))
    # Explicitly set top_k for primary
    cfg["top_k"] = args.primary_top_k

    # Meta configuration (copy everything, then override size/sparsity)
    meta_cfg = cfg.copy()
    meta_cfg["dict_size"] = args.meta_dict_size
    meta_cfg["top_k"] = args.meta_top_k
    
    return cfg, meta_cfg


def load_sae_from_checkpoint(checkpoint_path: str, sae_class, cfg: Dict[str, Any], meta_sae=None):
    """Load a trained SAE from a checkpoint file."""
    print(f"📂 Loading SAE from {checkpoint_path}...")
    
    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_path)
    
    # Create SAE instance
    if sae_class == MetaSAEWrapper:
        sae = sae_class(BatchTopKSAE, cfg)
    elif sae_class == BatchTopKSAEWithPenalty:
        # Expect penalty_cfg inside checkpoint and a provided meta_sae instance
        if 'penalty_cfg' not in checkpoint:
            raise ValueError("Checkpoint missing 'penalty_cfg' needed for BatchTopKSAEWithPenalty")
        if meta_sae is None:
            raise ValueError("meta_sae must be provided when loading BatchTopKSAEWithPenalty")
        penalty_cfg = checkpoint['penalty_cfg']
        sae = sae_class(cfg, meta_sae, penalty_cfg)
    else:
        sae = sae_class(cfg)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['state_dict'])
    else:
        sae.load_state_dict(checkpoint)
    
    sae.eval()  # Set to evaluation mode
    print(f"   ✅ Loaded SAE with {sum(p.numel() for p in sae.parameters()):,} parameters")
    
    return sae


#%%
###### Check CUDA availability and GPU info ######
##################################################
if torch.cuda.is_available():
    print(f"🚀 CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ CUDA not available. Assessment will use CPU (much slower).")

# Parse arguments and build configs
parser = build_common_arg_parser("Assess trained SAEs (joint and sequential variants).")
args = parse_args_common(parser)

# Set assessment flags for interactive mode
if len([arg for arg in sys.argv if not arg.startswith('--f=')]) == 1 or any('ipykernel' in arg for arg in sys.argv):
    args.assess_joint_saes = True
    args.assess_sequential_saes = True

cfg, meta_cfg = build_configs(args)

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

print_configs(cfg, meta_cfg)

#%%
###### Load Trained SAEs ######
###############################

print("🔍 Starting comprehensive SAE assessment pipeline...")
print("Will assess the following SAE variants:")
if args.assess_joint_saes:
    print("1. Joint training: Primary + Meta SAE (trained together)")
if args.assess_sequential_saes:
    print("2. Sequential training: Solo primary + Meta SAE (trained separately)")
print("="*60)

# Load trained SAEs
joint_primary_sae = None
joint_meta_sae = None
solo_primary_sae = None
sequential_meta_sae = None

if args.assess_joint_saes:
    print("\n📂 Loading joint training models...")
    try:
        # Load meta first, then primary-with-penalty using the loaded meta
        joint_meta_sae = load_sae_from_checkpoint(args.joint_meta_path, MetaSAEWrapper, meta_cfg)
        joint_primary_sae = load_sae_from_checkpoint(args.joint_primary_path, BatchTopKSAEWithPenalty, cfg, meta_sae=joint_meta_sae)
        print("   ✅ Joint models loaded successfully")
    except FileNotFoundError as e:
        print(f"   ⚠️ Could not load joint models: {e}")
        print("   Skipping joint model assessment")
        args.assess_joint_saes = False

if args.assess_sequential_saes:
    print("\n📂 Loading sequential training models...")
    try:
        solo_primary_sae = load_sae_from_checkpoint(args.solo_primary_path, BatchTopKSAE, cfg)
        sequential_meta_sae = load_sae_from_checkpoint(args.sequential_meta_path, MetaSAEWrapper, meta_cfg)
        print("   ✅ Sequential models loaded successfully")
    except FileNotFoundError as e:
        print(f"   ⚠️ Could not load sequential models: {e}")
        print("   Skipping sequential model assessment")
        args.assess_sequential_saes = False

if not args.assess_joint_saes and not args.assess_sequential_saes:
    print("⚠️ No SAEs could be loaded. Please ensure the .pt files exist in the current directory.")
    print("Expected files:")
    print("  - joint_primary_sae.pt")
    print("  - joint_meta_sae.pt") 
    print("  - solo_primary_sae.pt")
    print("  - sequential_meta_sae.pt")
    exit(1)

print_gpu_memory_usage()


#%%
################### Find L0 Thresholds ####################
###########################################################

print("\n" + "="*60)
print("RUNNING COMPREHENSIVE POST-TRAINING ASSESSMENTS")
print("="*60)

# Determine which assessments to run
assessments_to_run = []
if args.assess_joint_saes:
    assessments_to_run.append("Joint training models (primary + meta)")
if args.assess_sequential_saes:
    assessments_to_run.extend(["Solo primary SAE", "Sequential meta SAE"])

if not assessments_to_run:
    print("⚠️ No models were loaded, skipping assessments")
else:
    print("Will assess loaded model variants:")
    for i, assessment in enumerate(assessments_to_run, 1):
        print(f"{i}. {assessment}")

# Create fresh config for assessment
assessment_cfg = cfg.copy()
assessment_cfg["num_batches_in_buffer"] = args.num_batches_in_buffer
assessment_cfg["batch_size"] = min(cfg["batch_size"], 512)  # Smaller batch size
assessment_cfg["model_batch_size"] = min(cfg["model_batch_size"], 256)  # Reduce model batch size too

# Initialize results storage
assessment_results = {}

if args.assess_joint_saes:
    # ===== ASSESS JOINT TRAINING MODELS =====
    print(f"\n📊 ASSESSMENT 1: JOINT TRAINING MODELS")
    print("="*50)
    assessment_store = ActivationsStore(model, assessment_cfg)
    
    joint_threshold_results = assess_sae_thresholds(
        primary_sae=joint_primary_sae,
        meta_sae=joint_meta_sae, 
        activation_store=assessment_store,
        cfg=cfg,
        meta_cfg=meta_cfg,
        num_assessment_batches=args.num_assessment_batches
    )
    assessment_results['joint'] = joint_threshold_results

if args.assess_sequential_saes:
    # ===== ASSESS SEQUENTIAL TRAINING MODELS =====
    print(f"\n📊 ASSESSMENT 2: SEQUENTIAL TRAINING MODELS")
    print("="*50)
    assessment_store_seq = ActivationsStore(model, assessment_cfg)
    
    sequential_threshold_results = assess_sae_thresholds(
        primary_sae=solo_primary_sae,  # Same frozen primary as used for meta training
        meta_sae=sequential_meta_sae,
        activation_store=assessment_store_seq,
        cfg=cfg,
        meta_cfg=meta_cfg,
        num_assessment_batches=args.num_assessment_batches
    )
    assessment_results['sequential'] = sequential_threshold_results

# Save all assessment results
print("\n💾 Saving all assessment results...")
for key, results in assessment_results.items():
    with open(f"{key}_threshold_assessment_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"   ✅ Saved: {key}_threshold_assessment_results.pkl")

#%%
######### Find L0 Counts with Learned Thresholds #########
##########################################################

print("\n" + "="*60)
print("RUNNING COMPREHENSIVE L0 ASSESSMENTS WITH LEARNED THRESHOLDS")
print("="*60)

if not assessment_results:
    print("⚠️ No threshold assessments were run, skipping L0 assessments")
else:
    print("Will run L0 assessments for model variants using their respective thresholds")
    
    # Initialize L0 results storage
    l0_results = {}
    
    if 'joint' in assessment_results:
        # ===== L0 ASSESSMENT: JOINT TRAINING MODELS =====
        print(f"\n📊 L0 ASSESSMENT 1: JOINT TRAINING MODELS")
        print("="*50)
        joint_l0_results = assess_l0_with_thresholds(
            primary_sae=joint_primary_sae,
            meta_sae=joint_meta_sae,
            activation_store=assessment_store,  # Continue from threshold assessment
            cfg=cfg,
            meta_cfg=meta_cfg,
            threshold_stats=assessment_results['joint'],
            num_assessment_batches=args.num_assessment_batches
        )
        l0_results['joint'] = joint_l0_results
    
    if 'sequential' in assessment_results:
        # ===== L0 ASSESSMENT: SEQUENTIAL TRAINING MODELS =====
        print(f"\n📊 L0 ASSESSMENT 2: SEQUENTIAL TRAINING MODELS")
        print("="*50)
        sequential_l0_results = assess_l0_with_thresholds(
            primary_sae=solo_primary_sae,  # Same frozen primary
            meta_sae=sequential_meta_sae,
            activation_store=assessment_store_seq,  # Continue from threshold assessment
            cfg=cfg,
            meta_cfg=meta_cfg,
            threshold_stats=assessment_results['sequential'],
            num_assessment_batches=args.num_assessment_batches
        )
        l0_results['sequential'] = sequential_l0_results
    
    # Save all L0 assessment results
    print("\n💾 Saving all L0 assessment results...")
    for key, results in l0_results.items():
        with open(f"{key}_l0_assessment_results.pkl", "wb") as f:
            pickle.dump(results, f)
        print(f"   ✅ Saved: {key}_l0_assessment_results.pkl")

print("\n🎉 COMPREHENSIVE ASSESSMENT PIPELINE COMPLETED!")
print("="*60)

#%%
################### Reconstruction Quality Assessment ###############
#####################################################################

print("\n" + "="*60)
print("RUNNING RECONSTRUCTION QUALITY ASSESSMENTS")
print("="*60)

if args.assess_joint_saes:
    print(f"\n🔍 RECONSTRUCTION ASSESSMENT: JOINT TRAINING MODELS")
    print("="*50)
    
    # Create fresh activation store for reconstruction assessment
    recon_cfg = assessment_cfg.copy()
    recon_cfg["num_batches_in_buffer"] = 2  # Smaller buffer for reconstruction assessment
    recon_store = ActivationsStore(model, recon_cfg)
    
    joint_assessor = ReconstructionAssessor(joint_primary_sae, model, recon_store)
    joint_recon_results = joint_assessor.run_all_assessments()

if args.assess_sequential_saes:
    print(f"\n🔍 RECONSTRUCTION ASSESSMENT: SEQUENTIAL TRAINING MODELS")
    print("="*50)
    
    # Create fresh activation store for reconstruction assessment
    recon_cfg = assessment_cfg.copy()
    recon_cfg["num_batches_in_buffer"] = 2  # Smaller buffer for reconstruction assessment
    recon_store_seq = ActivationsStore(model, recon_cfg)
    
    solo_assessor = ReconstructionAssessor(solo_primary_sae, model, recon_store_seq)
    solo_recon_results = solo_assessor.run_all_assessments()

#%%
################### Functional Similarity Analysis ##################
#####################################################################

if args.assess_joint_saes and args.assess_sequential_saes:
    print("\n" + "="*60)
    print("RUNNING FUNCTIONAL SIMILARITY ANALYSIS")
    print("="*60)
    
    print("🔍 Comparing joint vs solo primary SAE feature functionality...")
    
    # Build a small, self-contained config for this section (do not rely on assessment_cfg)
    similarity_cfg = cfg.copy()
    similarity_cfg["num_batches_in_buffer"] = 1  # Minimal buffer for similarity analysis
    similarity_cfg["batch_size"] = min(cfg.get("batch_size", 1024), 256)
    similarity_cfg["model_batch_size"] = min(cfg.get("model_batch_size", 256), 8)
    
    # No need to instantiate an ActivationsStore here; the assessor streams from HF directly
    functional_assessor = FunctionalSimilarityAssessor(
        solo_primary_sae, 
        joint_primary_sae, 
        model, 
        similarity_cfg
    )
    
    similarity_results = functional_assessor.run_analysis()
    
    print(f"\n📊 Functional Similarity Results:")
    print(f"   Mean correlation: {similarity_results['correlation'].mean():.3f}")
    print(f"   Median correlation: {similarity_results['correlation'].median():.3f}")
    print(f"   Std correlation: {similarity_results['correlation'].std():.3f}")
    print(f"   Min correlation: {similarity_results['correlation'].min():.3f}")
    print(f"   Max correlation: {similarity_results['correlation'].max():.3f}")
    
    # Save similarity results
    with open("functional_similarity_results.pkl", "wb") as f:
        pickle.dump(similarity_results, f)
    print("   ✅ Saved: functional_similarity_results.pkl")


#%%
################### Final Cleanup and Summary #######################
#####################################################################

print("\n" + "="*60)
print("🎉 ALL ASSESSMENTS COMPLETED!")
print("="*60)

# Clean up memory
print("🧹 Cleaning up memory...")
try:
    del assessment_store
except NameError:
    pass
try:
    del assessment_store_seq
except NameError:
    pass
if args.assess_joint_saes:
    del joint_primary_sae, joint_meta_sae
if args.assess_sequential_saes:
    del solo_primary_sae, sequential_meta_sae

gc.collect()
torch.cuda.empty_cache()

print_gpu_memory_usage()

print("\n📁 Assessment files created:")
if args.assess_joint_saes:
    print("  • joint_threshold_assessment_results.pkl")
    print("  • joint_l0_assessment_results.pkl")
if args.assess_sequential_saes:
    print("  • sequential_threshold_assessment_results.pkl")
    print("  • sequential_l0_assessment_results.pkl")
if args.assess_joint_saes and args.assess_sequential_saes:
    print("  • functional_similarity_results.pkl")

print("\n🎯 Assessment Summary:")
if args.assess_joint_saes:
    print("  ✅ Joint training models assessed")
if args.assess_sequential_saes:
    print("  ✅ Sequential training models assessed")

print("="*60)

#%%


#%%