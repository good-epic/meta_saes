"""
Utility functions shared between training and assessment scripts.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Any

import torch


def build_common_arg_parser(description: str = "Common Arg Parser") -> argparse.ArgumentParser:
    """
    Builds a superset argument parser for both training and assessment scripts.
    It's okay if some arguments are unused in a given script.
    """
    parser = argparse.ArgumentParser(description=description)

    # Data & model hooks
    parser.add_argument("--dataset_path", type=str, required=False,
                        default="HuggingFaceFW/fineweb",
                        help="HuggingFace dataset identifier (supports streaming)")
    parser.add_argument("--dataset_name", type=str, required=False,
                        default="sample-10BT",
                        help="Dataset subset/config name (e.g., 'sample-10BT' for FineWeb)")
    parser.add_argument("--layer", type=int, default=8, help="Transformer layer to hook (0‑indexed)")
    parser.add_argument("--site", type=str, default="resid_pre", help="Hook site (resid_pre, resid_post, mlp_out, attn_out)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    # Model/dataset related optional knobs (defaults often taken from get_default_cfg)
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length for tokenization/ctx")
    parser.add_argument("--model_batch_size", type=int, default=256, help="Batch size per model forward for activation generation/assessment")

    # SAE sizes & sparsity
    parser.add_argument("--dict_size", type=int, default=36864, help="Primary SAE dictionary size")
    parser.add_argument("--meta_dict_size", type=int, default=1536, help="Meta SAE dictionary size")
    parser.add_argument("--primary_top_k", type=int, default=32, help="Top-K for primary BatchTopK SAE")
    parser.add_argument("--meta_top_k", type=int, default=4, help="Top-K for meta BatchTopK SAE")

    # SAE types
    parser.add_argument("--primary_sae_type", type=str, default="batchtopk",
                        choices=["batchtopk", "jumprelu", "topk", "vanilla"],
                        help="SAE architecture for primary SAE")
    parser.add_argument("--meta_sae_type", type=str, default="batchtopk",
                        choices=["batchtopk", "jumprelu", "topk", "vanilla"],
                        help="SAE architecture for meta SAE")
    parser.add_argument("--bandwidth", type=float, default=0.001,
                        help="Bandwidth for JumpReLU SAE (epsilon parameter)")

    # Training hyperparameters (for training script)
    parser.add_argument("--num_tokens", type=int, default=100000000, help="Number of tokens to process during training")
    parser.add_argument("--batch_size", type=int, default=1024, help="SAE training batch size (activations per batch)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lambda2", type=float, default=1e-3, help="Weight for decomposability penalty")
    parser.add_argument("--sigma_sq", type=float, default=0.1, help="Variance for penalty function")
    parser.add_argument("--n_primary_steps", type=int, default=10, help="# primary SAE steps per alternation")
    parser.add_argument("--n_meta_steps", type=int, default=5, help="# meta SAE steps per alternation")

    # Buffering/memory knobs
    parser.add_argument("--num_batches_in_buffer_joint", type=int, default=5, help="# model batches to store in activation buffer for joint training")
    parser.add_argument("--num_batches_in_buffer_sequential", type=int, default=3, help="# model batches to store in activation buffer for sequential training")
    parser.add_argument("--num_batches_in_buffer", type=int, default=3, help="# model batches to store in activation buffer for assessment")

    # Assessment knobs
    parser.add_argument("--num_assessment_batches", type=int, default=1000, help="# batches to analyze in assessments")

    # Phase toggles (training)
    parser.add_argument("--train_joint_saes", action="store_true", default=False, help="Run joint training phase")
    parser.add_argument("--train_sequential_saes", action="store_true", default=False, help="Run sequential phases")

    # Phase toggles (assessment)
    parser.add_argument("--assess_joint_saes", action="store_true", default=True, help="Assess joint-trained models")
    parser.add_argument("--assess_sequential_saes", action="store_true", default=True, help="Assess sequentially-trained models")

    # Checkpoint paths
    parser.add_argument("--joint_primary_path", type=str, default="joint_primary_sae.pt", help="Path to save/load joint primary SAE checkpoint")
    parser.add_argument("--joint_meta_path", type=str, default="joint_meta_sae.pt", help="Path to save/load joint meta SAE checkpoint")
    parser.add_argument("--solo_primary_path", type=str, default="solo_primary_sae.pt", help="Path to save/load solo primary SAE checkpoint")
    parser.add_argument("--sequential_meta_path", type=str, default="sequential_meta_sae.pt", help="Path to save/load sequential meta SAE checkpoint")

    return parser


def parse_args_common(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Common argument parsing logic for both training and assessment scripts.
    Handles interactive mode detection and Jupyter kernel argument filtering.
    """
    # Default dataset configuration
    DEFAULT_DATASET_PATH = "HuggingFaceFW/fineweb"
    DEFAULT_DATASET_NAME = "sample-10BT"

    try:
        filtered_argv = [arg for arg in sys.argv if not arg.startswith('--f=')]
        if len(filtered_argv) == 1 or any('ipykernel' in arg for arg in sys.argv):
            print("🧪 Interactive mode detected, using parser defaults")
            args = parser.parse_args([])
            if getattr(args, 'dataset_path', None) is None:
                args.dataset_path = DEFAULT_DATASET_PATH
                args.dataset_name = DEFAULT_DATASET_NAME
            return args
        else:
            original_argv = sys.argv
            sys.argv = filtered_argv
            try:
                args = parser.parse_args()
                if getattr(args, 'dataset_path', None) is None:
                    args.dataset_path = DEFAULT_DATASET_PATH
                    args.dataset_name = DEFAULT_DATASET_NAME
                    print(f"⚠️ No dataset_path provided, using default: {DEFAULT_DATASET_PATH}/{DEFAULT_DATASET_NAME}")
                return args
            finally:
                sys.argv = original_argv
    except SystemExit:
        print("🧪 Failed to parse command line args, using parser defaults")
        args = parser.parse_args([])
        if getattr(args, 'dataset_path', None) is None:
            args.dataset_path = DEFAULT_DATASET_PATH
            args.dataset_name = DEFAULT_DATASET_NAME
    return args


def load_model_and_set_sizes(cfg: Dict[str, Any]) -> Any:
    """Load the transformer model specified by ``cfg['model_name']`` and update act_size."""
    try:
        from transformer_lens import HookedTransformer
    except ImportError as e:
        raise ImportError("transformer_lens is required to load the model. Install via 'pip install transformer_lens'.") from e

    model_name = cfg["model_name"]
    model = HookedTransformer.from_pretrained(model_name, device=cfg["device"])
    hook_point = cfg["hook_point"] if "hook_point" in cfg else None
    if hook_point is None:
        from transformer_lens.utils import get_act_name
        hook_point = get_act_name(cfg["site"], cfg["layer"])
        cfg["hook_point"] = hook_point
    act_size = model.cfg.d_model
    try:
        dummy_tokens = model.tokenizer.encode("Hello world", return_tensors="pt").to(cfg["device"])
        _, cache = model.run_with_cache(dummy_tokens, names_filter=[hook_point], stop_at_layer=cfg["layer"] + 1)
        act_tensor = cache[hook_point]
        act_size = act_tensor.shape[-1]
    except Exception:
        pass
    cfg["act_size"] = act_size
    return model


def print_gpu_memory_usage():
    """Print current GPU memory usage for monitoring."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB, Peak: {max_allocated:.1f}GB")
    else:
        print("CUDA not available - using CPU")


def print_configs(cfg, meta_cfg, penalty_cfg=None):
    """Print all configuration dictionaries in a readable format."""
    import pprint
    print("=== Primary SAE Config (cfg) ===")
    pprint.pprint(cfg)
    print("\n=== Meta SAE Config (meta_cfg) ===")
    pprint.pprint(meta_cfg)
    if penalty_cfg is not None:
        print("\n=== Penalty Config (penalty_cfg) ===")
        pprint.pprint(penalty_cfg)
    print("=" * 40) 