import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial

# from BatchTopK.sae import BatchTopKSAE
# from BatchTopK.activation_store import ActivationsStore
# from transformer_lens import HookedTransformer



def assess_sae_thresholds(primary_sae, meta_sae, activation_store, cfg, meta_cfg, num_assessment_batches=1000):
    """
    Assess the threshold distributions for both primary and meta SAEs.
    
    Args:
        primary_sae: The primary SAE model
        meta_sae: The meta SAE wrapper (can be None for solo primary assessment)
        activation_store: Source of activation batches
        cfg: Primary SAE configuration
        meta_cfg: Meta SAE configuration (can be None if meta_sae is None)
        num_assessment_batches: Number of batches to analyze
    
    Returns:
        Dictionary with threshold statistics for both SAEs
    """
    print(f"🔍 Starting threshold assessment over {num_assessment_batches} batches...")
    
    # Set models to eval mode to disable training-specific behavior
    primary_sae.eval()
    if meta_sae is not None:
        meta_sae.eval()
    
    primary_thresholds = []
    
    # Compute meta SAE threshold once since decoder weights don't change
    meta_threshold_value = None
    if meta_sae is not None and meta_cfg is not None:
        print("   Computing meta SAE threshold (decoder weights are constant across batches)...")
        W_dec = primary_sae.W_dec.detach()  # Shape: [dict_size, act_size]
        meta_x, meta_x_mean, meta_x_std = meta_sae.meta_sae.preprocess_input(W_dec)
        meta_x_cent = meta_x - meta_sae.meta_sae.b_dec
        meta_acts = torch.nn.functional.relu(meta_x_cent @ meta_sae.meta_sae.W_enc)
        meta_topk_result = torch.topk(meta_acts.flatten(), meta_cfg["top_k"] * W_dec.shape[0])  # BatchTopK: top_k * num_vectors
        meta_threshold_value = meta_topk_result.values[-1].item()  # Single threshold value
        print(f"   Meta SAE threshold: {meta_threshold_value:.6f}")
    else:
        print("   Skipping meta SAE assessment (no meta SAE provided)")
    
    with torch.no_grad():
        for batch_idx in range(num_assessment_batches):
            if batch_idx % 100 == 0:
                print(f"   Processed {batch_idx}/{num_assessment_batches} batches...")
            
            # Get activation batch for primary SAE
            try:
                batch = activation_store.next_batch()
            except StopIteration:
                print(f"   Dataset exhausted at batch {batch_idx}, stopping assessment.")
                break
                
            # Analyze primary SAE thresholds
            x, x_mean, x_std = primary_sae.preprocess_input(batch)
            x_cent = x - primary_sae.b_dec
            primary_acts = torch.nn.functional.relu(x_cent @ primary_sae.W_enc)
            
            # Get the top_k threshold (value of the k-th largest activation using BatchTopK logic)
            primary_topk_result = torch.topk(primary_acts.flatten(), cfg["top_k"] * x.shape[0])  # BatchTopK: top_k * batch_size
            primary_threshold = primary_topk_result.values[-1].item()  # Smallest value in top-k
            primary_thresholds.append(primary_threshold)
            
            # Meta SAE threshold is constant (computed once above)
            
    # Convert to numpy for statistical analysis
    primary_thresholds = np.array(primary_thresholds)
    
    # Compute comprehensive statistics
    def compute_stats(thresholds, name):
        stats = {
            'count': len(thresholds),
            'mean': np.mean(thresholds),
            'median': np.median(thresholds),
            'std': np.std(thresholds),
            'variance': np.var(thresholds),
            'min': np.min(thresholds),
            'max': np.max(thresholds),
        }
        
        # Add deciles (0th, 10th, 20th, ..., 100th percentiles)
        deciles = np.percentile(thresholds, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        for i, decile in enumerate(deciles):
            stats[f'decile_{i*10}'] = decile
            
        return stats
    
    primary_stats = compute_stats(primary_thresholds, "Primary SAE")
    
    # Print results
    print("\n" + "="*60)
    print("THRESHOLD ASSESSMENT RESULTS")
    print("="*60)
    
    print(f"\n🎯 PRIMARY SAE THRESHOLDS (top_k={cfg['top_k']}):")
    print(f"   Count: {primary_stats['count']}")
    print(f"   Mean: {primary_stats['mean']:.6f}")
    print(f"   Median: {primary_stats['median']:.6f}")
    print(f"   Std Dev: {primary_stats['std']:.6f}")
    print(f"   Variance: {primary_stats['variance']:.6f}")
    print(f"   Range: [{primary_stats['min']:.6f}, {primary_stats['max']:.6f}]")
    print(f"   Deciles:")
    for i in range(0, 101, 10):
        print(f"     {i:3d}%: {primary_stats[f'decile_{i}']:.6f}")
    
    if meta_threshold_value is not None and meta_cfg is not None:
        print(f"\n🎯 META SAE THRESHOLD (top_k={meta_cfg['top_k']}):")
        print(f"   Single threshold value: {meta_threshold_value:.6f}")
        print(f"   (Constant across all batches since decoder weights are fixed)")
    else:
        print(f"\n🎯 META SAE THRESHOLD:")
        print("   Meta SAE threshold not computed (no meta SAE provided).")
    
    print("="*60)
    
    # Set models back to train mode
    primary_sae.train()
    if meta_sae is not None:
        meta_sae.train()
    
    return {
        'primary_thresholds': primary_thresholds,
        'primary_stats': primary_stats,
        'meta_threshold_value': meta_threshold_value
    }


def assess_l0_with_thresholds(primary_sae, meta_sae, activation_store, cfg, meta_cfg, 
                             threshold_stats, num_assessment_batches=1000):
    """
    Assess L0 sparsity when using different threshold values from the first assessment.
    
    Args:
        primary_sae: The primary SAE model
        meta_sae: The meta SAE wrapper (can be None for solo primary assessment)
        activation_store: Source of activation batches (should be shuffled)
        cfg: Primary SAE configuration
        meta_cfg: Meta SAE configuration (can be None if meta_sae is None)
        threshold_stats: Results from assess_sae_thresholds()
        num_assessment_batches: Number of batches to analyze
        
    Returns:
        Dictionary with L0 statistics for different threshold values
    """
    print(f"🎯 Starting L0 assessment with learned thresholds over {num_assessment_batches} batches...")
    
    # Set models to eval mode
    primary_sae.eval() 
    if meta_sae is not None:
        meta_sae.eval()
    
    # Extract threshold values to test
    primary_stats = threshold_stats['primary_stats']
    
    # Test thresholds: mean + deciles 20-80 (including median at 50)
    primary_test_thresholds = {
        'mean': primary_stats['mean'],
        'decile_20': primary_stats['decile_20'],
        'decile_30': primary_stats['decile_30'], 
        'decile_40': primary_stats['decile_40'],
        'decile_50': primary_stats['decile_50'],  # median
        'decile_60': primary_stats['decile_60'],
        'decile_70': primary_stats['decile_70'],
        'decile_80': primary_stats['decile_80'],
    }
    
    # Store L0 results for each threshold
    primary_l0_results = {name: [] for name in primary_test_thresholds.keys()}
    
    # Compute meta SAE L0 values once since decoder weights are constant
    meta_l0_per_column = None
    meta_threshold = None
    if meta_sae is not None and meta_cfg is not None and 'meta_threshold_value' in threshold_stats:
        print("   Computing meta SAE L0 per decoder column (decoder weights are constant)...")
        W_dec = primary_sae.W_dec.detach()  # Shape: [dict_size, act_size]
        meta_x, meta_x_mean, meta_x_std = meta_sae.meta_sae.preprocess_input(W_dec)
        meta_x_cent = meta_x - meta_sae.meta_sae.b_dec
        meta_acts = torch.nn.functional.relu(meta_x_cent @ meta_sae.meta_sae.W_enc)  # Shape: [dict_size, meta_dict_size]
        
        # Use the meta threshold from first assessment to compute L0 per decoder column
        meta_threshold = threshold_stats['meta_threshold_value']
        thresholded_meta_acts = torch.where(meta_acts > meta_threshold, meta_acts, 0.0)
        meta_l0_per_column = (thresholded_meta_acts > 0).sum(dim=-1).cpu().tolist()  # L0 per decoder column
        print(f"   Meta SAE L0 per column: mean={np.mean(meta_l0_per_column):.1f}, "
              f"median={np.median(meta_l0_per_column):.1f}, "
              f"range=[{np.min(meta_l0_per_column):.0f}, {np.max(meta_l0_per_column):.0f}]")
    else:
        print("   Skipping meta SAE L0 computation (no meta SAE provided)")
    
    with torch.no_grad():
        for batch_idx in range(num_assessment_batches):
            if batch_idx % 100 == 0:
                print(f"   Processed {batch_idx}/{num_assessment_batches} batches...")
                
            try:
                batch = activation_store.next_batch()
            except StopIteration:
                print(f"   Dataset exhausted at batch {batch_idx}, stopping assessment.")
                break
                
            # Test primary SAE with different thresholds
            x, x_mean, x_std = primary_sae.preprocess_input(batch)
            x_cent = x - primary_sae.b_dec
            primary_acts = torch.nn.functional.relu(x_cent @ primary_sae.W_enc)
            
            for threshold_name, threshold_val in primary_test_thresholds.items():
                # Apply threshold (like JumpReLU but with universal threshold)
                thresholded_acts = torch.where(primary_acts > threshold_val, primary_acts, 0.0)
                # Calculate L0 per example (number of non-zero activations per example)
                l0_per_example = (thresholded_acts > 0).sum(dim=-1).cpu().tolist()  # Sum over features
                primary_l0_results[threshold_name].extend(l0_per_example)  # Add all example L0s
    
    # Compute statistics for each threshold
    def compute_l0_stats(l0_values):
        l0_array = np.array(l0_values)
        stats = {
            'count': len(l0_array),
            'mean': np.mean(l0_array),
            'median': np.median(l0_array), 
            'std': np.std(l0_array),
            'min': np.min(l0_array),
            'max': np.max(l0_array)
        }
        
        # Add deciles (0th, 10th, 20th, ..., 100th percentiles)
        deciles = np.percentile(l0_array, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        for i, decile in enumerate(deciles):
            stats[f'decile_{i*10}'] = decile
            
        return stats
    
    primary_l0_stats = {name: compute_l0_stats(values) for name, values in primary_l0_results.items()}
    
    # Print results
    print("\n" + "="*60)
    print("L0 ASSESSMENT WITH LEARNED THRESHOLDS")
    print("="*60)
    
    print(f"\n🎯 PRIMARY SAE L0 RESULTS (trained top_k={cfg['top_k']}):")
    # Show number of data points for first threshold as example
    example_stats = list(primary_l0_stats.values())[0]
    print(f"   (Collected {example_stats['count']} individual L0 values across all batches)")
    for threshold_name in primary_test_thresholds.keys():
        threshold_val = primary_test_thresholds[threshold_name]
        stats = primary_l0_stats[threshold_name]
        print(f"   {threshold_name:>10} (thresh={threshold_val:.6f}): "
              f"L0 mean={stats['mean']:6.1f}, median={stats['median']:6.1f}, "
              f"range=[{stats['min']:5.1f}, {stats['max']:5.1f}]")
    
    if meta_l0_per_column is not None and meta_cfg is not None:
        print(f"\n🎯 META SAE L0 RESULTS (trained top_k={meta_cfg['top_k']}):")
        meta_stats = compute_l0_stats(meta_l0_per_column)
        print(f"   (L0 values per decoder column: {len(meta_l0_per_column)} columns)")
        print(f"   Using threshold: {meta_threshold:.6f}")
        print(f"   L0 per column - mean={meta_stats['mean']:6.1f}, median={meta_stats['median']:6.1f}, "
              f"range=[{meta_stats['min']:5.1f}, {meta_stats['max']:5.1f}]")
        print(f"   Deciles:")
        for i in range(0, 101, 10):
            print(f"     {i:3d}%: {meta_stats[f'decile_{i}']:.1f}")
    else:
        print(f"\n🎯 META SAE L0 RESULTS:")
        print("   Meta SAE L0 results not computed (no meta SAE provided).")
    
    print("="*60)
    
    # Set models back to train mode
    primary_sae.train()
    if meta_sae is not None:
        meta_sae.train()
    
    return {
        'primary_test_thresholds': primary_test_thresholds,
        'primary_l0_results': primary_l0_results,
        'primary_l0_stats': primary_l0_stats,
        'meta_l0_per_column': meta_l0_per_column,
        'meta_threshold': meta_threshold
    }


# Append this code to your existing assessment.py file


# --- Hooks for CE Impact Assessment (adapted from logs.py) ---

def reconstr_hook(activation, hook, sae_out):
    """Replaces the original activation with the SAE's reconstruction."""
    return sae_out

def zero_abl_hook(activation, hook):
    """Ablates the activation to zero."""
    return torch.zeros_like(activation)

# --- The Assessor Class ---

class ReconstructionAssessor:
    """
    A class to perform a comprehensive assessment of a trained SAE's performance.
    """
    def __init__(self, sae, model, activation_store):
        """
        Initializes the assessor.

        Args:
            sae (BatchTopKSAE): The trained SAE to evaluate.
            model (HookedTransformer): The base model.
            activation_store (ActivationsStore): The data source for held-out activations.
        """
        self.sae = sae
        self.model = model
        self.activation_store = activation_store
        self.device = self.sae.cfg["device"]
        self.sae.eval() # Set to evaluation mode

    @torch.no_grad()
    def assess_l2_loss(self, num_batches=100):
        """Calculates the average L2 reconstruction loss (MSE)."""
        total_l2_loss = 0.0
        pbar = tqdm(range(num_batches), desc="Assessing L2 Loss")
        for _ in pbar:
            batch = self.activation_store.next_batch()
            output = self.sae(batch)
            total_l2_loss += output['l2_loss'].item()
            pbar.set_postfix({"MSE": f"{total_l2_loss / (pbar.n + 1):.4f}"})
        return total_l2_loss / num_batches

    @torch.no_grad()
    def assess_ce_impact(self, num_batches=100):
        """
        Assesses the impact on the model's cross-entropy loss.
        Returns the loss with reconstruction, the loss with zero ablation, and the original loss.
        """
        total_reconstr_loss = 0.0
        total_zero_abl_loss = 0.0
        total_original_loss = 0.0
        
        # Use a small batch size for the model to avoid OOM
        model_batch_size = min(self.sae.cfg.get("model_batch_size", 64), 32)
        seq_len = self.sae.cfg.get("seq_len", 128)

        # Get dataset config from SAE config
        dataset_path = self.sae.cfg.get("dataset_path", "HuggingFaceFW/fineweb")
        dataset_name = self.sae.cfg.get("dataset_name", "sample-10BT")

        # Create a fresh streaming data loader for CE assessment
        from datasets import load_dataset
        if dataset_name:
            dataset = load_dataset(dataset_path, name=dataset_name, streaming=True, split="train")
        else:
            dataset = load_dataset(dataset_path, streaming=True, split="train")

        pbar = tqdm(range(num_batches), desc="Assessing CE Impact")
        for _ in pbar:
            # Get a fresh batch of tokens from the streaming dataset
            batch_texts = []
            for _ in range(model_batch_size):
                try:
                    example = next(iter(dataset))
                    batch_texts.append(example["text"])
                except StopIteration:
                    # If dataset is exhausted, restart
                    if dataset_name:
                        dataset = load_dataset(dataset_path, name=dataset_name, streaming=True, split="train")
                    else:
                        dataset = load_dataset(dataset_path, streaming=True, split="train")
                    example = next(iter(dataset))
                    batch_texts.append(example["text"])
            
            # Tokenize the batch
            batch_tokens = self.model.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=seq_len,
                return_tensors="pt"
            ).to(self.device)
            
            # Get activations at the hook point for this batch
            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    batch_tokens["input_ids"], 
                    names_filter=[self.sae.cfg["hook_point"]],
                    stop_at_layer=self.sae.cfg["layer"] + 1
                )
                batch_activations = cache[self.sae.cfg["hook_point"]]
            
            # Reshape activations for SAE
            sae_in = batch_activations.reshape(-1, self.sae.cfg["act_size"])
            sae_out = self.sae(sae_in)['sae_out'].reshape(batch_activations.shape)

            # Run model with hooks using real tokens and SAE reconstruction
            reconstr_loss = self.model.run_with_hooks(
                batch_tokens["input_ids"],
                fwd_hooks=[(self.sae.cfg["hook_point"], partial(reconstr_hook, sae_out=sae_out))],
                return_type="loss",
            ).item()

            zero_loss = self.model.run_with_hooks(
                batch_tokens["input_ids"],
                fwd_hooks=[(self.sae.cfg["hook_point"], zero_abl_hook)],
                return_type="loss",
            ).item()
            
            # Get original model loss (no SAE intervention)
            original_loss = self.model(batch_tokens["input_ids"], return_type="loss").item()
            
            total_reconstr_loss += reconstr_loss
            total_zero_abl_loss += zero_loss
            total_original_loss += original_loss
            
            pbar.set_postfix({
                "Original": f"{total_original_loss / (pbar.n + 1):.4f}",
                "Recon Loss": f"{total_reconstr_loss / (pbar.n + 1):.4f}",
                "Zero Abl Loss": f"{total_zero_abl_loss / (pbar.n + 1):.4f}"
            })
            
            # Clean up intermediate tensors
            del batch_activations, sae_in, sae_out, batch_tokens

        return {
            "original_loss": total_original_loss / num_batches,
            "reconstruction_loss": total_reconstr_loss / num_batches,
            "zero_ablation_loss": total_zero_abl_loss / num_batches,
        }

    @torch.no_grad()
    def assess_feature_utilization(self, num_batches=1000):
        """
        Assesses feature sparsity (L0 norm) and counts dead features.
        Uses a larger number of batches to get a reliable dead feature count.
        """
        total_l0 = 0.0
        feature_activations = torch.zeros(self.sae.cfg['dict_size'], device=self.device)

        pbar = tqdm(range(num_batches), desc="Assessing Feature Utilization")
        for _ in pbar:
            batch = self.activation_store.next_batch()
            output = self.sae(batch)
            total_l0 += output['l0_norm'].item()
            
            # Sum activations across the batch dimension to find active features
            feature_activations += (output['feature_acts'] > 0).float().sum(dim=0)
            
            pbar.set_postfix({
                "Avg L0": f"{total_l0 / (pbar.n + 1):.2f}",
                "Dead": f"{(feature_activations == 0).sum().item()}"
            })

        avg_l0 = total_l0 / num_batches
        dead_features = (feature_activations == 0).sum().item()
        
        return {
            "avg_l0": avg_l0,
            "dead_features": dead_features,
            "dead_feature_ratio": dead_features / self.sae.cfg['dict_size']
        }
        
    def run_all_assessments(self, l2_batches=100, ce_batches=100, util_batches=1000):
        """
        Runs all assessments and returns a consolidated dictionary of results.
        """
        print(f"--- Starting Comprehensive Assessment for SAE ---")
        print(f"Dict Size: {self.sae.cfg['dict_size']}, Top-K: {self.sae.cfg['top_k']}")
        
        l2_results = self.assess_l2_loss(num_batches=l2_batches)
        ce_results = self.assess_ce_impact(num_batches=ce_batches)
        util_results = self.assess_feature_utilization(num_batches=util_batches)
        
        results = {
            "l2_reconstruction_mse": l2_results,
            **ce_results,
            **util_results
        }
        
        print("\n--- Assessment Complete ---")
        # Pretty print results using pandas for alignment
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
        results_df.index.name = "Metric"
        print(results_df)
        print("---------------------------\n")
        
        return results

# Example usage block to show how to run it
if __name__ == '__main__':
    # This block is for demonstration. You would import and use
    # ReconstructionAssessor in your main evaluation script.
    
    # 1. Load your models and data (placeholders below)
    # cfg = get_default_cfg()
    # model = HookedTransformer.from_pretrained(cfg["model_name"])
    # activation_store = ActivationsStore(model, cfg)
    # sae = BatchTopKSAE(cfg)
    # sae.load_state_dict(torch.load("path_to_your_sae.pt"))

    # 2. Instantiate the assessor
    # assessor = ReconstructionAssessor(sae, model, activation_store)

    # 3. Run the evaluation
    # all_results = assessor.run_all_assessments()

    print("This is an example usage block.")
    print("Import 'ReconstructionAssessor' in your main evaluation script.")