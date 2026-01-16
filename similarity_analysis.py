import torch
import numpy as np
import pandas as pd
import tqdm
from collections import defaultdict
from datasets import load_dataset
import gc

# We assume the SAE and model classes are importable from your project
# from sae import BatchTopKSAE
# from transformer_lens import HookedTransformer

class FunctionalSimilarityAssessor:
    """
    Compares two SAEs based on the functional similarity of their features,
    determined by the rank correlation of their top activating contexts.
    """
    def __init__(self, solo_sae, joint_sae, model, cfg):
        """
        Initializes the assessor.

        Args:
            solo_sae (BatchTopKSAE): The standard SAE trained alone.
            joint_sae (BatchTopKSAE): The SAE trained with the meta-SAE penalty.
            model (HookedTransformer): The base model.
            cfg (dict): A configuration dictionary containing dataset_path, seq_len, etc.
        """
        self.solo_sae = solo_sae.eval()
        self.joint_sae = joint_sae.eval()
        self.model = model
        self.cfg = cfg
        self.device = self.solo_sae.cfg["device"]

        self.solo_top_contexts = None
        self.joint_top_contexts = None

    @torch.no_grad()
    def _collect_top_contexts(self, sae, num_batches=1000, top_k_contexts=100, model_batch_size=8):
        """
        Processes a dataset to find the top k activating contexts for each feature.
        This is a memory-intensive operation adapted from feature_statistics.py.
        """
        n_features = sae.cfg['dict_size']
        # Store (activation_value, context_id)
        top_contexts = [[(-1.0, -1)] * top_k_contexts for _ in range(n_features)]
        min_activations = torch.full((n_features,), -1.0, device=self.device)

        # Get dataset config
        dataset_path = self.cfg.get("dataset_path", "HuggingFaceFW/fineweb")
        dataset_name = self.cfg.get("dataset_name", "sample-10BT")

        if dataset_name:
            dataset = load_dataset(dataset_path, name=dataset_name, streaming=True, split="train")
        else:
            dataset = load_dataset(dataset_path, streaming=True, split="train")
        ds_iter = iter(dataset)

        pbar = tqdm.tqdm(range(num_batches), desc=f"Finding Top Contexts (Dict Size: {n_features})")
        for batch_idx in pbar:
            # --- Data loading logic adapted from user's successful implementation ---
            batch_texts = []
            # Can manually send in None to use the config batch size
            if model_batch_size is None:
                model_batch_size = self.cfg.get("model_batch_size", 8)
            for _ in range(model_batch_size):
                try:
                    example = next(ds_iter)
                    batch_texts.append(example["text"])
                except StopIteration:
                    if dataset_name:
                        dataset = load_dataset(dataset_path, name=dataset_name, streaming=True, split="train")
                    else:
                        dataset = load_dataset(dataset_path, streaming=True, split="train")
                    ds_iter = iter(dataset)
                    example = next(ds_iter)
                    batch_texts.append(example["text"])

            batch_tokens = self.model.tokenizer(
                batch_texts, padding=True, truncation=True, 
                max_length=self.cfg['seq_len'], return_tensors="pt"
            ).to(self.device)

            _, cache = self.model.run_with_cache(
                batch_tokens["input_ids"],
                names_filter=[sae.cfg["hook_point"]],
                stop_at_layer=sae.cfg["layer"] + 1
            )
            activations = cache[sae.cfg["hook_point"]].reshape(-1, sae.cfg["act_size"])
            
            feature_acts = sae(activations)['feature_acts']

            # Efficiently find activations that are higher than the current minimums
            candidate_mask = feature_acts > min_activations.unsqueeze(0)
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=True)
            
            # Unpack indices
            context_indices, feature_indices = candidate_indices
            
            for context_idx, feature_idx in zip(context_indices, feature_indices):
                activation_value = feature_acts[context_idx, feature_idx].item()
                # A "context_id" is a unique identifier for a token's activation
                context_id = batch_idx * activations.shape[0] + context_idx.item()
                
                # Replace the smallest element if the new one is larger
                if activation_value > top_contexts[feature_idx][-1][0]:
                    top_contexts[feature_idx][-1] = (activation_value, context_id)
                    # Re-sort and update the minimum activation for this feature
                    top_contexts[feature_idx].sort(key=lambda x: x[0], reverse=True)
                    min_activations[feature_idx] = top_contexts[feature_idx][-1][0]
            
            del activations, feature_acts, batch_tokens, cache, candidate_mask, context_indices, feature_indices
            if batch_idx % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Return a dictionary of {feature_idx: [ranked_context_ids]}
        return {f_idx: [ctx[1] for ctx in contexts] for f_idx, contexts in enumerate(top_contexts)}

    def _calculate_spearman_correlation(self, list1, list2):
        """Calculates Spearman's rank correlation for two lists of context IDs."""
        # Create ranks for items in each list
        rank1 = {item: i for i, item in enumerate(list1)}
        rank2 = {item: i for i, item in enumerate(list2)}
        
        # Find common items
        common_items = set(rank1.keys()) & set(rank2.keys())
        
        if len(common_items) < 2:
            return 0.0 # Correlation is undefined for less than 2 points

        # Get ranks of common items
        common_ranks1 = [rank1[item] for item in common_items]
        common_ranks2 = [rank2[item] for item in common_items]
        
        # Calculate Pearson correlation on the ranks
        mean1, mean2 = np.mean(common_ranks1), np.mean(common_ranks2)
        cov = np.sum((np.array(common_ranks1) - mean1) * (np.array(common_ranks2) - mean2))
        std1, std2 = np.std(common_ranks1), np.std(common_ranks2)
        
        if std1 * std2 == 0:
            return 0.0
            
        return cov / (len(common_items) * std1 * std2)

    def find_best_matches(self):
        """
        Finds the best matching feature in the solo SAE for each feature in the joint SAE.
        """
        if self.joint_top_contexts is None or self.solo_top_contexts is None:
            print("Top contexts not collected. Running collection first.")
            self.solo_top_contexts = self._collect_top_contexts(self.solo_sae)
            self.joint_top_contexts = self._collect_top_contexts(self.joint_sae)

        best_matches = {}
        pbar = tqdm.tqdm(self.joint_top_contexts.items(), desc="Finding Best Matches")
        for joint_idx, joint_contexts in pbar:
            best_corr = -1.0
            best_solo_idx = -1
            
            for solo_idx, solo_contexts in self.solo_top_contexts.items():
                corr = self._calculate_spearman_correlation(joint_contexts, solo_contexts)
                if corr > best_corr:
                    best_corr = corr
                    best_solo_idx = solo_idx
            
            best_matches[joint_idx] = {
                "best_match_solo_idx": best_solo_idx,
                "correlation": best_corr
            }
            pbar.set_postfix({"LastCorr": f"{best_corr:.3f}"})
        
        return best_matches

    def run_analysis(self):
        """
        Orchestrates the full functional similarity analysis.
        """
        print("--- Starting Functional Similarity Analysis ---")
        best_matches = self.find_best_matches()
        
        # Create a DataFrame for easy analysis
        results_df = pd.DataFrame.from_dict(best_matches, orient='index')
        
        print("\n--- Analysis Complete ---")
        print("Correlation Score Distribution:")
        print(results_df['correlation'].describe())
        
        # You can now plot a histogram of results_df['correlation']
        # import matplotlib.pyplot as plt
        # results_df['correlation'].hist(bins=50)
        # plt.title("Distribution of Best-Match Functional Similarity Scores")
        # plt.xlabel("Spearman Correlation")
        # plt.ylabel("Frequency")
        # plt.show()
        
        return results_df

if __name__ == '__main__':
    print("This file contains the FunctionalSimilarityAssessor class.")
    print("To use it, import it and run the analysis on your trained SAEs.")