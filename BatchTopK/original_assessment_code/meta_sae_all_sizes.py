#%%% 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from sae import TopKSAE, VanillaSAE, BatchTopKSAE
import wandb 
import os
import sys
sys.path.append("/workspace/SAELens")
from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from config import get_default_cfg, post_init_cfg
from training import train_meta_sae, train_meta_sae_on_W_dec
import json

def load_legacy_saes(start=1, end=8):
    saes = []
    api = wandb.Api()
    for i in range(start, end + 1):
        wandb_link = f"jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small_blocks.8.hook_resid_pre_{768 * 2**(i-1)}:v9"

        artifact = api.artifact(
            wandb_link,
            type="model",
        )
        artifact_dir = artifact.download()
        file = os.listdir(artifact_dir)[0]
        sae = SparseAutoencoder.load_from_pretrained_legacy(
            os.path.join(artifact_dir, file)
        )
        saes.append(sae.to("cuda"))
    return saes


def load_sae(artifact_name, sae_class):
    # Initialize wandb
    api = wandb.Api()

    # Download the artifact
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()

    # Load the configuration
    config_path = os.path.join(artifact_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Convert string representations back to torch.dtype
    if "dtype" in cfg:
        cfg["dtype"] = getattr(torch, cfg["dtype"].split(".")[-1])

    sae = sae_class(cfg)

    # Load the state dict
    state_dict_path = os.path.join(artifact_dir, "sae.pt")
    state_dict = torch.load(state_dict_path, map_location=cfg["device"])
    sae.load_state_dict(state_dict)

    return sae, cfg


#%%
# Assuming you have a pre-trained base SAE
josephs_saes = load_legacy_saes(start=1, end=8)
base_sae = josephs_saes[-1]  # You need to implement this function to load your base SAE
print(base_sae.W_dec.shape)


#%%
from transformer_lens import HookedTransformer
from activation_store import ActivationsStore
import tqdm

@torch.no_grad()
def get_combined_W_dec(saes, activations_store, remove_dead_features=False):
    combined_W_dec = []
    feature_map = {}
    current_index = 0

    if not remove_dead_features:
        for sae_num, sae in enumerate(saes):
            combined_W_dec.append(sae.W_dec)
            for feature_num in range(sae.W_dec.shape[0]):
                feature_map[current_index] = (sae_num, feature_num)
                current_index += 1
        combined_W_dec = torch.cat(combined_W_dec, dim=0)
    else:
        total_acts = [torch.zeros(sae.W_dec.shape[0]).to("cuda") for sae in saes]
        total = 0

        for _ in tqdm.tqdm(range(256)):
            batch = activations_store.next_batch()
            total += batch.shape[0]
            for i, sae in enumerate(saes):
                acts = sae(batch, batch).feature_acts
                total_acts[i] += (acts > 0).float().sum(dim=0)
        
        alive_features = [total_acts[i] / total > 1e-5 for i in range(len(saes))]
        
        for sae_num, (sae, alive) in enumerate(zip(saes, alive_features)):
            sae_W_dec = sae.W_dec[alive].detach()
            combined_W_dec.append(sae_W_dec)
            for feature_num, is_alive in enumerate(alive):
                if is_alive:
                    feature_map[current_index] = (sae_num, feature_num)
                    current_index += 1
        
        combined_W_dec = torch.cat(combined_W_dec, dim=0)

    return combined_W_dec, feature_map




model = HookedTransformer.from_pretrained("gpt2-small").to(cfg["dtype"]).to(cfg["device"])
activations_store = ActivationsStore(model, meta_sae.cfg)

combined_w_dec, feature_map = get_combined_W_dec(josephs_saes, activations_store, remove_dead_features=True)
print(combined_w_dec.shape)
#%%
# train_meta_sae(meta_sae, base_sae)

cfg = get_default_cfg()
cfg["dict_size"] = 4096
cfg["act_size"] = 768
cfg["top_k"] = 32
cfg["lr"] = 1e-3
cfg["l1_coeff"] = 0
cfg["aux_penalty"] = (1/16)
cfg["input_unit_norm"] = False
cfg["wandb_project"] = "meta_sae"
cfg["n_batches_to_dead"] = 100

cfg["threshold"] = None

meta_sae = BatchTopKSAE(cfg)
post_init_cfg(cfg)

train_meta_sae_on_W_dec(meta_sae, combined_w_dec)

# %%
meta_sae, cfg = load_sae("mats-sprint/meta_sae/gpt2-small_blocks.8.hook_resid_pre_24576_topk_8_0.001_240000:v0", BatchTopKSAE)
# %%
import tqdm
from collections import defaultdict
def corrected_feature_clustering(combined_W_dec, meta_sae, device="cuda"):
    feature_to_clusters = {}
    for i in tqdm.tqdm(range(combined_W_dec.shape[0])):
        x = combined_W_dec[i].unsqueeze(0).to(device)
        with torch.no_grad():
            activations = meta_sae(x, threshold=0.06)["feature_acts"].squeeze()

        # Find the non-zero indices
        non_zero_indices = torch.nonzero(activations).squeeze(1)
        non_zero_acts = activations[non_zero_indices]
        
        feature_to_clusters[i] = [(idx.item(), act.item()) for idx, act in zip(non_zero_indices, non_zero_acts)]
    
    return feature_to_clusters


def create_cluster_to_features(feature_to_clusters):
    cluster_to_features = defaultdict(list)
    
    for feature, clusters in feature_to_clusters.items():
        for cluster, activation in clusters:
            cluster_to_features[cluster].append((feature, activation))
    
    # Optional: Sort each cluster's features by activation (descending order)
    for cluster in cluster_to_features:
        cluster_to_features[cluster].sort(key=lambda x: x[1], reverse=True)
    
    return cluster_to_features

# Usage
feature_to_clusters = corrected_feature_clustering(combined_w_dec, meta_sae)
cluster_to_features = create_cluster_to_features(feature_to_clusters)

#%%
import matplotlib.pyplot as plt
#make a histogram of the number of features in each cluster
fig, ax = plt.subplots()
ax.hist([len(cluster_to_features[cluster]) for cluster in cluster_to_features], bins=50, range=(0, 200))
ax.set_xlabel("Number of features in cluster")
ax.set_ylabel("Number of clusters")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
#make a histogram of the number of clusters for each feature
fig, ax = plt.subplots()
ax.hist([len(feature_to_clusters[feature]) for feature in feature_to_clusters], bins=50, range=(0, 50))
ax.set_xlabel("Number of clusters for feature")
ax.set_ylabel("Number of features")
plt.show()

def plot_sae_cluster_histograms(feature_to_clusters, feature_map, num_saes=8):
    # Create a figure with 8 subplots (2 rows, 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Number of Clusters per Feature for each SAE', fontsize=16)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    for sae_number in range(num_saes):
        # Get features for this SAE
        sae_features = [idx for idx, (sae, _) in feature_map.items() if sae == sae_number]
        
        # Count clusters for each feature in this SAE
        cluster_counts = []
        missing_features = []
        for feat in sae_features:
            if feat in feature_to_clusters:
                cluster_counts.append(len(feature_to_clusters[feat]))
            else:
                missing_features.append(feat)
        
        if missing_features:
            print(f"SAE {sae_number}: Missing {len(missing_features)} features in feature_to_clusters")
            print(f"First few missing features: {missing_features[:5]}")
        
        # Plot histogram for this SAE
        ax = axes[sae_number]
        if cluster_counts:
            ax.hist(cluster_counts, bins=50, range=(0, 50), edgecolor='black')
            
            # Add some stats to the plot
            mean_clusters = np.mean(cluster_counts)
            median_clusters = np.median(cluster_counts)
            ax.text(0.95, 0.95, f'Mean: {mean_clusters:.2f}\nMedian: {median_clusters:.2f}', 
                    transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        ax.set_title(f'SAE {sae_number}')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Number of Features')

    plt.tight_layout()
    plt.show()

    # Print some additional information
    print("\nFeature map size:", len(feature_map))
    print("feature_to_clusters size:", len(feature_to_clusters))
    print("\nSample from feature_map:")
    for i, (k, v) in enumerate(list(feature_map.items())[:5]):
        print(f"{k}: {v}")
    print("\nSample from feature_to_clusters:")
    for i, (k, v) in enumerate(list(feature_to_clusters.items())[:5]):
        print(f"{k}: {v}")
plot_sae_cluster_histograms(feature_to_clusters, feature_map, num_saes=8)

#%%
#%%
import json
import urllib.parse
import webbrowser

feature_map_reverse = {v: k for k, v in feature_map.items()}

feature = 4490
sae_number = 5


feature = feature_map_reverse[(sae_number, feature)]

clusters = feature_to_clusters[feature]
clusters.sort(key=lambda x: x[1], reverse=True)

for cluster in clusters[:20]:
    cluster = cluster[0]
    LIST_NAME = f"cluster_{cluster}"
    LIST_FEATURES = []
    print(len(cluster_to_features[cluster]))
    for feature in cluster_to_features[cluster][:20]:
        sae_n, feature_num = feature_map[feature[0]]
        LIST_FEATURES.append({"modelId": "gpt2-small", "layer": f"8-res_fs{768*2**sae_n}-jb", "index": str(feature_num)})


    url = "https://neuronpedia.org/quick-list/"
    name = urllib.parse.quote(LIST_NAME)
    url = url + "?name=" + name
    url = url + "&features=" + urllib.parse.quote(json.dumps(LIST_FEATURES))
    print(url)

# webbrowser.open(url)


# %%
from transformer_lens import HookedTransformer
import pandas as pd
model = HookedTransformer.from_pretrained("gpt2-small").to(cfg["dtype"]).to(cfg["device"])
#%%
cluster_num = 206
cluster_direction = meta_sae.W_dec[cluster_num]
logit_effect = cluster_direction @ model.W_U
logit_effect_topk = torch.topk(logit_effect, 10)
print(logit_effect_topk.indices)

# make a df of the top 10 tokens
top_k_indices = logit_effect_topk.indices
top_k_values = logit_effect_topk.values

# Get the corresponding token strings
top_k_tokens = [model.to_string(idx.item()) for idx in top_k_indices]

# Create a DataFrame
df = pd.DataFrame({
    'Token': top_k_tokens,
    'Logit Effect': top_k_values.tolist()
})

# Display the DataFrame
print(df)


# %%
import copy
import tqdm
from functools import partial


def get_logit_difference(model, input_ids):
    logits = model(input_ids)
    return logits[0, -1, model.to_single_token(logit_true)] - logits[0, -1, model.to_single_token(logit_false)]

def get_logit_difference_with_sae(model, input_ids, sae):
    def reconstr_hook(activation, hook, sae_out):
        return sae_out

    output, cache = model.run_with_cache(input_ids)
    batch = cache[sae.cfg.hook_point].reshape(-1, 768)
    sae_output = sae(batch, batch).sae_out.reshape(input_ids.shape[0], input_ids.shape[1], -1)
    
    logits = model.run_with_hooks(
        input_ids,
        fwd_hooks=[(sae.cfg.hook_point, partial(reconstr_hook, sae_out=sae_output))],
        return_type="logits",
    )
    return logits[0, -1, model.to_single_token(logit_true)] - logits[0, -1, model.to_single_token(logit_false)]

def get_logits_for_input(input_text, logit_true, logit_false, model, base_sae, meta_sae, cluster_of_interest):
    input_ids = model.to_tokens(input_text)
    
    original_logit = get_logit_difference(model, input_ids)
    sae_logit = get_logit_difference_with_sae(model, input_ids, base_sae)
    
    reconstructed_sae = copy.deepcopy(base_sae)
    for i in tqdm.tqdm(range(base_sae.W_dec.shape[0])):
        feature = base_sae.W_dec[i].unsqueeze(0)
        reconstructed_sae.W_dec.data[i] = meta_sae(feature, threshold=0.06)["sae_out"]
    sae_logit2 = get_logit_difference_with_sae(model, input_ids, reconstructed_sae)
    
    adapted_sae_all = copy.deepcopy(reconstructed_sae)
    for i in tqdm.tqdm(range(base_sae.W_dec.shape[0])):
        feature = reconstructed_sae.W_dec[i].unsqueeze(0)
        meta_output = meta_sae(feature, threshold=0.06)
        meta_acts = meta_output["feature_acts"].squeeze()
        
        if meta_acts[cluster_of_interest] > 0:
            meta_component = meta_sae.W_dec[cluster_of_interest] * meta_acts[cluster_of_interest]
            modified_feature = feature - meta_component
            adapted_sae_all.W_dec.data[i] = modified_feature
    
    sae_logit4 = get_logit_difference_with_sae(model, input_ids, adapted_sae_all)
    
    return [original_logit.item(), sae_logit.item(), sae_logit2.item(), sae_logit4.item()]

# Setup
model = HookedTransformer.from_pretrained("gpt2-small")
cluster_of_interest = 1910

# Prepare data for both plots
inputs = [
    ("Paris is the capital of", " France", " Germany"),
    ("Berlin is the capital of", " Germany", " France"),
]

all_values = []
for input_text, logit_true, logit_false in inputs:
    values = get_logits_for_input(input_text, logit_true, logit_false, model, base_sae, meta_sae, cluster_of_interest)
    all_values.append(values)
#%%
# Create the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

labels = ['Original', 'SAE reconstruction', "Reconstructed \n SAE reconstruction", 'Modified Reconstructed \n SAE reconstruction']
colors = ['blue', 'green', 'red', "purple"]

for ax, (input_text, logit_true, logit_false), values in zip([ax1, ax2], inputs, all_values):
    bars = ax.bar(labels, values)
    
    ax.set_ylabel('Logit Difference')
    ax.set_title(f'Logit Difference Comparison\n{input_text} ({logit_true} - {logit_false})')
    ax.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_ylim(min(min(values) - 0.5, -0.5), max(max(values) + 0.5, 0.5))

    #rotate the x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

plt.tight_layout()
plt.show()
#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity_matrix(A, B):
    A_norm = A / A.norm(dim=1)[:, None]
    B_norm = B / B.norm(dim=1)[:, None]
    return torch.mm(A_norm, B_norm.transpose(0, 1))

def get_max_similarities(vec_set1, vec_set2, exclude_self=False):
    similarity_matrix = cosine_similarity_matrix(vec_set1, vec_set2)
    if exclude_self:
        similarity_matrix.fill_diagonal_(-1)  # Exclude self-similarity
    max_similarities, _ = similarity_matrix.max(dim=1)
    return max_similarities.detach().cpu().numpy()

# Assume base_sae and meta_sae are your SAE models
base_decoder = base_sae.W_dec
meta_decoder = meta_sae.W_dec

# Calculate all four sets of maximum cosine similarities
base_to_base = get_max_similarities(base_decoder, base_decoder, exclude_self=True)
base_to_meta = get_max_similarities(base_decoder, meta_decoder)
meta_to_base = get_max_similarities(meta_decoder, base_decoder)
meta_to_meta = get_max_similarities(meta_decoder, meta_decoder, exclude_self=True)

# Create 2x2 plot
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
fig.suptitle('Max Cosine Similarity', fontsize=16)

data = [
    (base_to_base, 'Base-features with closest base-feature'),
    (base_to_meta, 'Base-features with closest meta-feature'),
    (meta_to_base, 'Meta-features with closest base-feature'),
    (meta_to_meta, 'Meta-features with closest meta-feature')
]

for ax, (similarities, title) in zip(axes.flatten(), data):
    ax.hist(similarities, bins=50, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Maximum Cosine Similarity')
    ax.set_ylabel('Frequency')
    
    mean_similarity = np.mean(similarities)
    ax.axvline(mean_similarity, color='r', linestyle='dashed', linewidth=2)
    ax.text(mean_similarity*1.01, ax.get_ylim()[1]*0.9, f'Mean: {mean_similarity:.4f}', 
            rotation=90, verticalalignment='top')

plt.tight_layout()
plt.show()
#%%
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from activation_store import ActivationsStore
import copy

def get_cross_entropy(model, batch_tokens, batch):
    loss = model(batch_tokens, return_type="loss")
    return loss.item()

def get_cross_entropy_with_sae(model, batch_tokens, batch, sae):
    def reconstr_hook(activation, hook, sae_out):
        return sae_out
    sae_out = sae(batch, batch).sae_out.reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)
    output = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg.hook_point, partial(reconstr_hook, sae_out=sae_out))],
        return_type="loss",
    )
    return output.item()

# Load model and SAEs
model = HookedTransformer.from_pretrained("gpt2-small")
# Assume base_sae and meta_sae are already loaded

# Create reconstructed base SAE
reconstructed_sae = copy.deepcopy(base_sae)
for i in tqdm(range(base_sae.W_dec.shape[0]), desc="Reconstructing base SAE"):
    feature = base_sae.W_dec[i].unsqueeze(0)
    reconstructed_sae.W_dec.data[i] = meta_sae(feature, threshold=0.06)["sae_out"]

# Prepare data
activations_store = ActivationsStore(model, meta_sae.cfg)

# Lists to store cross-entropy values for each batch
original_ces = []
base_sae_ces = []
reconstructed_sae_ces = []

# Run evaluation over 100 batches
num_batches = 100
for _ in tqdm(range(num_batches), desc="Evaluating batches"):
    batch_tokens = activations_store.get_batch_tokens()[:meta_sae.cfg["batch_size"] // meta_sae.cfg["seq_len"]]
    input_ids = batch_tokens.to(meta_sae.cfg["device"])
    batch = activations_store.get_activations(batch_tokens).reshape(-1, meta_sae.cfg["act_size"])

    # Calculate cross-entropy for each model
    original_ces.append(get_cross_entropy(model, batch_tokens, batch))
    base_sae_ces.append(get_cross_entropy_with_sae(model, batch_tokens, batch, base_sae))
    reconstructed_sae_ces.append(get_cross_entropy_with_sae(model, batch_tokens, batch, reconstructed_sae))

# Calculate average cross-entropy for each model
avg_original_ce = np.mean(original_ces)
avg_base_sae_ce = np.mean(base_sae_ces)
avg_reconstructed_sae_ce = np.mean(reconstructed_sae_ces)

# Print results
print(f"Average Original LLM Cross-Entropy: {avg_original_ce:.4f}")
print(f"Average Base SAE Cross-Entropy: {avg_base_sae_ce:.4f}")
print(f"Average Reconstructed SAE Cross-Entropy: {avg_reconstructed_sae_ce:.4f}")

# Create bar plot
models = ['Original LLM', 'Base SAE', 'Reconstructed SAE']
ce_values = [avg_original_ce, avg_base_sae_ce, avg_reconstructed_sae_ce]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, ce_values)
plt.title('Average Cross-Entropy Comparison (100 batches)')
plt.ylabel('Average Cross-Entropy Loss')
plt.ylim(0, max(ce_values) * 1.1)  # Set y-axis limit with some margin

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Create box plot to show distribution of cross-entropy values
plt.figure(figsize=(10, 6))
plt.boxplot([original_ces, base_sae_ces, reconstructed_sae_ces], labels=models)
plt.title('Distribution of Cross-Entropy Values (100 batches)')
plt.ylabel('Cross-Entropy Loss')
plt.tight_layout()
plt.show()

