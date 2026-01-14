#%%
import os 
import wandb
import json
import sys
sys.path.append("/workspace/Gemma/GemmaSAELens")
from sae_lens import SAE
from transformer_lens import HookedTransformer
from meta_saes.activation_store import ActivationsStore
import sys
sys.path.append("/workspace/meta_saes/")
from meta_saes.sae import BatchTopKSAE, load_gpt2_saes, load_sae
import torch
from feature_statistics import FeatureStatistics

# Assuming you have a pre-trained base SAE
model, josephs_saes, token_dataset = load_gpt2_saes(device='cuda:0')
base_sae = josephs_saes[-2]  # You need to implement this function to load your base SAE
print(base_sae.W_dec.shape)
meta_sae, cfg = load_sae(
    "mats-sprint/meta_sae/gpt2-small_blocks.8.hook_resid_pre_24576_topk_8_0.001_240000:v0",
    BatchTopKSAE,
)
activations_store = ActivationsStore(model, meta_sae.cfg)
# %%
import tqdm
from collections import defaultdict
def corrected_feature_clustering(base_sae, meta_sae, device="cuda"):
    feature_to_clusters = {}
    for i in tqdm.tqdm(range(base_sae.W_dec.shape[0])):
        x = base_sae.W_dec[i].unsqueeze(0).to(device)
        with torch.no_grad():
            activations = meta_sae(x, threshold=0.07)["feature_acts"].squeeze()

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
feature_to_clusters = corrected_feature_clustering(base_sae, meta_sae)
cluster_to_features = create_cluster_to_features(feature_to_clusters)

#%%
import matplotlib.pyplot as plt
#make a histogram of the number of features in each cluster
fig, ax = plt.subplots()
ax.hist([len(cluster_to_features[cluster]) for cluster in cluster_to_features], bins=50, range=(0, 50))
ax.set_xlabel("Number of features in cluster")
ax.set_ylabel("Number of clusters")
plt.show()

import matplotlib.pyplot as plt
#make a histogram of the number of clusters for each feature
fig, ax = plt.subplots()
ax.hist([len(feature_to_clusters[feature]) for feature in feature_to_clusters], bins=50, range=(0, 50))
ax.set_xlabel("Number of clusters for feature")
ax.set_ylabel("Number of features")
plt.show()

#%%
#%%
import json
import urllib.parse
import webbrowser

# feature = 8780
# feature = 8307
feature = 41351
clusters = feature_to_clusters[feature]
clusters.sort(key=lambda x: x[1], reverse=True)
print(clusters[:20])
for cluster in clusters[:20]:
    cluster = cluster[0]
    LIST_NAME = f"cluster_{cluster}"
    LIST_FEATURES = []
    print(len(cluster_to_features[cluster]))
    for feature in cluster_to_features[cluster][:20]:
        LIST_FEATURES.append({"modelId": "gpt2-small", "layer": "8-res_fs49152-jb", "index": str(feature[0])})


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
cluster_num = 24245
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
    sae_output = sae(batch).reshape(input_ids.shape[0], input_ids.shape[1], -1)
    
    logits = model.run_with_hooks(
        input_ids,
        fwd_hooks=[(sae.cfg.hook_point, partial(reconstr_hook, sae_out=sae_output))],
        return_type="logits",
    )
    return logits[0, -1, model.to_single_token(logit_true)] - logits[0, -1, model.to_single_token(logit_false)]

def generate_with_sae(model, sae, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
    input_ids = model.to_tokens(prompt, prepend_bos=True)
    
    def reconstr_hook(activation, hook, sae_out):
        return sae_out

    for _ in range(max_new_tokens):
        with torch.no_grad():
            output, cache = model.run_with_cache(input_ids)
            batch = cache[sae.cfg.hook_point].reshape(-1, model.cfg.d_model)
            sae_output = sae(batch).reshape(input_ids.shape[0], input_ids.shape[1], -1)
            
            logits = model.run_with_hooks(
                input_ids,
                fwd_hooks=[(sae.cfg.hook_point, partial(reconstr_hook, sae_out=sae_output))],
                return_type="logits",
            )
            
            next_token_logits = logits[0, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    
    return model.to_string(input_ids[0])

def generate_without_sae(model, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
    input_ids = model.to_tokens(prompt, prepend_bos=True)
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            
            next_token_logits = logits[0, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    
    return model.to_string(input_ids[0])


def get_reconstructed_sae(base_sae, meta_sae):
    reconstructed_sae = copy.deepcopy(base_sae)
    for i in tqdm.tqdm(range(base_sae.W_dec.shape[0])):
        feature = base_sae.W_dec[i].unsqueeze(0)
        reconstructed_sae.W_dec.data[i] = meta_sae(feature, threshold=0.06)["sae_out"]
    return reconstructed_sae

def get_adapted_sae(reconstructed_sae, meta_sae, cluster_of_interest):
    adapted_sae_all = copy.deepcopy(reconstructed_sae)
    for i in tqdm.tqdm(range(reconstructed_sae.W_dec.shape[0])):
        feature = reconstructed_sae.W_dec[i].unsqueeze(0)
        meta_output = meta_sae(feature, threshold=0.06)
        meta_acts = meta_output["feature_acts"].squeeze()
        
        if meta_acts[cluster_of_interest] > 0:
            meta_component = meta_sae.W_dec[cluster_of_interest] * meta_acts[cluster_of_interest]
            modified_feature = feature - meta_component
            adapted_sae_all.W_dec.data[i] = modified_feature
    return adapted_sae_all

def get_logits_for_input(input_text, logit_true, logit_false, model, base_sae, meta_sae, cluster_of_interest):
    input_ids = model.to_tokens(input_text)
    
    original_logit = get_logit_difference(model, input_ids)
    sae_logit = get_logit_difference_with_sae(model, input_ids, base_sae)
    
    reconstructed_sae = get_reconstructed_sae(base_sae, meta_sae)
    sae_logit2 = get_logit_difference_with_sae(model, input_ids, reconstructed_sae)
    
    adapted_sae_all = get_adapted_sae(reconstructed_sae, meta_sae, cluster_of_interest)
    sae_logit4 = get_logit_difference_with_sae(model, input_ids, adapted_sae_all)
    
    return [original_logit.item(), sae_logit.item(), sae_logit2.item(), sae_logit4.item()]

#%%
# Setup
model = HookedTransformer.from_pretrained("gpt2-small")
cluster_of_interest = 24245

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
reconstructed_sae = get_reconstructed_sae(base_sae, meta_sae)
adapted_sae = get_adapted_sae(reconstructed_sae, meta_sae, cluster_of_interest)
#%%
prompt = "Marie Curie"
generated_text = generate_without_sae(model, prompt, temperature=0.01)
print(generated_text)
generated_text = generate_with_sae(model, base_sae, prompt, temperature=0.01)
print(generated_text)
generated_text = generate_with_sae(model, reconstructed_sae, prompt, temperature=0.01)
print(generated_text)
generated_text = generate_with_sae(model, adapted_sae, prompt, temperature=0.01)
print(generated_text)
# %%


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
from meta_saes.activation_store import ActivationsStore
import copy

def get_cross_entropy(model, batch_tokens, batch):
    loss = model(batch_tokens, return_type="loss")
    return loss.item()

def get_cross_entropy_with_sae(model, batch_tokens, batch, sae):
    def reconstr_hook(activation, hook, sae_out):
        return sae_out
    sae_out = sae(batch).sae_out.reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)
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
#%%

import matplotlib.pyplot as plt
import numpy as np

# Choose two meta-features for x and y axes
meta_feature_x = 0  
meta_feature_y = 1  

# Compute activations of base features on these meta-features
activations_x = []
activations_y = []

for i in tqdm.tqdm(range(base_sae.W_dec.shape[0])):
    feature = base_sae.W_dec[i].unsqueeze(0).to(meta_sae.cfg["device"])
    with torch.no_grad():
        meta_activations = meta_sae(feature)["feature_acts"].squeeze()
    activations_x.append(meta_activations[meta_feature_x].item())
    activations_y.append(meta_activations[meta_feature_y].item())

# Create the scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(activations_x, activations_y, alpha=0.5)
plt.xlabel(f'Meta-feature {meta_feature_x}')
plt.ylabel(f'Meta-feature {meta_feature_y}')
plt.title('Base Features Projected onto Two Meta-Features')

# # Add a colorbar to represent density
# density = plt.hexbin(activations_x, activations_y, gridsize=20, cmap='viridis')
# plt.colorbar(density, label='Density')

# Optionally, highlight some specific features
# For example, highlight the top 5 most activated features for each meta-feature
#%%
top_x = np.argsort(activations_x)[-5:]
top_y = np.argsort(activations_y)[-5:]
plt.scatter([activations_x[i] for i in top_x], [activations_y[i] for i in top_x], color='red', s=100, label='Top X')
plt.scatter([activations_x[i] for i in top_y], [activations_y[i] for i in top_y], color='green', s=100, label='Top Y')

plt.legend()
plt.tight_layout()
plt.show()

# Print information about the most activated features
print("Top 5 features for meta-feature X:")
for i in top_x:
    print(f"Base feature {i}: activation = {activations_x[i]:.4f}")

print("\nTop 5 features for meta-feature Y:")
for i in top_y:
    print(f"Base feature {i}: activation = {activations_y[i]:.4f}")

# %%
