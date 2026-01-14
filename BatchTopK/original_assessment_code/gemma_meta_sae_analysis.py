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
from meta_saes.sae import BatchTopKSAE
import torch
from feature_statistics import FeatureStatistics

torch.set_grad_enabled(False)
#%%
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


def load_gemma_sae():
    os.environ["GEMMA_2_SAE_WEIGHTS_ROOT"] = "/workspace/Gemma/weights/"
    assert os.path.exists(os.environ["GEMMA_2_SAE_WEIGHTS_ROOT"])
    device = "cuda"
    INSTRUCTION_TUNED = False  # You can also try True here; this loads a different SAE
    SITE = "post_mlp_residual"
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = "gemma-2-it-saes" if INSTRUCTION_TUNED else "gemma-2-saes",
        sae_id = f"9/{SITE}/16384/0_0006" if INSTRUCTION_TUNED else f"30/{SITE}/131072/0_0005",
        device = device
    )
    return sae

#%%
base_sae = load_gemma_sae()
meta_sae, cfg = load_sae("mats-sprint/meta_sae/gemma-9b_blocks.8.hook_resid_pre_24576_topk_8_0.001_10000:v0", BatchTopKSAE)
stats = FeatureStatistics(base_sae)
cfg = {
"dataset_path": "NeelNanda/c4-10k",
"hook_point": base_sae.cfg.hook_name,
"seq_len": 1024,
"model_batch_size": 2,
"device": "cuda",
"num_batches_in_buffer": 1,
"layer": base_sae.cfg.hook_layer,
"act_size": 3584,
"batch_size": 512
}
model = HookedTransformer.from_pretrained_no_processing("gemma-2-9b").to("cuda")
activations_store = ActivationsStore(model, cfg)
dataset = activations_store.get_complete_tokenized_dataset(add_bos=True)
#%%
stats = FeatureStatistics.load("new_features.pth", base_sae)
#%%
import copy
cluster_to_features = copy.deepcopy(stats.cluster_to_features)
feature_to_clusters = copy.deepcopy(stats.feature_to_clusters)

del stats
del dataset
# %%
def get_top_activating_features(input_string, sae, model, topk=10):
    input_ids = model.to_tokens(input_string, prepend_bos=True)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids)
        activations = cache[sae.cfg.hook_name].squeeze()[1:, :]
        print(activations.shape) 
        feature_activations = sae.encode(activations)
        print(feature_activations.shape)
    
    top_features = torch.topk(feature_activations, k=topk, dim=1)
    
    results = []
    for i, (token, acts) in enumerate(zip(input_ids[0][1:], top_features.values)):
        token_str = model.to_string(token.item())
        results.append({
            'token': token_str,
            'position': i,
            'top_features': list(zip(top_features.indices[i].tolist(), acts.tolist()))
        })
    
    return results

# input_text = "The capital of France is Paris"
# top_activations = get_top_activating_features(input_text, base_sae, model)
# for item in top_activations:
#     print(f"Token: {item['token']}, Position: {item['position']}")
#     print("Top features:", item['top_features'])
#     print()



# meta-feature 10340

#%%
# import tqdm
# from collections import defaultdict
# def corrected_feature_clustering(base_sae, meta_sae, device="cuda"):
#     feature_to_clusters = {}
#     for i in tqdm.tqdm(range(base_sae.W_dec.shape[0])):
#         x = base_sae.W_dec[i].unsqueeze(0).to(device)
#         with torch.no_grad():
#             activations = meta_sae(x, threshold=0.07)["feature_acts"].squeeze()

#         # Find the non-zero indices
#         non_zero_indices = torch.nonzero(activations).squeeze(1)
#         non_zero_acts = activations[non_zero_indices]
        
#         feature_to_clusters[i] = [(idx.item(), act.item()) for idx, act in zip(non_zero_indices, non_zero_acts)]
    
#     return feature_to_clusters


# def create_cluster_to_features(feature_to_clusters):
#     cluster_to_features = defaultdict(list)
    
#     for feature, clusters in feature_to_clusters.items():
#         for cluster, activation in clusters:
#             cluster_to_features[cluster].append((feature, activation))
    
#     # Optional: Sort each cluster's features by activation (descending order)
#     for cluster in cluster_to_features:
#         cluster_to_features[cluster].sort(key=lambda x: x[1], reverse=True)
    
#     return cluster_to_features

# # Usage
# feature_to_clusters = corrected_feature_clustering(base_sae, meta_sae)
# cluster_to_features = create_cluster_to_features(feature_to_clusters)

#%%
# import matplotlib.pyplot as plt
# #make a histogram of the number of features in each cluster
# fig, ax = plt.subplots()
# ax.hist([len(cluster_to_features[cluster]) for cluster in cluster_to_features], bins=50, range=(0, 50))
# ax.set_xlabel("Number of features in cluster")
# ax.set_ylabel("Number of clusters")
# plt.show()

# import matplotlib.pyplot as plt
# #make a histogram of the number of clusters for each feature
# fig, ax = plt.subplots()
# ax.hist([len(feature_to_clusters[feature]) for feature in feature_to_clusters], bins=50, range=(0, 50))
# ax.set_xlabel("Number of clusters for feature")
# ax.set_ylabel("Number of features")
# plt.show()

#%%


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
        out = sae_out
        print(out.shape, activation.shape)
        out[:, 0, :] = activation[:, 0, :]
        return out

    for _ in range(max_new_tokens):
        with torch.no_grad():
            output, cache = model.run_with_cache(input_ids)
            batch = cache[sae.cfg.hook_name].reshape(-1, model.cfg.d_model)
            sae_output = sae(batch).reshape(input_ids.shape[0], input_ids.shape[1], -1)
            
            logits = model.run_with_hooks(
                input_ids,
                fwd_hooks=[(sae.cfg.hook_name, partial(reconstr_hook, sae_out=sae_output))],
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

def get_adapted_sae(reconstructed_sae, meta_sae, cluster_of_interest, cluster_to_features):
    adapted_sae_all = copy.deepcopy(reconstructed_sae)
    for i, _ in cluster_to_features[cluster_of_interest]:
        feature = reconstructed_sae.W_dec[i].unsqueeze(0)
        meta_output = meta_sae(feature, threshold=0.06)
        meta_acts = meta_output["feature_acts"].squeeze()
        
        if meta_acts[cluster_of_interest] > 0:
            print(f"adapting feature {i}")
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
# cluster_of_interest = 14013

# # Prepare data for both plots
# inputs = [
#     ("Paris is the capital of", " France", " Germany"),
#     ("Berlin is the capital of", " Germany", " France"),
# ]

# all_values = []
# for input_text, logit_true, logit_false in inputs:
#     values = get_logits_for_input(input_text, logit_true, logit_false, model, base_sae, meta_sae, cluster_of_interest)
#     all_values.append(values)
# #%%
# # Create the plots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# labels = ['Original', 'SAE reconstruction', "Reconstructed \n SAE reconstruction", 'Modified Reconstructed \n SAE reconstruction']
# colors = ['blue', 'green', 'red', "purple"]

# for ax, (input_text, logit_true, logit_false), values in zip([ax1, ax2], inputs, all_values):
#     bars = ax.bar(labels, values)
    
#     ax.set_ylabel('Logit Difference')
#     ax.set_title(f'Logit Difference Comparison\n{input_text} ({logit_true} - {logit_false})')
#     ax.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
    
#     for bar, value in zip(bars, values):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.4f}',
#                 ha='center', va='bottom')
    
#     for bar, color in zip(bars, colors):
#         bar.set_color(color)
    
#     ax.set_ylim(min(min(values) - 0.5, -0.5), max(max(values) + 0.5, 0.5))

#     #rotate the x-axis labels
#     for tick in ax.get_xticklabels():
#         tick.set_rotation(45)

# plt.tight_layout()
# plt.show()

#%%
#%%
del base_sae
import gc
gc.collect()
torch.cuda.empty_cache()
gc.collect()
base_sae = load_gemma_sae()
base_sae = get_adapted_sae(base_sae, meta_sae, 88, cluster_to_features)
prompt = "We should prevent our children from starting"
generated_text = generate_with_sae(model, base_sae, prompt, temperature=0.001, max_new_tokens=10)
print(generated_text)
generated_text = generate_without_sae(model, prompt, temperature=0.001, max_new_tokens=10)
print(generated_text)


# %%
