# %%
import gc
import json
import os
import urllib
from dataclasses import dataclass
from itertools import combinations

import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm

from meta_saes.sae import BatchTopKSAE, load_gemma_sae, load_wandb_sae, load_feature_splitting_saes


def clear():
    gc.collect()
    torch.cuda.empty_cache()


# %%

MAIN = __name__ == "__main__"

if MAIN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    model, saes, token_dataset = load_feature_splitting_saes(
        device=device, sae_idxs=[7]
    )
    sae = saes[0].to(device).eval()

    # layer = 25
    # model, saes, token_dataset = load_gemma_sae(
    #     # sae_id=f"layer_{layer}/width_1m/canonical",
    #     # sae_id=f"layer_{layer}/width_16k/canonical",
    #     release="gemma-scope-2b-pt-res-canonical",
    #     # release="gemma-scope-2b-pt-res",
    #     sae_id=f"layer_{layer}/width_65k/canonical",
    #     # sae_id=f"layer_{layer}/width_16k/canonical",
    #     device=device,
    # )
    # sae = saes[0].to(device).eval().to(torch.float16)
    meta_sae, cfg = load_wandb_sae(
        'patrickaaleask/gated-batchtopk/gpt2-small_blocks.8.hook_resid_post_1536_topk_8_0.0001_9001:v0',
        # 'patrickaaleask/gated-batchtopk/gpt2-small_blocks.8.hook_resid_post_768_topk_8_0.001_4001:v0',
        # 'patrickaaleask/gated-batchtopk/gpt2-small_blocks.8.hook_resid_post_768_topk_4_0.0001_5001:v0',
        # "mats-sprint/gemma-2b-layer-25-canonical-tiny-meta-saes/gemma-2-2b_blocks.25.hook_resid_post_2304_topk_4_0.001_2000:v0",
        # 'patrickaaleask/gemma-scope-2b-pt-res-meta-saes/gemma-2-2b_blocks.5.hook_resid_post_2304_topk_4_0.0001_192032:v0',
        BatchTopKSAE,
    )
    meta_sae = meta_sae.to(device).eval().to(torch.float16)
    original_W_dec = sae.W_dec.clone().detach().to(torch.float16)
    sae.W_dec.requires_grad = False
# %%

NEURONPEDIA_DOMAIN = "https://neuronpedia.org"

def quicklist_gpt2_feature_splitting(sae, features):
    """
    Wrapper for neuronpedia quick list for gpt2 feature splitting SAEs.
    """
    url = NEURONPEDIA_DOMAIN + "/quick-list/"
    name = 'bleep'
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    layer = 8
    dataset = 'res_fs49152-jb'
    list_feature = [
        {
            "modelId": 'gpt2-small',
            "layer": f"{layer}-{dataset}",
            "index": str(feature),
        }
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))

    return url

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from meta_saes.training import cartesian_to_hyperspherical

def cartesian_to_hyperspherical(x):
    """
    Convert Cartesian coordinates to hyperspherical coordinates.
    
    Args:
    x (torch.Tensor): Input tensor of shape (..., n) where n is the number of dimensions.
    
    Returns:
    torch.Tensor: Hyperspherical coordinates of shape (..., n-1).
    """
    n = x.shape[-1]
    phi = torch.zeros((*x.shape[:-1], n-1), device=x.device, dtype=x.dtype)
    
    # Compute angles
    for i in range(n-1):
        if i == 0:
            phi[..., i] = torch.acos(x[..., i].clamp(-1, 1))
        else:
            numerator = x[..., i]
            denominator = torch.sqrt(torch.sum(x[..., i:]**2, dim=-1))
            phi[..., i] = torch.acos((numerator / denominator).clamp(-1, 1))
    
    # Handle the last angle separately
    last_coord = x[..., -1]
    phi[..., -1] = torch.where(last_coord < 0, 2*torch.pi - phi[..., -1], phi[..., -1])
    
    return phi

if MAIN:
    meta_acts = meta_sae(cartesian_to_hyperspherical(original_W_dec), threshold=0.13)['feature_acts'].detach()
    meta_2018 = meta_acts[32000].nonzero().squeeze()
    print(meta_2018)
    for i in range(6, len(meta_2018)):
        print(meta_acts[:, meta_2018[i]].nonzero().shape)
        sample = random_sample_nonzero_indices(meta_acts[:, meta_2018[i]], 10)
        print(meta_acts[sample, meta_2018[i]])
        topk_metafs = meta_acts[:, meta_2018[i]].abs().topk(meta_acts[:, meta_2018[i]].nonzero().size(0))
        # take a random sample of the topk features
        sample = topk_metafs.indices[torch.randperm(meta_acts[:, meta_2018[i]].nonzero().size(0))[:100]]
        print(quicklist_gpt2_feature_splitting(sae, sample.tolist()))

        # do pca on the topk features
        # pca = PCA(n_components=2) 
        # pca.fit(sae.W_dec[topk_metafs.indices].cpu().numpy())
        # transformed = pca.transform(sae.W_dec[topk_metafs.indices].cpu().numpy())

        # plt.figure(figsize=(10, 10))
        # plt.scatter(transformed[:, 0], transformed[:, 1])
        # # add labels to the points
        # for j, txt in enumerate(topk_metafs.indices.tolist()):
        #     plt.annotate(txt, (transformed[j][0], transformed[j][1]))
        # plt.show()

        break



# %% [markdown]
# ## Steering city attributes using SAE features

# %%


@dataclass
class CityAttributes:
    name: str
    country: str
    continent: str
    language: str


city_templates = [
    "{} is in the country of{}",
    "{} is in the continent of{}",
    "The primary language spoken in {} is{}",
]


def load_ravel_city_attributes(path="/workspace/ravel/data"):
    cities = {}
    with open(os.path.join(path, "ravel_city_entity_attributes.json"), "r") as f:
        d = json.load(f)
    for city, attributes in d.items():
        cities[city] = CityAttributes(
            city, attributes["Country"], attributes["Continent"], attributes["Language"]
        )
    return cities


def completed_city_strings(city, templates):
    return [
        templates[0].format(city.name, " " + city.country),
        templates[1].format(city.name, " " + city.continent),
        templates[2].format(city.name, " " + city.language),
    ]


def get_common_sae_features(model, sae, inputs, batch_size=32):
    feature_counts = torch.zeros((sae.W_dec.size(0)), device=sae.W_dec.device)
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]

        _, cache = model.run_with_cache(batch, prepend_bos=True)
        cache = cache[sae.cfg.hook_name].detach()[:, 1:].flatten(end_dim=1)
        sae_activations = sae.encode(cache).detach()
        feature_counts += sae_activations.sum(0)

        del cache
        del sae_activations
        torch.cuda.empty_cache()

    return feature_counts


primary_cities = ["Tokyo", "Paris"]
secondary_cities = ["Melbourne", "Marseille"]

if MAIN:
    model.remove_all_hook_fns()
    sae.W_dec.data = original_W_dec.clone().detach().to(torch.float16)
    cities = load_ravel_city_attributes()
    city_counts = [
        get_common_sae_features(
            model, sae, completed_city_strings(cities[city], city_templates)
        )
        for city in primary_cities
    ]
    feature_ratios = [
        city_counts[i] / (sum(city_counts) + 1) for i in range(len(primary_cities))
    ]
    city_features = [fr.topk(1).indices for fr in feature_ratios]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("City Features Ratios")
    top_city_features = [fr.topk(100).indices.tolist() for fr in feature_ratios]
    for i, (city, fr) in enumerate(zip(primary_cities, feature_ratios)):
        print(f"Top {city} features:", fr.topk(10).indices.tolist())

        axs[i].hist(fr[fr > 0].detach().cpu(), bins=100)
        axs[i].set_title(f"{city}")
        axs[i].set_xlabel("Feature Ratio")
        axs[i].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def uncompleted_city_strings(city, templates):
    return [t.format(city, "") for t in templates]


if MAIN:
    for city_combination in combinations(primary_cities, 2):
        model.remove_all_hook_fns()
        for city1, city2 in [city_combination, list(city_combination)[::-1]]:
            print(f"--- Steering {city1} to {city2} ---")

            def hook_city2(act, hook):
                if act.size(1) <= 1:
                    return
                sae_acts = sae.encode(act)[:, 1:]
                sae_acts[:, :, city_features[primary_cities.index(city1)]] = 0
                sae_acts[:, :, city_features[primary_cities.index(city2)]] = sae_acts.max()
                recon = sae.decode(sae_acts)
                act[:, 1:] = recon[:]

            model.add_hook(sae.cfg.hook_name, hook_city2)
            city1_prompts = uncompleted_city_strings(city1, city_templates)
            for prompt in city1_prompts:
                print(
                    model.generate(
                        prompt,
                        prepend_bos=True,
                        max_new_tokens=2,
                        verbose=False,
                        temperature=0.0,
                    )
                )

            print("\n")


def get_common_sae_features(model, sae, inputs, batch_size=32):
    feature_counts = torch.zeros((sae.W_dec.size(0)), device=sae.W_dec.device)
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]

        _, cache = model.run_with_cache(batch, prepend_bos=True)
        cache = cache[sae.cfg.hook_name].detach()[:, 1:].flatten(end_dim=1)
        sae_activations = sae.encode(cache).detach()
        feature_counts += sae_activations.sum(0)

        del cache
        del sae_activations
        torch.cuda.empty_cache()

    return feature_counts


def get_city_sae_features(model, sae, city, templates):
    city_prompts = uncompleted_city_strings(city, templates)
    city_features = get_common_sae_features(model, sae, city_prompts)
    return city_features.topk(32)


def get_meta_sae_features(meta_sae, sae, topk):
    meta_activations = meta_sae.encode(sae.W_dec[topk.indices]).detach()
    meta_activations[meta_activations <= 0.11] = 0
    meta_activations *= topk.values.unsqueeze(1)
    sorted_meta_activations = meta_activations.sum(0).sort(descending=True)
    return sorted_meta_activations.indices[sorted_meta_activations.values > 0]


if MAIN:
    sae.W_dec.data = original_W_dec.clone().detach().to(torch.float16)
    model.remove_all_hook_fns()
    primary_city_features = [
        get_city_sae_features(model, sae, cities[city], city_templates)
        for city in primary_cities
    ]
    primary_city_meta_features = [
        get_meta_sae_features(meta_sae, sae, features).tolist()
        for features in primary_city_features
    ]

    secondary_city_features = [
        get_city_sae_features(model, sae, cities[city], city_templates)
        for city in secondary_cities
    ]
    secondary_city_meta_features = [
        get_meta_sae_features(meta_sae, sae, features).tolist()
        for features in secondary_city_features
    ]

    unique_city_meta_features = [set(f) for f in primary_city_meta_features]
    for i in range(len(primary_cities)):
        for j in range(len(primary_cities)):
            if i != j:
                unique_city_meta_features[i] -= set(primary_city_meta_features[j])
        # unique_city_meta_features[i] &= set(secondary_city_meta_features[i])
    sorted_unique_features = [
        [f for f in fs if f in uf]
        for fs, uf in zip(primary_city_meta_features, unique_city_meta_features)
    ]

# Sydney feature = 2834
# Paris feature = 6540


def generate_with_sae(model, sae, prompt):
    def hook_city2(act, hook):
        act[:, 1:] = sae(act[:, 1:])

    model.add_hook(sae.cfg.hook_name, hook_city2)
    print(
        model.generate(
            prompt,
            prepend_bos=True,
            max_new_tokens=2,
            verbose=False,
            temperature=0.0,
        )
    )


if MAIN:
    for initial_city in [0, 1]:
        model.remove_all_hook_fns()
        sae.W_dec.data = meta_sae(original_W_dec.clone().detach().to(torch.float16))['sae_out']

        n_unique_features = 1
        target_city = int(not initial_city)
        prompt = f"{primary_cities[initial_city]} is in the country of"

        meta_activations = meta_sae.encode(sae.W_dec[top_city_features[initial_city]]).detach()
        for f in sorted_unique_features[initial_city]:
            sae.W_dec[top_city_features[initial_city]] -= meta_activations[:, f].unsqueeze(
                1
            ) @ meta_sae.W_dec[f].unsqueeze(0)
            # sae.W_dec[top_city_features[initial_city]] -= meta_sae.W_dec[f]

        meta_activations = meta_sae.encode(sae.W_dec[top_city_features[target_city]]).detach()
        for f in sorted_unique_features[target_city]: 
            # sae.W_dec[top_city_features[initial_city]] += meta_activations[:, f].unsqueeze(
            #     1
            # ) @ meta_sae.W_dec[f].unsqueeze(0) 
            sae.W_dec[top_city_features[initial_city]] += meta_sae.W_dec[f] * 1.

        print('Norm:', sae.W_dec.norm(dim=1, keepdim=True).max().item())

        sae.W_dec.data /= sae.W_dec.norm(dim=1, keepdim=True)

        generate_with_sae(model, sae, prompt)
        generate_with_sae(model, sae, f'{primary_cities[initial_city]} is in the continent of')
