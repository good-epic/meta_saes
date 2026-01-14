# %%
import gc
import json
import os
import urllib
from dataclasses import dataclass
from itertools import combinations

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.table import Table
from tqdm.notebook import tqdm

from meta_saes.sae import (
    BatchTopKSAE,
    load_feature_splitting_saes,
    load_gemma_sae,
    load_wandb_sae,
)


def clear():
    gc.collect()
    torch.cuda.empty_cache()


# %%

MAIN = __name__ == "__main__"

if MAIN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.float32
    torch.set_grad_enabled(False)

    layer = 5
    # model, saes, token_dataset = load_gemma_sae(
    #     sae_id=f"layer_{layer}/width_1m/canonical",
    #     # sae_id=f"layer_{layer}/width_16k/canonical",
    #     release="gemma-scope-2b-pt-res-canonical",
    #     # release="gemma-scope-2b-pt-res",
    #     # sae_id=f"layer_{layer}/width_65k/canonical",
    #     # sae_id=f"layer_{layer}/width_16k/canonical",
    #     device=device,
    # )
    # meta_sae, cfg = load_wandb_sae(
    #     # "mats-sprint/gemma-2b-layer-25-canonical-tiny-meta-saes/gemma-2-2b_blocks.25.hook_resid_post_2304_topk_4_0.001_2000:v0",
    #     "patrickaaleask/gemma-scope-2b-pt-res-meta-saes/gemma-2-2b_blocks.5.hook_resid_post_2304_topk_4_0.0001_192032:v0",
    #     BatchTopKSAE,
    # )
    model, saes, token_dataset = load_feature_splitting_saes(device=device)
    meta_sae, cfg = load_wandb_sae(
        'mats-sprint/feature-splitting-saes-combined-W-dec/gpt2-small_blocks.8.hook_combined_W_dec_2304_topk_4_0.001_25000:v0',
        BatchTopKSAE,
    )
    sae = saes[-1].to(device).eval().to(dtype)
    meta_sae = meta_sae.to(device).eval().to(dtype)
    original_W_dec = sae.W_dec.clone().detach().to(dtype)
    sae.W_dec.requires_grad = False

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
    "{} is a city in{}",
    "{} is a city on the continent of{}",
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
    sae.W_dec.data = original_W_dec.clone().detach().to(dtype)
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
                sae_acts[:, :, city_features[primary_cities.index(city2)]] = (
                    sae_acts.max()
                )
                recon = sae.decode(sae_acts)
                act[:, 1:] = recon[:]

            model.add_hook(sae.cfg.hook_name, hook_city2)
            city1_prompts = uncompleted_city_strings(city1, city_templates)
            for coutry_prompt in city1_prompts:
                print(
                    model.generate(
                        coutry_prompt,
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
    return city_features.topk(2048)


def get_meta_sae_features(meta_sae, sae, topk):
    meta_activations = meta_sae.encode(sae.W_dec[topk.indices]).detach()
    meta_activations[meta_activations <= 0.11] = 0
    meta_activations *= topk.values.unsqueeze(1)
    sorted_meta_activations = meta_activations.sum(0).sort(descending=True)
    return sorted_meta_activations.indices[sorted_meta_activations.values > 0]


if MAIN:
    sae.W_dec.data = original_W_dec.clone().detach().to(dtype)
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


def print_token_probs(model, prompt, strings):
    tokens = model.tokenizer(strings, add_special_tokens=False)["input_ids"]
    logits = model(prompt)[0, -1, tokens]
    probs = torch.softmax(logits, 0)

    print('Generated:', model.generate(prompt, prepend_bos=True, max_new_tokens=10, verbose=False, temperature=0))
    print('Max logit:', prompt, strings[probs.argmax()])
    console = Console()
    table = Table()
    table.add_column("Token", style="cyan", no_wrap=True)
    table.add_column("Probability")
    for i, token in enumerate(tokens):
        token_string = model.tokenizer.decode(token)
        probability = f"{probs[i, 0].item():.3f}"
        table.add_row(token_string, probability)
    console.print(table)


if MAIN:
    for initial_city in [0, 1]:
        model.remove_all_hook_fns()
        sae.W_dec.data = original_W_dec.clone().detach().to(dtype)

        which_unique_features = slice(0, 1)
        target_city = int(not initial_city)
        coutry_prompt = f"{primary_cities[initial_city]} is a city in the country of"

        meta_activations = meta_sae.encode(
            sae.W_dec[top_city_features[initial_city]]
        ).detach()
        # for f in sorted_unique_features[initial_city][which_unique_features]:
            # sae.W_dec[top_city_features[initial_city]] -= meta_sae.W_dec[f]

        meta_activations = meta_sae.encode(
            sae.W_dec[top_city_features[target_city]]
        ).detach()
        # for f in sorted_unique_features[target_city][which_unique_features]:
            # sae.W_dec[top_city_features[initial_city]] += meta_sae.W_dec[f] * 1.0

        sae.W_dec.data /= sae.W_dec.norm(dim=1, keepdim=True)

        def hook_sae(act, hook):
            act[:, 1:] = sae(act[:, 1:])

        model.add_hook(sae.cfg.hook_name, hook_sae)

        countries = [cities[city].country for city in primary_cities]
        print_token_probs(model, coutry_prompt, countries)

        continent_prompt = f"{primary_cities[initial_city]} is a city on the continent of"
        continents = [cities[city].continent for city in primary_cities]
        print_token_probs(model, continent_prompt, continents)

        continent_prompt = (
            f"The main language spoken in {primary_cities[initial_city]} is"
        )
        continents = [cities[city].language for city in primary_cities]
        print_token_probs(model, continent_prompt, continents)
