# %%
import json
import os
from dataclasses import dataclass
import itertools
from itertools import combinations
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text
from transformer_lens.HookedTransformer import HookedTransformer
from tqdm import tqdm

import wandb
from meta_saes.config import Config
from meta_saes.sae import (
    BatchTopKSAE,
    load_feature_splitting_saes,
    load_wandb_sae,
    load_gemma_sae,
)
from meta_saes.training import train_meta_sae

# %%

MAIN = __name__ == "__main__"


def _load_gpt2():
    model, saes, token_dataset = load_feature_splitting_saes(device=device)
    sae = saes[-2].eval()
    sae.W_dec.requires_grad = False
    original_W_dec = sae.W_dec.clone().detach().to(device)

    try:
        meta_sae, meta_cfg = load_wandb_sae(
            # "patrickaaleask/gpt2-sm-feature-splitting-meta-saes/gpt2-small_blocks.8.hook_resid_post_512_topk_4_0.0001_9001:v0",
            # "patrickaaleask/gpt2-sm-feature-splitting-meta-saes/gpt2-small_blocks.8.hook_resid_post_2048_topk_4_0.0001_10001:v0",
            # "patrickaaleask/gpt2-sm-feature-splitting-meta-saes/gpt2-small_blocks.8.hook_resid_post_8192_topk_4_0.0001_9001:v0",
            "mats-sprint/gpt2-feature-splitting-saes/gpt2-small_blocks.8.hook_resid_pre_2304_topk_4_0.001_2000:v0",
            BatchTopKSAE,
        )
    except wandb.errors.CommError as e:
        print("Could not load model from W&B, training from scratch")
        cfg = Config(
            hook_point=sae.cfg.hook_name,
            model_name=sae.cfg.model_name,
            layer=sae.cfg.hook_layer,
            site="resid_post",
            dict_size=512,
            act_size=768,
            top_k=4,
            lr=1e-4,
            l1_coeff=0,
            aux_penalty=1 / 16,
            input_unit_norm=False,
            wandb_project="gpt2-sm-feature-splitting-meta-saes",
            n_batches_to_dead=100,
            threshold=None,
            cosine_penalty=0.0000,
            epochs=10_000,
            batch_size=2**15,
        )

        meta_sae = BatchTopKSAE(cfg)
        train_meta_sae(meta_sae, sae)
    return model, saes, sae, original_W_dec, meta_sae


def _load_gemma():
    model, saes, token_dataset = load_gemma_sae(device=device)
    sae = saes[0].eval()
    sae.W_dec.requires_grad = False
    original_W_dec = sae.W_dec.clone().detach().to(device)

    try:
        meta_sae, meta_cfg = load_wandb_sae(
            "patrickaaleask/gemma-meta-saes/gemma-2-2b_blocks.3.hook_resid_post_2304_topk_4_0.0001_6001:v0",
            BatchTopKSAE,
        )
    except wandb.errors.CommError as e:
        print("Could not load model from W&B, training from scratch")
        cfg = Config(
            hook_point=sae.cfg.hook_name,
            model_name=sae.cfg.model_name,
            layer=sae.cfg.hook_layer,
            site="resid_post",
            dict_size=2304,
            act_size=2304,
            top_k=4,
            lr=1e-4,
            l1_coeff=0,
            aux_penalty=1 / 16,
            input_unit_norm=False,
            wandb_project="gemma-meta-saes",
            n_batches_to_dead=100,
            threshold=None,
            cosine_penalty=0.0000,
            epochs=10_000,
            batch_size=2**15,
        )

        meta_sae = BatchTopKSAE(cfg)
        train_meta_sae(meta_sae, sae)

    return model, saes, sae, original_W_dec, meta_sae


MODEL = "gemma"

if MAIN:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL == "gpt2":
        model, saes, sae, original_W_dec, meta_sae = _load_gpt2()
    else:
        model, saes, sae, original_W_dec, meta_sae = _load_gemma()
    dtype = original_W_dec.dtype

# %%
@dataclass
class CityAttributes:
    name: str
    country: str
    continent: str
    language: str


target_cities = ["Tokyo", "Paris"]


def load_ravel_city_attributes(path="/workspace/ravel/data"):
    cities = {}
    with open(os.path.join(path, "ravel_city_entity_attributes.json"), "r") as f:
        d = json.load(f)
    for city, attributes in d.items():
        cities[city] = CityAttributes(
            city, attributes["Country"], attributes["Continent"], attributes["Language"]
        )
    return cities


cities = load_ravel_city_attributes()


def print_token_probs(model, prompts, string_lists):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Prompt", style="cyan")
    table.add_column("String", style="green")
    table.add_column("Logprobs", style="yellow")

    for prompt, strings in zip(prompts, string_lists):
        tokens = model.tokenizer(strings, add_special_tokens=False)["input_ids"]
        tokens = [ts[:1] for ts in tokens]
        logprobs = F.log_softmax(model(prompt), dim=-1)[0, -1, tokens].squeeze()

        for i, (string, token, logit) in enumerate(zip(strings, tokens, logprobs)):
            logit_str = f"{logit.item():.3f}"

            if i == 0:
                table.add_row(
                    f"{prompt} {strings[logprobs.flatten().argmax().item()]}",
                    string,
                    logit_str,
                )
            else:
                table.add_row("", string, logit_str)

    console.print(table)


def print_city_completions(model, target_cities: List[str]):
    countries = [cities[city].country for city in target_cities] + ["Egypt"]
    country_prompt = f"{target_cities[0]} is a city in the country of"

    languages = [cities[city].language for city in target_cities] + ["Arabic"]
    language_prompt = f"The primary language spoken in {target_cities[0]} is"

    continents = [cities[city].continent for city in target_cities] + ["Africa"]
    continent_prompt = f"{target_cities[0]} is a city on the continent of"

    currencies = ["Yen", "Euro", "Pound"]
    currency_prompt = f"The currency used in {target_cities[0]} is the"

    print_token_probs(
        model,
        [
            country_prompt,
            language_prompt,
            continent_prompt,
            currency_prompt,
        ],
        [countries, languages, continents, currencies],
    )


# %% [markdown]
# ## Steering city attributes
# GPT2-Small doesn't do a great job of answering questions about city
# attributes, but inspecting the logits it is confident about completion
# probabilities.

# %%


if MAIN:
    model.remove_all_hook_fns()
    print_city_completions(model, target_cities)


# %% [markdown]
# ### Steering using SAE features
# In this section we try to steer Tokyo to be a city in France instead of in Japan.
# There are separate SAE features for Tokyo and Paris, so the best we can do
# with SAEs is replacing Tokyo with Paris entirely.

# %%


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


city_templates = [
    "{} is a city in{}",
    "{} is a city on the continent of{}",
    "The primary language spoken in {} is{}",
    "The currency used in {} is the{}",
]

if MAIN:
    model.remove_all_hook_fns()
    sae.W_dec.data = original_W_dec.clone().detach().to(dtype)

    city_counts = [
        get_common_sae_features(
            model, sae, completed_city_strings(cities[city], city_templates)
        )
        for city in target_cities
    ]
    feature_ratios = [
        city_counts[i] / (sum(city_counts) + 1) for i in range(len(target_cities))
    ]
    city_features = torch.cat([fr.topk(1).indices for fr in feature_ratios], dim=0)

    def hook_city(act, hook):
        if act.size(1) <= 1:
            return
        sae_acts = sae.encode(act)[:, 1:]
        sae_acts[:, :, city_features[0]] = 0
        sae_acts[:, :, city_features[1]] = sae_acts.max()
        recon = sae.decode(sae_acts)
        act[:, 1:] = recon[:]

    model.add_hook(sae.cfg.hook_name, hook_city)
    print_city_completions(model, target_cities)

# %% [markdown]
# ### Steering using Meta SAE features
# However, we can decompose those features using a meta SAE which lets us steer
# the geographic location of Tokyo without also steering its language or
# currency.

# %%
from itertools import product


def get_city_completions(model, target_cities: List[str]):
    countries = [cities[city].country for city in target_cities] + ["Egypt"]
    country_prompt = f"{target_cities[0]} is a city in the country of"

    languages = [cities[city].language for city in target_cities] + ["Arabic"]
    language_prompt = f"The primary language spoken in {target_cities[0]} is"

    continents = [cities[city].continent for city in target_cities] + ["Africa"]
    continent_prompt = f"{target_cities[0]} is a city on the continent of"

    currencies = ["Yen", "Euro", "Pound"]
    currency_prompt = f"The currency used in {target_cities[0]} is the"

    city_completions = []
    for prompt, strings in zip(
        [country_prompt, language_prompt, continent_prompt, currency_prompt],
        [countries, languages, continents, currencies],
    ):
        tokens = model.tokenizer(strings, add_special_tokens=False)["input_ids"]
        tokens = [ts[:1] for ts in tokens]
        logprobs = F.log_softmax(model(prompt), dim=-1)[0, -1, tokens].squeeze()
        city_completions.append(logprobs.argmax().item())
    return city_completions


def all_combinations(list1, list2):
    result = []

    # Generate all possible lengths for combinations from each list
    for len1 in range(len(list1) + 1):
        for len2 in range(len(list2) + 1):
            combs1 = itertools.combinations(list1, len1)
            combs2 = itertools.combinations(list2, len2)
            for c1 in combs1:
                for c2 in combs2:
                    result.append((tuple(c1), tuple(c2)))

    return result


# TODO: Replace this with a clearer display
if MAIN:
    model.remove_all_hook_fns()
    sae.W_dec.data = original_W_dec.clone().detach().to(dtype)

    meta_activations = meta_sae(sae.W_dec[city_features], threshold=0.14)[
        "feature_acts"
    ].detach()

    start_city_meta_features = meta_activations[0].nonzero().squeeze().tolist()[:-1]
    target_city_meta_features = meta_activations[1].nonzero().squeeze().tolist()[:-1]

    all_city_completions = []

    combinations = all_combinations(start_city_meta_features, target_city_meta_features)

    print_completion = True
    for start_fs, target_fs in tqdm(combinations, desc="Steering city attributes"):
        model.remove_all_hook_fns()
        sae.W_dec.data = original_W_dec.clone().detach().to(dtype)

        # Remove the original meta features from the city
        for f in start_fs:
            sae.W_dec[city_features[0]] -= meta_sae.W_dec[f] * 100 #meta_activations[0, f]

        # Add the meta features from the target city
        for f in target_fs:
            sae.W_dec[city_features[0]] += meta_sae.W_dec[f] * 100 #meta_activations[1, f]

        sae.W_dec.data /= sae.W_dec.data.norm(dim=1, keepdim=True)

        def hook_city(act, hook):
            if act.size(1) <= 1:
                return
            sae_acts = sae.encode(act)[:, 1:]
            recon = sae.decode(sae_acts)
            act[:, 1:] = recon[:]

        model.add_hook(sae.cfg.hook_name, hook_city)
        all_city_completions.append(tuple(get_city_completions(model, target_cities)))

        if (all_city_completions[-1] == tuple([0, 1, 0, 1])) and print_completion:
            print_city_completions(model, target_cities)
            print_completion = False

    unique_city_completions = list(set(all_city_completions))
    unique_city_completions = sorted(
        unique_city_completions, key=lambda x: sum(x), reverse=False
    )

    columns = [
        "Country",
        "Language",
        "Continent",
        "Currency",
        "Start city meta features removed",
        "Target city meta features added",
    ]
    console = Console()
    table = Table(
        title="Which attributes can be edited from Tokyo to Paris?", box=box.ROUNDED
    )

    for column in columns:
        table.add_column(column, style="cyan", justify="center")

    for row in unique_city_completions:
        for i in range(len(combinations)):
            if row == all_city_completions[i]:
                break

        table.add_row(
            *[
                Text("✓", style="green") if value else Text("✗", style="red")
                for value in row
            ]
            + [f"{combinations[i][0]}", f"{combinations[i][1]}"]
        )

    console.print(table)
