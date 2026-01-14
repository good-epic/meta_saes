import json
import os
import urllib
from dataclasses import dataclass
import itertools
from itertools import combinations
import random
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
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


def clear():
    """
    Force garbage collection and empty the cache.
    """
    import gc

    gc.collect()
    torch.cuda.empty_cache()


NEURONPEDIA_DOMAIN = "https://neuronpedia.org"


def quicklist_gemmascope(sae, features, name="list"):
    """
    Wrapper for neuronpedia quick list for gpt2 feature splitting SAEs.
    """
    url = NEURONPEDIA_DOMAIN + "/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    layer = 3
    dataset = "gemmascope-res-16k"
    list_feature = [
        {
            "modelId": "gemma-2-2b",
            "layer": f"{layer}-{dataset}",
            "index": str(feature),
        }
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))

    return url


# %%

MAIN = __name__ == "__main__"

THRESHOLD = 0.096


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
    return model, sae, original_W_dec, meta_sae


def _load_gemma():
    model, saes, token_dataset = load_gemma_sae(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_3/width_16k/average_l0_59",
        dataset="NeelNanda/c4-10k",
        device="cuda",
    )
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

    return model, sae, original_W_dec, meta_sae


MODEL = "gemma"

if MAIN:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "model" not in locals():  # avoid re-loading model when hitting run all
        if MODEL == "gpt2":
            model, sae, original_W_dec, meta_sae = _load_gpt2()
        else:
            model, sae, original_W_dec, meta_sae = _load_gemma()
        model = model.eval()
        sae = sae.eval()
        meta_sae = meta_sae.eval()
        dtype = original_W_dec.dtype

    model.remove_all_hook_fns()

    def hook_fn(act, hook):
        with torch.no_grad():
            act[1:] = sae(act[1:])
        return act

    model.add_hook(sae.cfg.hook_name, hook_fn)
    print(model.generate("The capital of Venezula is", verbose=False))

# %%


def generate(model, prompts, batch_size=128):
    generations = []
    for batch in tqdm(
        [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    ):
        tokens = model.to_tokens(batch, padding_side="left")
        generated_tokens = model.generate(tokens, verbose=False, temperature=0)
        outputs = model.to_string(generated_tokens)
        outputs = [o.replace("<bos>", "").replace("<pad>", "") for o in outputs]
        generations.extend(outputs)
    return generations


if MAIN:
    DATA_DIR = "/workspace/ravel/data"
    model_name = sae.cfg.model_name
    entity_type = "city"

    prompt_to_output_path = os.path.join(
        DATA_DIR, model_name, f"ravel_{model_name}_{entity_type}_prompt_to_output.csv"
    )

    attribute_templates_json = json.load(
        open(
            os.path.join(
                DATA_DIR, "base", f"ravel_{entity_type}_attribute_to_prompts.json"
            )
        )
    )
    attribute_templates_json = {
        k: v
        for k, v in attribute_templates_json.items()
        # Just use country for now as it's easier to work on
        if k in ["Country", "Continent"]
    }
    attribute_templates = []
    for attribute, templates in attribute_templates_json.items():
        for template in templates:
            attribute_templates.append({"attribute": attribute, "template": template})
    attribute_templates = pd.DataFrame(attribute_templates)

    template_splits = json.load(
        open(
            os.path.join(DATA_DIR, "base", f"ravel_{entity_type}_prompt_to_split.json")
        )
    )
    splits = [template_splits[template] for template in attribute_templates["template"]]
    attribute_templates["splits"] = splits

    entity_attributes_json = json.load(
        open(
            os.path.join(
                DATA_DIR, "base", f"ravel_{entity_type}_entity_attributes.json"
            )
        )
    )
    entity_rows = []
    for entity, attributes in entity_attributes_json.items():
        for attribute, value in attributes.items():
            entity_rows.append(
                {"entity": entity, "attribute": attribute, "value": value}
            )
    entity_attributes = pd.DataFrame(entity_rows)
    entity_attributes = entity_attributes[
        entity_attributes["attribute"].isin(["Country", "Continent"])
    ]
    entities = entity_attributes["entity"].sample(n=10, random_state=0)
    entities = ["London", "Tokyo", "Rio de Janeiro", "New York", "Cape Town"]

    entity_attributes = entity_attributes[entity_attributes["entity"].isin(entities)]

    prompt_data = attribute_templates.merge(entity_attributes, on="attribute")
    prompts = []
    for template, entity in prompt_data[["template", "entity"]].itertuples(
        index=False, name=None
    ):
        prompts.append(template % entity)
    prompt_data["prompt"] = prompts

    if os.path.exists(prompt_to_output_path):
        os.remove(prompt_to_output_path)

    try:
        prompt_data = pd.read_csv(prompt_to_output_path)
    except FileNotFoundError:
        clear()
        completions = generate(model, prompt_data["prompt"].tolist(), batch_size=128)
        prompt_data["completion"] = completions
        prompt_data.to_csv(prompt_to_output_path)

    success = []
    for value, completion in prompt_data[["value", "completion"]].itertuples(
        index=False, name=None
    ):
        success.append(value in completion)
    prompt_data["success"] = success
    print("Success ratio:", prompt_data["success"].mean())

    template_successes = prompt_data.groupby("template")["success"].mean()
    attribute_templates = attribute_templates.merge(
        template_successes, left_on="template", right_index=True
    )

    attribute_successes = prompt_data.groupby(["attribute", "entity"])["success"].mean()
    entity_attributes = entity_attributes.merge(
        attribute_successes, left_on=["attribute", "entity"], right_index=True
    )

    entity_to_split_json = json.load(
        open(
            os.path.join(DATA_DIR, "base", f"ravel_{entity_type}_entity_to_split.json")
        )
    )
    entity_to_split = pd.DataFrame(
        [{"entity": k, "split": v} for k, v in entity_to_split_json.items()]
    )
    entity_attributes = entity_attributes.merge(entity_to_split, on="entity")

    # TODO: Filter for the attributes we want to keep when we increase number of
    # examples

    wiki_prompts_json = json.load(
        open(
            os.path.join(
                DATA_DIR, "base", f"wikipedia_{entity_type}_entity_prompts.json"
            )
        )
    )
    wiki_prompts = pd.DataFrame(
        [
            {"prompt": k % v["entity"], "entity": v["entity"], "split": v["split"]}
            for k, v in wiki_prompts_json.items()
        ]
    )
    entity_attributes = entity_attributes.merge(
        wiki_prompts.drop_duplicates(subset=["entity"], keep="first"), on="entity"
    )

# %%


class LassoRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, device="cpu"):
        super(LassoRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, device=device)

    def forward(self, x):
        return self.linear(x)


def train_lasso_regressor(X, y, num_epochs=100, batch_size=32, lr=0.01, l1_lambda=0.01):
    label_to_index = {label: idx for idx, label in enumerate(set(y))}
    numerical_labels = torch.tensor(
        [label_to_index[label] for label in y], device=device
    )

    dataset = TensorDataset(X, numerical_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    num_classes = len(label_to_index)
    model = LassoRegressor(input_dim, num_classes, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(range(num_epochs), desc="Training")
    for _ in pbar:
        model.train()
        epoch_loss = 0.0
        for inputs, labels in loader:
            outputs = model(inputs.to(device))

            loss = criterion(outputs, labels.to(device))

            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        pbar.set_description(f"Training (loss: {avg_loss:.4f})")

    return model


def get_or_train_lasso(prompt_data, entity, sae, epochs=100, device="cpu"):
    lasso_file = f"{entity}_lasso.pt"
    # if os.path.exists(lasso_file):
    # os.remove(lasso_file)
    try:
        lasso = LassoRegressor(
            sae.W_dec.size(0), prompt_data["value"].nunique(), device=device
        )
        lasso.load_state_dict(torch.load(lasso_file))
    except FileNotFoundError:
        batch_size = 128
        sae_acts = []
        labels = []
        for batch in tqdm(
            [
                prompt_data[["prompt", "value"]][i : i + batch_size]
                for i in range(0, len(prompt_data["prompt"]), batch_size)
            ]
        ):
            with torch.no_grad():
                tokens = model.to_tokens(batch["prompt"].tolist(), padding_side="left")
                _, cache = model.run_with_cache(tokens)
                acts = cache[sae.cfg.hook_name]
                sa = sae.encode(acts).flatten(end_dim=1)
                sae_acts.append(sa)
            for i in range(acts.size(0)):
                for _ in range(acts.size(1)):
                    labels.append(batch["value"].iloc[i])
        sae_acts = torch.cat(sae_acts, dim=0)
        lasso = train_lasso_regressor(
            sae_acts, labels, lr=0.0001, batch_size=2**12, num_epochs=epochs
        )
        torch.save(lasso.state_dict(), lasso_file)
    return lasso


def hook_sae_steering(model, features, target_prompt):
    model.remove_all_hook_fns()
    _, cache = model.run_with_cache(model.to_tokens(target_prompt, padding_side="left"))
    cache = cache[sae.cfg.hook_name]
    wiki_acts = sae.encode(cache)

    def hook_fn(act, hook):
        sae_acts = sae.encode(act[0, 1:])
        sae_acts[1:, features] = wiki_acts[0, 1:, features].max(0).values
        act[:, 1:] = sae.decode(sae_acts.unsqueeze(0))
        return act

    model.add_hook(sae.cfg.hook_name, hook_fn)


def get_or_train_meta_sae_lasso(
    prompt_data, entity, sae, sae_features, meta_sae, epochs=100, device="cpu"
):
    lasso_file = f"{entity}_meta_sae_lasso.pt"
    # if os.path.exists(lasso_file):
    # os.remove(lasso_file)
    try:
        lasso = LassoRegressor(
            meta_sae.W_dec.size(0), prompt_data["value"].nunique(), device=device
        )
        lasso.load_state_dict(torch.load(lasso_file))
    except FileNotFoundError:
        meta_activations = meta_sae.encode(sae.W_dec[sae_features], threshold=THRESHOLD)
        batch_size = 128
        msae_acts = []
        labels = []
        for batch in tqdm(
            [
                prompt_data[["prompt", "value"]][i : i + batch_size]
                for i in range(0, len(prompt_data["prompt"]), batch_size)
            ]
        ):
            with torch.no_grad():
                tokens = model.to_tokens(batch["prompt"].tolist(), padding_side="left")
                _, cache = model.run_with_cache(tokens)
                acts = cache[sae.cfg.hook_name]
                sa = sae.encode(acts).flatten(end_dim=1)[:, sae_features]
                msae_acts.append(
                    (sa.unsqueeze(-1) * meta_activations.unsqueeze(0)).flatten(
                        end_dim=1
                    )
                )
                for i in range(acts.size(0)):
                    for _ in range(acts.size(1)):
                        for _ in range(sae_features.nonzero().size(0)):
                            labels.append(batch["value"].iloc[i])
            msae_acts = torch.cat(msae_acts, dim=0)
        lasso = train_lasso_regressor(
            msae_acts, labels, lr=0.0001, batch_size=2**12, num_epochs=epochs
        )
        torch.save(lasso.state_dict(), lasso_file)
    return lasso


def hook_sae(sae, sae_features, meta_sae, meta_features, start_prompt, target_prompt):
    _, target_cache = model.run_with_cache(
        model.to_tokens(target_prompt, padding_side="left")
    )
    target_cache = target_cache[sae.cfg.hook_name]
    target_acts = sae.encode(target_cache)[0]
    target_acts[:, ~sae_features] = 0
    target_features = target_acts.sum(0) > 0
    target_meta_acts = meta_sae.encode(sae.W_dec, threshold=THRESHOLD)

    _, start_cache = model.run_with_cache(
        model.to_tokens(start_prompt, padding_side="left")
    )
    start_cache = start_cache[sae.cfg.hook_name]
    start_acts = sae.encode(start_cache)[0]
    start_acts[:, ~sae_features] = 0
    start_features = start_acts.sum(0) > 0
    start_meta_acts = meta_sae.encode(sae.W_dec, threshold=THRESHOLD)

    for sf in start_features.nonzero().flatten():
        for mfi, mf in enumerate(meta_features.nonzero().flatten().tolist()):
            start_meta_acts[sf, mf] = target_meta_acts[target_features][:, meta_features].max(
                0
            ).values[mfi]

    recon = meta_sae.decode(start_meta_acts, sae.W_dec)["sae_out"]
    sae.W_dec.data = recon


if MAIN:
    clear()
    print("Training lasso for country")
    country_lasso = get_or_train_lasso(
        prompt_data[prompt_data["attribute"] == "Country"], "Country", sae, epochs=1000
    )
    country_features = (country_lasso.linear.weight.abs() > 0.01).sum(0) > 0
    print("Country features:", country_features.sum().item())

    print("Training meta feature lasso for country")
    country_meta_lasso = get_or_train_meta_sae_lasso(
        prompt_data[prompt_data["attribute"] == "Country"],
        "Country",
        sae,
        country_features,
        meta_sae,
        epochs=1000,
    )
    country_meta_features = (country_meta_lasso.linear.weight.abs() > 0.0001).sum(0) > 0
    print("Country meta features:", country_meta_features.sum().item())


# TODO: Use train splits when doing for real!
if MAIN:
    import warnings

    warnings.filterwarnings("ignore")

    # train_df = prompt_data[prompt_data["splits"] == "train"]
    # train_df = train_df.merge(wiki_prompts, on="entity")
    # attributes_all_success = train_df.groupby("entity")["success"].mean() > 0.8
    # successful_attributes = attributes_all_success[attributes_all_success]
    # start_city, target_city = successful_attributes.sample(
    #     n=2, random_state=0
    # ).index.tolist()
    start_city = "Tokyo"
    target_city = "London"

    print(f"Start city: {start_city}, target city: {target_city}")

    country_prompt = (
        prompt_data[prompt_data["attribute"] == "Country"][
            prompt_data["entity"] == start_city
        ][prompt_data["success"] == True]
        .groupby("value")
        .sample(n=1, random_state=0)["prompt"]
        .tolist()[0]
    )
    print(country_prompt)
    continent_prompt = (
        prompt_data[prompt_data["attribute"] == "Continent"][
            prompt_data["entity"] == start_city
        ][prompt_data["success"] == True]
        .groupby("value")
        .sample(n=1, random_state=0)["prompt"]
        .tolist()[0]
    )
    print(continent_prompt)

    wiki_prompt = wiki_prompts[wiki_prompts["entity"] == target_city][
        "prompt"
    ].tolist()[0]
    print(wiki_prompt)

    model.remove_all_hook_fns()

    def hook_fn(act, hook):
        act[1:] = sae(act[1:])
        return act

    model.add_hook(sae.cfg.hook_name, hook_fn)
    print("Without steering")
    print(
        "Country prompt:",
        model.generate(country_prompt, verbose=False, max_new_tokens=2, temperature=0),
    )
    print(
        "Continent prompt:",
        model.generate(
            continent_prompt, verbose=False, max_new_tokens=2, temperature=0
        ),
    )

    model.remove_all_hook_fns()
    hook_sae_steering(model, country_features, wiki_prompt)
    print("With steering")
    print(
        "Country prompt:",
        model.generate(country_prompt, verbose=False, max_new_tokens=2, temperature=0),
    )
    print(
        "Continent prompt:",
        model.generate(
            continent_prompt, verbose=False, max_new_tokens=2, temperature=0
        ),
    )

    sae.remove_all_hook_fns()
    model.remove_all_hook_fns()
    hook_sae(
        sae,
        country_features,
        meta_sae,
        country_meta_features,
        country_prompt,
        wiki_prompt,
    )
    print("With steering")
    print(
        "Country prompt:",
        model.generate(country_prompt, verbose=False, max_new_tokens=2, temperature=0),
    )
    print(
        "Continent prompt:",
        model.generate(
            continent_prompt, verbose=False, max_new_tokens=2, temperature=0
        ),
    )
    sae.W_dec.data = original_W_dec

# %%
