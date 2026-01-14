import json
import os
import pickle
from typing import List

import sys
sys.path.append("/workspace/Gemma2/SAELens")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sae_lens import SAE
from sae_lens.load_model import load_model
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from huggingface_hub import hf_hub_download


import wandb
from meta_saes.config import Config

SAMPLE_SAES = {
    "gpt2": "mats-sprint/meta_sae/gpt2-small_blocks.8.hook_resid_pre_24576_topk_8_0.001_240000:v0",
    "gemma-2-9b": "mats-sprint/meta_sae/gemma-9b_blocks.8.hook_resid_pre_24576_topk_8_0.001_10000:v0",
}


class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(self.cfg.seed)

        self.b_dec = nn.Parameter(torch.zeros(self.cfg.act_size))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg.dict_size))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg.act_size, self.cfg.dict_size)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg.dict_size, self.cfg.act_size)
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.cfg.dict_size,)).to(cfg.device)

        self.to(cfg.dtype).to(cfg.device)

    def preprocess_input(self, x):
        if self.cfg.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.input_unit_norm:
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        try:
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
                -1, keepdim=True
            ) * W_dec_normed
            self.W_dec.grad -= W_dec_grad_proj
        except TypeError:
            pass
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    @torch.no_grad()
    def fold_W_dec_norm(self):
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T
        self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()


class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def encode(self, x, threshold=None):
        x, _, _ = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        if threshold is not None:
            acts_topk = (acts > threshold).float().to(acts.dtype) * acts
        else:
            acts_topk = torch.topk(acts.flatten(), self.cfg.top_k * x.shape[0], dim=-1)
            acts_topk = (
                torch.zeros_like(acts.flatten(), dtype=acts.dtype)
                .scatter(-1, acts_topk.indices, acts_topk.values)
                .reshape(acts.shape)
            )
        return acts_topk

    def decode(self, acts, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_reconstruct = acts @ self.W_dec + self.b_dec

        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts, x_mean, x_std)
        return output

    def forward(self, x, threshold=None):
        acts = self.encode(x, threshold=threshold)
        return self.decode(acts, x)

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss, mean_similarity, max_similarity = self.get_auxiliary_loss(
            x, x_reconstruct, acts
        )
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg.n_batches_to_dead
        ).sum()
        threshold = (
            acts_topk[acts_topk > 0].min()
            if acts_topk.sum() > 0
            else torch.tensor(0, dtype=x.dtype, device=x.device)
        )
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "threshold": threshold,
            "mean_similarity": mean_similarity,
            "max_similarity": max_similarity,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        loss = torch.tensor(0, dtype=x.dtype, device=x.device)
        mean_similarity = torch.tensor(0, dtype=x.dtype, device=x.device)
        max_similarity = torch.tensor(0, dtype=x.dtype, device=x.device)
        dead_features = self.num_batches_not_active >= self.cfg.n_batches_to_dead
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg.top_k_aux, dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            loss += (
                self.cfg.aux_penalty
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )

        cosine_penalty = self.cfg.cosine_penalty
        if cosine_penalty > 0:
            batch_size, dict_size = acts.shape
            _, top_k_indices = torch.topk(acts, k=self.cfg.top_k, dim=1)
            batch_indices = (
                torch.arange(batch_size, device=acts.device)
                .unsqueeze(1)
                .expand(-1, self.cfg.top_k)
            )
            top_k_decoders = self.W_dec[
                top_k_indices[
                    batch_indices, torch.arange(self.cfg.top_k, device=acts.device)
                ]
            ]
            cosine_similarities = torch.bmm(
                top_k_decoders, top_k_decoders.transpose(1, 2)
            )
            mask = 1 - torch.eye(
                self.cfg.top_k, device=cosine_similarities.device
            ).unsqueeze(0)
            masked_similarities = torch.abs(cosine_similarities * mask)
            mean_similarity = masked_similarities.sum() / (
                batch_size * self.cfg.top_k * (self.cfg.top_k - 1)
            )
            max_similarity = masked_similarities.max(dim=-1).values.mean()
            loss += cosine_penalty * max_similarity

        return loss, mean_similarity, max_similarity


class TopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x, threshold=None):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        if threshold is not None:
            acts_topk = (acts > threshold).float() * acts
        else:
            acts_topk = torch.topk(acts, self.cfg.top_k, dim=-1)
            acts_topk = torch.zeros_like(acts).scatter(
                -1, acts_topk.indices, acts_topk.values
            )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss, mean_similarity = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg.n_batches_to_dead
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "mean_similarity": mean_similarity,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        loss = torch.tensor(0, dtype=x.dtype, device=x.device)
        mean_similarity = torch.tensor(0, dtype=x.dtype, device=x.device)
        dead_features = self.num_batches_not_active >= self.cfg.n_batches_to_dead
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg.top_k_aux, dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            loss += (
                self.cfg.aux_penalty
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )

        cosine_penalty = self.cfg.cosine_penalty
        if cosine_penalty > 0:
            batch_size, dict_size = acts.shape
            _, top_k_indices = torch.topk(acts, k=self.cfg.top_k, dim=1)
            batch_indices = (
                torch.arange(batch_size, device=acts.device)
                .unsqueeze(1)
                .expand(-1, self.cfg.top_k)
            )
            top_k_decoders = self.W_dec[
                top_k_indices[
                    batch_indices, torch.arange(self.cfg.top_k, device=acts.device)
                ]
            ]
            cosine_similarities = torch.bmm(
                top_k_decoders, top_k_decoders.transpose(1, 2)
            )
            mask = 1 - torch.eye(
                self.cfg.top_k, device=cosine_similarities.device
            ).unsqueeze(0)
            masked_similarities = torch.abs(cosine_similarities * mask)
            mean_similarity = masked_similarities.sum() / (
                batch_size * self.cfg.top_k * (self.cfg.top_k - 1)
            )
            loss += cosine_penalty * mean_similarity

        return loss, mean_similarity


class VanillaSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg.l1_coeff * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg.n_batches_to_dead
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
        }
        return output


# TODO: This takes frustratingly long to run.
def load_feature_splitting_saes(device="cpu", saes_idxs=list(range(1, 9))):
    saes = []
    api = wandb.Api()

    class BackwardsCompatibleUnpickler(pickle.Unpickler):
        """
        An Unpickler that can load files saved before the "sae_lens" package namechange
        """

        def find_class(self, module: str, name: str):
            if name == "LanguageModelSAERunnerConfig":
                return super().find_class("sae_lens.config", name)
            return super().find_class(module, name)

    class BackwardsCompatiblePickleClass:
        Unpickler = BackwardsCompatibleUnpickler

    for i in saes_idxs:
        wandb_link = f"jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small_blocks.8.hook_resid_pre_{768 * 2**(i-1)}:v9"

        artifact = api.artifact(
            wandb_link,
            type="model",
        )
        artifact_dir = artifact.download()
        file = os.listdir(artifact_dir)[0]

        state_dict = torch.load(
            os.path.join(artifact_dir, file),
            pickle_module=BackwardsCompatiblePickleClass,
        )
        state_dict["cfg"].activation_fn_kwargs = None
        state_dict["cfg"].model_kwargs = None
        state_dict["cfg"].model_from_pretrained_kwargs = None
        state_dict["cfg"].sae_lens_version = None
        state_dict["cfg"].sae_lens_training_version = None
        state_dict["cfg"].activation_fn_str = "relu"
        state_dict["cfg"].dtype = "torch.float32"
        state_dict["cfg"].finetuning_scaling_factor = 1.0
        state_dict["cfg"].hook_name = "blocks.8.hook_resid_pre"
        state_dict["cfg"].hook_layer = 8
        instance = SAE(cfg=state_dict["cfg"])
        instance.finetuning_scaling_factor = nn.Parameter(torch.tensor(1.0))
        state_dict["state_dict"]["finetuning_scaling_factor"] = nn.Parameter(
            torch.tensor(1.0)
        )
        instance.load_state_dict(state_dict["state_dict"], strict=True)
        instance.to(device)

        saes.append(instance)

    model = load_model("HookedTransformer", "gpt2-small", device=device)

    dataset = load_dataset(
        path="NeelNanda/pile-10k",
        split="train",
        streaming=False,
    )
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=False,
        add_bos_token=saes[0].cfg.prepend_bos,
    )

    return model, saes, token_dataset


def load_wandb_sae(artifact_name, sae_class):
    api = wandb.Api()

    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()

    config_path = os.path.join(artifact_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    if "dtype" in cfg:
        cfg["dtype"] = getattr(torch, cfg["dtype"].split(".")[-1])

    cfg = Config(**cfg)

    sae = sae_class(cfg)

    state_dict_path = os.path.join(artifact_dir, "sae.pt")
    state_dict = torch.load(state_dict_path, map_location=cfg.device)
    sae.load_state_dict(state_dict)

    return sae, cfg


def load_gemma_sae(release="gemma-scope-2b-pt-res-canonical", 
                   sae_id="layer_3/width_16k/canonical", 
                   dataset="NeelNanda/c4-10k", device="cuda"):

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()

    model = HookedTransformer.from_pretrained_no_processing(
        cfg_dict["model_name"], device=device
    )
    model.eval()

    dataset = load_dataset(
        path=dataset,
        split="train",
        streaming=False,
    )

    token_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=model.tokenizer, 
        streaming=False,
        max_length=sae.cfg.context_size,
        add_bos_token=sae.cfg.prepend_bos,
    )

    return model, [sae], token_dataset


