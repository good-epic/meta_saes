# %%
import math

import torch
import tqdm
from meta_saes.activation_store import ActivationsStore
from meta_saes.config import Config
from meta_saes.sae import BatchTopKSAE, load_feature_splitting_saes, load_gemma_sae
from meta_saes.logs import init_wandb, log_model_performance, log_wandb, save_checkpoint
from torch.utils.data import DataLoader, Sampler, TensorDataset
from transformer_lens import HookedTransformer


@torch.no_grad()
def create_dataloader(base_sae, batch_size: int, W_dec=None):
    if W_dec is not None:
        base_decoder_directions = W_dec
    else:
        base_decoder_directions = base_sae.W_dec.detach()

    dataset = TensorDataset(base_decoder_directions)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, drop_last=batch_size < len(dataset), shuffle=batch_size < len(dataset),
    )
    return dataloader


def train_meta_sae(sae, base_sae, wandb_disabled=False):
    cfg = sae.cfg
    optimizer = torch.optim.Adam(
        sae.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
    )
    pbar = tqdm.trange(cfg.epochs)

    wandb_run = init_wandb(cfg, disabled=wandb_disabled)
    dataloader = create_dataloader(base_sae, cfg.batch_size)

    i = 0
    for epoch in pbar:
        for batch in dataloader:
            batch = batch[0].to(cfg.device)
            sae_output = sae(batch, threshold=cfg.threshold)
            log_wandb(sae_output, i, wandb_run)

            loss = sae_output["loss"] * cfg.batch_size
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "L0": f"{sae_output['l0_norm']:.4f}",
                    "L2": f"{sae_output['l2_loss']}",
                    "L1": f"{sae_output['l1_loss']:.4f}",
                    "L1_norm": f"{sae_output['l1_norm']:.4f}",
                }
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.max_grad_norm)
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

            i += 1

        if epoch % 1_000 == 0:
            save_checkpoint(wandb_run, sae, cfg, i)


def train_meta_sae_on_W_dec(sae, W_dec):
    cfg = sae.cfg
    num_batches = cfg.num_tokens // cfg.batch_size
    optimizer = torch.optim.Adam(
        sae.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
    )
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)
    dataloader = create_dataloader(None, cfg.batch_size, W_dec)
    data_iter = iter(dataloader)

    for i in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            dataloader = create_dataloader(None, cfg.batch_size, W_dec)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(cfg.device)
        sae_output = sae(batch, threshold=cfg.threshold)
        log_wandb(sae_output, i, wandb_run)

        if i % cfg.checkpoint_freq == 0:
            save_checkpoint(wandb_run, sae, cfg, i)

        loss = sae_output["loss"] * cfg.batch_size
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "L0": f"{sae_output['l0_norm']:.4f}",
                "L2": f"{sae_output['l2_loss']:.4f}",
                "L1": f"{sae_output['l1_loss']:.4f}",
                "L1_norm": f"{sae_output['l1_norm']:.4f}",
            }
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.max_grad_norm)
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    save_checkpoint(wandb_run, sae, cfg, i)


@torch.no_grad()
def get_combined_W_dec(saes, activations_store, remove_dead_features=False):
    if not remove_dead_features:
        combined_W_dec = torch.cat([sae.W_dec for sae in saes], dim=0)
    else:
        combined_W_dec = []
        total_acts = [torch.zeros(sae.W_dec.shape[0]).to("cuda") for sae in saes]
        total = 0

        for _ in tqdm.tqdm(range(256)):
            batch = activations_store.next_batch()
            total += batch.shape[0]
            for i, sae in enumerate(saes):
                try:
                    acts = sae(batch).feature_acts
                except AttributeError:
                    acts = sae.encode(batch)
                total_acts[i] += (acts > 0).float().sum(dim=0)
        alive_features = [total_acts[i] / total > 1e-5 for i in range(len(saes))]
        combined_W_dec = [
            sae.W_dec[alive_features[i]].detach() for i, sae in enumerate(saes)
        ]
        combined_W_dec = torch.cat(combined_W_dec, dim=0)

    return combined_W_dec


MAIN = __name__ == '__main__'

if MAIN:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, saes, token_dataset = load_gemma_sae(
        release="gemma-scope-2b-pt-res-canonical", 
        sae_id="layer_5/width_1m/canonical",
        device=device,
    )

    sae = saes[0]
    cfg = Config(
        hook_point = sae.cfg.hook_name,
        model_name = sae.cfg.model_name,
        layer = sae.cfg.hook_layer,
        site='resid_post',
        dict_size = 2304,
        act_size = 2304,
        top_k = 4,
        lr = 1e-4,
        l1_coeff = 0,
        aux_penalty = 1 / 16,
        input_unit_norm = False,
        wandb_project = "gemma-scope-2b-pt-res-meta-saes",
        n_batches_to_dead = 100,
        threshold = None,
        cosine_penalty = 0.0000,
        epochs = 30_000,
        batch_size = 2**15,
    )

    meta_sae = BatchTopKSAE(cfg)
    activations_store = ActivationsStore(model, cfg)
    train_meta_sae(meta_sae, sae)
