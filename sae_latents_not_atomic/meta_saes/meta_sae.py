import numpy as np
import torch
from typing import List
from meta_saes.sae import BaseAutoencoder
from sae_lens.analysis.neuronpedia_integration import \
    get_neuronpedia_quick_list


def get_meta_feature_activations(
    meta_sae: BaseAutoencoder, W_dec: torch.Tensor, cfg: dict, batch_size: int = 4096
) -> np.ndarray:
    """
    Get the meta SAE feature activations for the provided decoder weights.  Uses
    numpy rather than torch for GPU memory efficiency - for larger SAEs it may
    not be possible to load the activations into either GPU or CPU memory, and
    numpy lets us use files as arrays instead.
    """
    activations = []
    for batch in torch.split(W_dec, batch_size):
        batch = batch.to(meta_sae.cfg["device"])
        with torch.no_grad():
            acts = meta_sae(batch, threshold=cfg["threshold"])["feature_acts"].to(
                torch.float16
            )
            activations.append(acts)
    return torch.cat(activations, dim=0).cpu().numpy()



def get_meta_feature_reconstruction(
    meta_sae: BaseAutoencoder, W_dec: torch.Tensor, cfg: dict, batch_size: int = 4096
) -> np.ndarray:
    """
    Get the meta SAE feature activations for the provided decoder weights.  Uses
    numpy rather than torch for GPU memory efficiency - for larger SAEs it may
    not be possible to load the activations into either GPU or CPU memory, and
    numpy lets us use files as arrays instead.
    """
    activations = []
    for batch in torch.split(W_dec, batch_size):
        batch = batch.to(meta_sae.cfg.device)
        with torch.no_grad():
            acts = meta_sae(batch, threshold=cfg.threshold)["sae_out"].to(torch.float16)
            activations.append(acts)
    return torch.cat(activations, dim=0).cpu().numpy()


def get_cluster_features(activations: np.ndarray, cluster: int) -> List:
    """
    Gets the features that are active in the given cluster.
    """
    return np.nonzero(activations[:, cluster] > 0)[0].tolist()


def get_feature_clusters(activations: np.ndarray, feature: int) -> List[int]:
    """
    Gets the clusters that the given feature is active in.
    """
    return activations[feature].nonzero()[0].tolist()


def quicklist_gpt2_feature_splitting(sae, features):
    """
    Wrapper for neuronpedia quick list for gpt2 feature splitting SAEs.
    """
    neuronpedia_quick_list = get_neuronpedia_quick_list(
        features,
        layer=8,
        dataset=f"res_fs{sae.W_dec.size(0)}-jb",
        model="gpt2-small",
        name="list",
    )
    return neuronpedia_quick_list