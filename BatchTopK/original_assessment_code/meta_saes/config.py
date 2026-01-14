from dataclasses import dataclass
from typing import Optional

import torch
import transformer_lens.utils as utils


def get_default_cfg():
    return Config()


@dataclass
class Config:
    seed: int = 49
    epochs: int = 1000
    batch_size: int = 4096
    lr: float = 3e-4
    l1_coeff: float = 0
    beta1: float = 0.9
    beta2: float = 0.99
    num_tokens: int = int(1e9)
    max_grad_norm: int = 100000
    seq_len: int = 128
    dtype: torch.dtype = torch.float32
    model_name: str = "gpt2-small"
    site: str = "resid_pre"
    layer: int = 8
    act_size: int = 768
    dict_size: int = 12288
    device: str = "cuda:0"
    model_batch_size: int = 512
    num_batches_in_buffer: int = 10
    dataset_path: str = "Skylion007/openwebtext"
    wandb_project: str = "sparse_autoencoders"
    input_unit_norm: bool = True
    perf_log_freq: int = 1000
    sae_type: str = "topk"
    architecture: str = "standard"
    checkpoint_freq: int = 10000
    n_batches_to_dead: int = 5
    top_k: int = 32
    top_k_aux: int = 512
    aux_penalty: float = 1 / 32
    hook_point: str = None
    threshold: Optional[torch.Tensor] = None
    name: Optional[str] = ""
    cosine_penalty: Optional[float] = 0.0

    def __post_init__(self):
        self.hook_point = utils.get_act_name(self.site, self.layer)
        self.name = f"{self.model_name}_{self.hook_point}_{self.dict_size}_{self.sae_type}_{self.top_k}_{self.lr}"
        return self
