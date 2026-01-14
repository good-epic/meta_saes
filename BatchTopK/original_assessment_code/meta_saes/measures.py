import torch
from torch.nn import functional as F


def cosine_similarity(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the cosine similarity between two sets of features.

    For example, the decoder or encoder weights of two SAEs.
    """
    f1_norm = F.normalize(m1, p=2, dim=0)
    f2_norm = F.normalize(m2, p=2, dim=0)
    return torch.mm(f1_norm.t(), f2_norm)
