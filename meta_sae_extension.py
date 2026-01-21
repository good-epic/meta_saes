"""
Extensions for the BatchTopK SAE implementation to support a meta‑SAE and
decomposability penalty.

This module provides two key classes:

* ``MetaSAEWrapper`` – a thin wrapper around an existing SAE class that
  exposes a forward method for arbitrary 2‑D tensors.  The meta SAE is
  trained on the decoder weights of a primary SAE rather than on model
  activations.

* ``BatchTopKSAEWithPenalty`` – a subclass of ``BatchTopKSAE`` that
  computes an additional penalty term based on how well its decoder
  vectors can be reconstructed by a frozen meta SAE.  The penalty is
  added to the overall loss and logged in the output dictionary.

Additionally this file defines ``train_sae_with_meta``, a training loop
that alternates between training the primary SAE and its associated
meta SAE.  Unlike the standard ``training.train_sae`` function, this
loop coordinates two separate models and supports configurable phase
lengths.

The goal of this implementation is to encourage the primary SAE to
learn features that are not simply sparse combinations of the meta
SAE's features.  Users should configure the meta SAE to have a
smaller dictionary size so that it learns atomic components of the
primary SAE's features.

Example usage::

    from sae import BatchTopKSAE
    from meta_sae_extension import MetaSAEWrapper, BatchTopKSAEWithPenalty, train_sae_with_meta

    primary_cfg = {...}  # standard SAE config
    meta_cfg = {...}     # smaller dictionary size for meta SAE
    penalty_cfg = {
        "lambda2": 1e-3,
        "sigma_sq": 0.1,
        "start_with_primary": True,
        "n_primary_steps": 100,
        "n_meta_steps": 10,
    }

    primary_sae = BatchTopKSAEWithPenalty(primary_cfg, meta_sae=None, penalty_cfg=penalty_cfg)
    meta_sae_wrapper = MetaSAEWrapper(BatchTopKSAE, meta_cfg)
    primary_sae.meta_sae = meta_sae_wrapper  # attach after instantiation

    train_sae_with_meta(primary_sae, meta_sae_wrapper, activation_store, model, primary_cfg, meta_cfg, penalty_cfg)

Note: This draft omits integrations with wandb logging and checkpointing
for brevity.  Those can be added by mirroring the existing ``train_sae``
functionality in ``training.py``.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Dict, Any
import tqdm
import torch.optim as optim

try:
    # Import the existing SAE classes from the repository.  If this
    # fails, ensure that the repository is on the Python path.
    from BatchTopK.sae import BatchTopKSAE, TopKSAE, VanillaSAE, JumpReLUSAE
except ImportError as e:
    raise ImportError("Could not import SAE classes from sae.py. "
                      "Ensure that the BatchTopK repository is in the Python path.") from e


# Registry of SAE classes by name
SAE_REGISTRY = {
    "batchtopk": BatchTopKSAE,
    "topk": TopKSAE,
    "vanilla": VanillaSAE,
    "jumprelu": JumpReLUSAE,
}


def get_sae_class(sae_type: str) -> type:
    """Get the SAE class for a given type name.

    Args:
        sae_type: One of 'batchtopk', 'topk', 'vanilla', 'jumprelu'

    Returns:
        The SAE class corresponding to the type.
    """
    sae_type = sae_type.lower()
    if sae_type not in SAE_REGISTRY:
        raise ValueError(f"Unknown SAE type: {sae_type}. Valid options: {list(SAE_REGISTRY.keys())}")
    return SAE_REGISTRY[sae_type]


class MetaSAEWrapper(nn.Module):
    """Wrap an SAE for training on arbitrary 2‑D tensors.

    The meta SAE shares the same architecture as the primary SAE but is
    trained on decoder vectors instead of activations.  The wrapper
    exposes a ``forward_on_vectors`` method that expects a 2‑D tensor
    ``vectors`` of shape ``(num_vectors, act_size)`` and returns the
    reconstruction and latent activations in a dictionary similar to
    the standard SAE output.
    """

    def __init__(self, sae_cls: type, cfg: Dict[str, Any]):
        super().__init__()
        self.meta_sae: nn.Module = sae_cls(cfg)

    def forward(self, vectors: torch.Tensor) -> Dict[str, Any]:
        """Compute the meta SAE output for a batch of vectors.

        Args:
            vectors: A tensor of shape ``(N, act_size)`` containing the
                decoder columns of the primary SAE.
        Returns:
            A dictionary with keys ``"sae_out"`` containing the
            reconstructions of the input vectors and other loss terms
            computed by the meta SAE (e.g. sparsity losses).
        """
        return self.meta_sae(vectors)

    def forward_on_vectors(self, vectors: torch.Tensor) -> Dict[str, Any]:
        """Alias for forward method for backward compatibility."""
        return self.forward(vectors)


class SAEWithPenalty(nn.Module):
    """Generic wrapper that adds decomposability penalty to any SAE.

    This wrapper can be used with any SAE class (BatchTopK, JumpReLU, etc.)
    by composing rather than inheriting. The penalty encourages the SAE
    to learn features that cannot be sparsely reconstructed by a separate,
    smaller meta SAE.
    """

    def __init__(self, sae_cls: type, cfg: Dict[str, Any], meta_sae: MetaSAEWrapper | None, penalty_cfg: Dict[str, Any]):
        super().__init__()
        self.inner_sae = sae_cls(cfg)
        self.meta_sae = meta_sae
        self.penalty_lambda = penalty_cfg.get("lambda2", 0.0)
        self.sigma_sq = penalty_cfg.get("sigma_sq", 1.0)
        self.cfg = cfg

    @property
    def W_dec(self):
        return self.inner_sae.W_dec

    @property
    def W_enc(self):
        return self.inner_sae.W_enc

    @property
    def b_dec(self):
        return self.inner_sae.b_dec

    @property
    def b_enc(self):
        return self.inner_sae.b_enc

    def parameters(self, recurse: bool = True):
        return self.inner_sae.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.inner_sae.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.inner_sae.load_state_dict(state_dict, *args, **kwargs)

    def compute_decomposability_penalty(self) -> torch.Tensor:
        """Compute the penalty based on meta SAE reconstruction error.

        The penalty encourages decoder vectors to be HARD to reconstruct
        by the meta SAE. When reconstruction error is low, the penalty is high.
        """
        if self.meta_sae is None:
            return torch.tensor(0.0, device=self.W_dec.device, dtype=self.W_dec.dtype)

        W_dec = self.W_dec

        with torch.no_grad():
            meta_output = self.meta_sae.forward_on_vectors(W_dec.detach())
            recon = meta_output["sae_out"].detach()

        errors = ((W_dec - recon).pow(2)).sum(dim=1)
        penalties = torch.exp(-errors / self.sigma_sq)
        return penalties.mean()

    def forward(self, x):
        output = self.inner_sae(x)
        decomp_penalty = self.compute_decomposability_penalty()
        output["decomp_penalty"] = decomp_penalty
        output["loss"] = output["loss"] + self.penalty_lambda * decomp_penalty
        return output

    def make_decoder_weights_and_grad_unit_norm(self):
        if hasattr(self.inner_sae, 'make_decoder_weights_and_grad_unit_norm'):
            self.inner_sae.make_decoder_weights_and_grad_unit_norm()


class BatchTopKSAEWithPenalty(BatchTopKSAE):
    """Primary SAE with an additional decomposability penalty.

    The penalty encourages the SAE to learn features that cannot be
    sparsely reconstructed by a separate, smaller meta SAE.  The meta
    SAE is passed in at initialisation time and should be frozen
    during the primary SAE training phase.
    """

    def __init__(self, cfg: Dict[str, Any], meta_sae: MetaSAEWrapper | None, penalty_cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.meta_sae = meta_sae
        # penalty hyperparameters
        self.penalty_lambda = penalty_cfg.get("lambda2", 0.0)
        self.sigma_sq = penalty_cfg.get("sigma_sq", 1.0)

    def compute_decomposability_penalty(self) -> torch.Tensor:
        """Compute the penalty based on meta SAE reconstruction error.

        Returns a scalar tensor representing the average penalty over
        all decoder columns.  If no meta SAE is attached, returns
        zero.

        The penalty encourages decoder vectors to be HARD to reconstruct
        by the meta SAE. When reconstruction error is low (meta SAE can
        easily reconstruct the vector), the penalty is high, pushing
        the decoder vector away from the meta SAE's representational space.

        Gradient flow:
        - Gradients flow TO W_dec (so the penalty can push decoder vectors)
        - Gradients do NOT flow THROUGH meta_sae (it's treated as frozen)
        - The reconstruction target is detached, so W_dec is pushed AWAY from it
        """
        # If no meta SAE is defined, skip penalty.
        if self.meta_sae is None:
            return torch.tensor(0.0, device=self.W_dec.device, dtype=self.W_dec.dtype)

        # Get decoder weights WITH gradient tracking (don't detach!)
        W_dec = self.W_dec  # Shape: [dict_size, act_size]

        # Compute meta SAE reconstruction with frozen meta_sae
        # We detach W_dec for the meta_sae forward pass so gradients
        # don't flow THROUGH meta_sae, but we keep the original W_dec
        # for computing the error so gradients flow TO W_dec
        with torch.no_grad():
            meta_output = self.meta_sae.forward_on_vectors(W_dec.detach())
            recon = meta_output["sae_out"].detach()  # Fixed target, shape: [dict_size, act_size]

        # Compute reconstruction error WITH gradient flow to W_dec
        # recon is detached (fixed target), so gradient of error w.r.t. W_dec is:
        # ∂error/∂W_dec = 2 * (W_dec - recon)
        # This points FROM recon TOWARD W_dec
        errors = ((W_dec - recon).pow(2)).sum(dim=1)  # per-vector reconstruction error

        # Convert errors into penalties: small error -> large penalty
        # penalty = exp(-error/σ²)
        # ∂penalty/∂error = -1/σ² * exp(-error/σ²)  (negative!)
        #
        # Chain rule: ∂penalty/∂W_dec = ∂penalty/∂error * ∂error/∂W_dec
        #            = -1/σ² * exp(-error/σ²) * 2(W_dec - recon)
        #
        # Since we MINIMIZE loss, gradient descent does W_dec -= lr * grad
        # The negative sign means W_dec moves AWAY from recon (increases error)
        penalties = torch.exp(-errors / self.sigma_sq)

        # Average penalty across all decoder vectors
        return penalties.mean()

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        # First compute the base losses (reconstruction, sparsity, aux)
        base_output = super().get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        # Compute decomposability penalty and add to loss
        decomp_penalty = self.compute_decomposability_penalty()
        base_output["decomp_penalty"] = decomp_penalty
        base_output["loss"] = base_output["loss"] + self.penalty_lambda * decomp_penalty
        return base_output


def train_sae_with_meta(
    primary_sae: BatchTopKSAEWithPenalty,
    meta_sae: MetaSAEWrapper,
    activation_store,
    model,
    primary_cfg: Dict[str, Any],
    meta_cfg: Dict[str, Any],
    penalty_cfg: Dict[str, Any],
) -> None:
    """Train a primary SAE with an alternating meta SAE training loop.

    The training alternates between updating the primary SAE on model
    activations and updating the meta SAE on the primary SAE's decoder
    weights.  The alternation begins with the primary SAE phase by
    default

    Args:
        primary_sae: The main SAE model augmented with a decomposability
            penalty.  It must have an attached meta_sae attribute.
        meta_sae: Wrapper around the meta SAE to be trained on
            decoder vectors.
        activation_store: Provides batches of activations for the
            primary SAE.  Must implement ``next_batch()`` returning
            tensors of shape (batch_size, act_size).
        model: The base model whose activations we encode.  Only
            needed for evaluation or logging; not used here.
        primary_cfg: Configuration for the primary SAE training.
        meta_cfg: Configuration for the meta SAE training.
        penalty_cfg: Hyperparameters controlling the alternation and
            penalty strength.  Expected keys include
            ``lambda2``, ``sigma_sq``, ``n_primary_steps``,
            and ``n_meta_steps``.
    """
    # Verify GPU placement
    device = primary_cfg.get("device", "cuda:0")
    print(f"   Device configuration: {device}")
    print(f"   Primary SAE W_enc device: {primary_sae.W_enc.device}")
    print(f"   Meta SAE W_enc device: {meta_sae.meta_sae.W_enc.device}")

    # Attach the meta SAE to the primary for penalty computation.
    primary_sae.meta_sae = meta_sae
    lambda2 = penalty_cfg.get("lambda2", 0.0)
    sigma_sq = penalty_cfg.get("sigma_sq", 1.0)
    n_primary_steps = penalty_cfg.get("n_primary_steps", 100)
    n_meta_steps = penalty_cfg.get("n_meta_steps", 10)

    # Set hyperparameters on the primary SAE
    primary_sae.penalty_lambda = lambda2
    primary_sae.sigma_sq = sigma_sq

    # Optimizers for each SAE
    primary_optimizer = torch.optim.Adam(
        primary_sae.parameters(), lr=primary_cfg["lr"], betas=(primary_cfg["beta1"], primary_cfg["beta2"])
    )
    meta_optimizer = torch.optim.Adam(
        meta_sae.parameters(), lr=meta_cfg["lr"], betas=(meta_cfg["beta1"], meta_cfg["beta2"])
    )

    # Determine number of total batches from primary config
    total_batches = primary_cfg["num_tokens"] // primary_cfg["batch_size"]
    batch_iter = 0
    
    # Create progress bar for overall training
    pbar = tqdm.tqdm(total=total_batches, desc="Training SAE + Meta SAE")
    
    # Alternate training
    # Log frequency - reduce .item() calls which cause CPU-GPU sync
    log_freq = 50

    # Track final metrics
    final_metrics = {}

    first_batch_logged = False
    while batch_iter < total_batches:
        # Primary SAE phase
        for _ in range(n_primary_steps):
            if batch_iter >= total_batches:
                break
            batch = activation_store.next_batch()

            # Verify first batch is on GPU
            if not first_batch_logged:
                print(f"   First batch device: {batch.device}")
                first_batch_logged = True

            output = primary_sae(batch)
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(primary_sae.parameters(), primary_cfg["max_grad_norm"])
            primary_sae.make_decoder_weights_and_grad_unit_norm()
            primary_optimizer.step()
            primary_optimizer.zero_grad()

            batch_iter += 1
            pbar.update(1)

            # Only log every log_freq steps to reduce CPU-GPU sync from .item() calls
            if batch_iter % log_freq == 0:
                final_metrics = {
                    "loss": loss.item(),
                    "l0": output['l0_norm'].item(),
                    "l2": output['l2_loss'].item(),
                    "decomp": output.get('decomp_penalty', torch.tensor(0.0)).item()
                }
                pbar.set_postfix({
                    "Loss": f"{final_metrics['loss']:.4f}",
                    "L0": f"{final_metrics['l0']:.1f}",
                    "L2": f"{final_metrics['l2']:.4f}",
                    "Decomp": f"{final_metrics['decomp']:.4f}"
                })

            # Explicit cleanup to prevent memory leaks
            del batch, output, loss

            # Clear GPU cache less frequently (every 1000 steps instead of 100)
            if batch_iter % 1000 == 0:
                torch.cuda.empty_cache()
        # Meta SAE phase - run without nested progress bar for speed
        for meta_step in range(n_meta_steps):
            # Use the current primary decoder columns as training data for meta SAE
            W_dec = primary_sae.W_dec.detach()
            meta_output = meta_sae(W_dec)
            meta_loss = meta_output["loss"]
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_sae.parameters(), meta_cfg["max_grad_norm"])
            if hasattr(meta_sae.meta_sae, "make_decoder_weights_and_grad_unit_norm"):
                meta_sae.meta_sae.make_decoder_weights_and_grad_unit_norm()
            meta_optimizer.step()
            meta_optimizer.zero_grad()

            # Cleanup meta SAE training variables
            del W_dec, meta_output, meta_loss

    # Close progress bar
    pbar.close()

    return final_metrics


def train_primary_sae_solo(primary_sae, activation_store, cfg):
    """
    Train a primary SAE without any meta SAE or decomposability penalty.
    
    Args:
        primary_sae: Primary SAE model (should be regular BatchTopKSAE, not BatchTopKSAEWithPenalty)
        activation_store: Source of activation batches
        cfg: Primary SAE configuration
    """
    
    print(f"🚀 Training solo primary SAE...")
    print(f"   Dictionary size: {cfg['dict_size']}")
    print(f"   Target tokens: {cfg['num_tokens']:,}")
    print(f"   Batch size: {cfg['batch_size']}")
    print(f"   Top-k: {cfg['top_k']}")
    print(f"   Device: {cfg.get('device', 'cuda:0')}")
    print(f"   SAE W_enc device: {primary_sae.W_enc.device}")
    
    # Setup optimizer
    optimizer = optim.Adam(primary_sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    
    # Calculate training steps
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    
    # Training loop with progress bar
    pbar = tqdm.tqdm(total=num_batches, desc="Solo Primary SAE Training", leave=True)

    primary_sae.train()

    # Log frequency - reduce .item() calls which cause CPU-GPU sync
    log_freq = 50

    # Track final metrics
    final_metrics = {}

    for step in range(num_batches):
        # Get batch
        try:
            batch = activation_store.next_batch()
        except StopIteration:
            print(f"   Dataset exhausted at step {step}, stopping training.")
            break

        # Verify first batch is on GPU
        if step == 0:
            print(f"   First batch device: {batch.device}")

        # Forward pass
        optimizer.zero_grad()
        output = primary_sae(batch)
        loss = output["loss"]

        # Backward pass
        loss.backward()

        # Gradient clipping and decoder weight normalization
        if hasattr(primary_sae, 'make_decoder_weights_and_grad_unit_norm'):
            primary_sae.make_decoder_weights_and_grad_unit_norm()

        optimizer.step()
        pbar.update(1)

        # Only log every log_freq steps to reduce CPU-GPU sync from .item() calls
        if (step + 1) % log_freq == 0:
            final_metrics = {
                'loss': loss.item(),
                'l2': output["l2_loss"].item(),
                'l1': output["l1_loss"].item(),
                'l0': output["l0_norm"].item(),
                'dead': output["num_dead_features"].item()
            }
            pbar.set_postfix({
                'loss': f'{final_metrics["loss"]:.4f}',
                'l2': f'{final_metrics["l2"]:.4f}',
                'l1': f'{final_metrics["l1"]:.4f}',
                'l0': f'{final_metrics["l0"]:.1f}',
                'dead': f'{final_metrics["dead"]}'
            })

        # Memory cleanup
        del batch, output, loss

        # Clear GPU cache less frequently
        if (step + 1) % 1000 == 0:
            torch.cuda.empty_cache()

    pbar.close()
    print("✅ Solo primary SAE training completed!")

    return final_metrics


def train_meta_sae_on_frozen_primary(meta_sae, primary_sae, meta_cfg, penalty_cfg):
    """
    Train a meta SAE on the frozen decoder weights of a pre-trained primary SAE.
    
    Args:
        meta_sae: Meta SAE wrapper
        primary_sae: Pre-trained primary SAE (will be frozen)
        meta_cfg: Meta SAE configuration  
        penalty_cfg: Penalty configuration (used for n_meta_steps)
    """
    
    print(f"🚀 Training meta SAE on frozen primary SAE...")
    print(f"   Meta dictionary size: {meta_cfg['dict_size']}")
    print(f"   Primary dictionary size: {primary_sae.W_dec.shape[0]}")
    print(f"   Meta top-k: {meta_cfg['top_k']}")
    
    # Freeze primary SAE
    for param in primary_sae.parameters():
        param.requires_grad = False
    primary_sae.eval()
    
    # Setup meta SAE optimizer
    meta_optimizer = optim.Adam(meta_sae.parameters(), lr=meta_cfg["lr"])
    
    # Calculate number of meta training steps
    # Use the same total as joint training would get
    primary_batches = penalty_cfg.get("num_tokens", meta_cfg["num_tokens"]) // meta_cfg["batch_size"]
    cycles = primary_batches // penalty_cfg["n_primary_steps"]
    total_meta_steps = cycles * penalty_cfg["n_meta_steps"]
    
    print(f"   Total meta training steps: {total_meta_steps}")
    
    # Training loop
    pbar = tqdm.tqdm(total=total_meta_steps, desc="Meta SAE Training", leave=True)

    meta_sae.train()

    # Log frequency - reduce .item() calls which cause CPU-GPU sync
    log_freq = 50

    # Track final metrics
    final_metrics = {}

    for step in range(total_meta_steps):
        # Get decoder weights (constant input)
        W_dec = primary_sae.W_dec.detach()  # Shape: [dict_size, act_size]

        # Forward pass through meta SAE
        meta_optimizer.zero_grad()
        meta_output = meta_sae(W_dec)
        meta_loss = meta_output["loss"]

        # Backward pass
        meta_loss.backward()

        # Gradient clipping and decoder weight normalization for meta SAE
        if hasattr(meta_sae.meta_sae, 'make_decoder_weights_and_grad_unit_norm'):
            meta_sae.meta_sae.make_decoder_weights_and_grad_unit_norm()

        meta_optimizer.step()
        pbar.update(1)

        # Only log every log_freq steps to reduce CPU-GPU sync from .item() calls
        if (step + 1) % log_freq == 0:
            final_metrics = {
                'loss': meta_loss.item(),
                'l2': meta_output["l2_loss"].item(),
                'l0': meta_output["l0_norm"].item(),
            }
            pbar.set_postfix({
                'loss': f'{final_metrics["loss"]:.4f}',
                'l2': f'{final_metrics["l2"]:.4f}',
                'l0': f'{final_metrics["l0"]:.1f}',
            })

        # Memory cleanup
        del W_dec, meta_output, meta_loss

        # Clear GPU cache less frequently
        if (step + 1) % 1000 == 0:
            torch.cuda.empty_cache()

    pbar.close()
    print("✅ Meta SAE training on frozen primary completed!")

    # Unfreeze primary SAE (restore original state)
    for param in primary_sae.parameters():
        param.requires_grad = True
    primary_sae.train()

    return final_metrics
