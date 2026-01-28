import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.cfg["dict_size"],)).to(
            cfg["device"]
        )

        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(self, x):
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0


class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"] * x.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
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
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


class TopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
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
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
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
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


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
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
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


class JumpReLUSAE(BaseAutoencoder):
    """
    JumpReLU SAE with proper L0 sparsity penalty using sigmoid STE.

    Uses x * H(x - θ) activation where:
    - Forward: hard threshold (preserves full magnitude, no shrinkage)
    - Backward for activations: straight-through (gradient flows through pre_acts)
    - Backward for threshold: sigmoid surrogate provides smooth gradients

    Sparsity modes:
    1. Fixed mode: Set l0_coeff directly, sparsity penalty = l0_coeff * L0
    2. Dynamic mode: Set target_l0, coefficient adapts to achieve target sparsity

    Key config params:
    - bandwidth: Temperature for sigmoid STE (lower = sharper, default 0.001)
    - jumprelu_init_threshold: Initial threshold value (default 0.001)

    Fixed mode:
    - l0_coeff: Fixed sparsity coefficient

    Dynamic mode (if target_l0 is set):
    - target_l0: Target L0 sparsity to achieve
    - l0_coeff_start: Initial sparsity coefficient (default 1e-5, start low)
    - l0_coeff_lr: Learning rate for coefficient updates (default 1e-4)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Learnable threshold per feature (direct, not log-space)
        init_threshold = cfg.get("jumprelu_init_threshold", 0.001)
        self.threshold = nn.Parameter(
            torch.full((cfg["dict_size"],), init_threshold, device=cfg["device"], dtype=cfg["dtype"])
        )
        self.bandwidth = cfg.get("bandwidth", 0.001)

        # Sparsity coefficient setup
        self.target_l0 = cfg.get("target_l0", None)
        if self.target_l0 is not None:
            # Dynamic mode: start with low coefficient, adapt during training
            self.l0_coeff = cfg.get("l0_coeff_start", 1e-5)
            self.l0_coeff_lr = cfg.get("l0_coeff_lr", 1e-4)
            # Track running average of L0 for smoother updates
            self.l0_ema = None
            self.l0_ema_decay = 0.99
        else:
            # Fixed mode: use l0_coeff directly (fall back to l1_coeff for backwards compat)
            self.l0_coeff = cfg.get("l0_coeff", cfg.get("l1_coeff", 0.0))

    def update_l0_coeff(self, current_l0):
        """
        Update sparsity coefficient to move toward target L0.
        Called after each forward pass in dynamic mode.

        Uses proportional control: if L0 > target, increase coefficient to encourage
        more sparsity. If L0 < target, decrease coefficient.
        """
        if self.target_l0 is None:
            return  # Fixed mode, no update

        # Update EMA of L0 for smoother coefficient updates
        current_l0_val = current_l0.item() if torch.is_tensor(current_l0) else current_l0
        if self.l0_ema is None:
            self.l0_ema = current_l0_val
        else:
            self.l0_ema = self.l0_ema_decay * self.l0_ema + (1 - self.l0_ema_decay) * current_l0_val

        # Proportional control: error = current - target
        # Positive error (too many features) -> increase coefficient
        # Negative error (too few features) -> decrease coefficient
        error = self.l0_ema - self.target_l0

        # Update coefficient (multiplicative for stability across scales)
        # Use tanh to bound the update magnitude
        update_factor = 1.0 + self.l0_coeff_lr * (error / max(self.target_l0, 1.0))
        update_factor = max(0.9, min(1.1, update_factor))  # Clamp to avoid instability

        self.l0_coeff = self.l0_coeff * update_factor

        # Clamp coefficient to reasonable range
        self.l0_coeff = max(1e-8, min(1.0, self.l0_coeff))

    def forward(self, x, use_pre_enc_bias=False):
        x, x_mean, x_std = self.preprocess_input(x)

        if use_pre_enc_bias:
            x = x - self.b_dec

        # Pre-activations (before thresholding)
        pre_acts = F.relu(x @ self.W_enc + self.b_enc)

        # JumpReLU: x * H(x - θ)
        # Forward uses hard threshold, gradient flows through pre_acts (straight-through)
        mask = (pre_acts > self.threshold).float()
        acts = pre_acts * mask

        x_reconstruct = acts @ self.W_dec + self.b_dec

        self.update_inactive_features(acts)

        return self.get_loss_dict(x, x_reconstruct, pre_acts, acts, x_mean, x_std)

    def get_loss_dict(self, x, x_reconstruct, pre_acts, acts, x_mean, x_std):
        # Reconstruction loss
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        # L0 sparsity using sigmoid STE for differentiable threshold learning
        # sigmoid((x - θ) / bandwidth) ≈ H(x - θ) but with smooth gradients
        l0_surrogate = torch.sigmoid(
            (pre_acts - self.threshold) / self.bandwidth
        ).sum(dim=-1).mean()

        # Actual L0 for logging (non-differentiable, uses hard threshold)
        l0_actual = (pre_acts > self.threshold).float().sum(dim=-1).mean()

        # Update coefficient if in dynamic mode (do this before computing loss)
        self.update_l0_coeff(l0_actual)

        # Sparsity loss (uses surrogate for gradient, but we report actual L0)
        sparsity_loss = self.l0_coeff * l0_surrogate

        loss = l2_loss + sparsity_loss

        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l0_loss": sparsity_loss,
            "l0_coeff": self.l0_coeff,  # Log current coefficient
            "l2_loss": l2_loss,
            "l0_norm": l0_actual,
            # Backwards compat
            "l1_loss": sparsity_loss,
            "l1_norm": l0_actual,
        }
        if self.target_l0 is not None:
            output["target_l0"] = self.target_l0
            output["l0_ema"] = self.l0_ema
        return output
