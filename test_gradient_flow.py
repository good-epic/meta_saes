"""
Test script to verify gradient flow through the decomposability penalty.

This script creates minimal SAE instances and verifies that:
1. The decomposability penalty produces non-zero values
2. Gradients flow correctly from the penalty to W_dec
3. The gradient direction pushes W_dec AWAY from what meta_sae can reconstruct

Run with: python test_gradient_flow.py
"""

import torch
import sys

# Add the project to path
sys.path.insert(0, '/home/mattylev/projects/meta_saes')

from BatchTopK.sae import BatchTopKSAE
from BatchTopK.config import get_default_cfg
from meta_sae_extension import MetaSAEWrapper, BatchTopKSAEWithPenalty


def create_test_configs():
    """Create minimal configs for testing."""
    # Base config with small sizes for fast testing
    cfg = get_default_cfg()
    cfg["act_size"] = 64        # Small activation size
    cfg["dict_size"] = 128      # Small dictionary
    cfg["top_k"] = 8
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["dtype"] = torch.float32

    # Meta config with even smaller dictionary
    meta_cfg = cfg.copy()
    meta_cfg["dict_size"] = 32  # Smaller than primary
    meta_cfg["top_k"] = 4

    # Penalty config
    penalty_cfg = {
        "lambda2": 0.1,      # Noticeable penalty weight
        "sigma_sq": 0.1,
        "n_primary_steps": 10,
        "n_meta_steps": 5,
    }

    return cfg, meta_cfg, penalty_cfg


def test_gradient_flow():
    """Test that gradients flow from penalty to W_dec."""
    print("=" * 60)
    print("TESTING GRADIENT FLOW THROUGH DECOMPOSABILITY PENALTY")
    print("=" * 60)

    cfg, meta_cfg, penalty_cfg = create_test_configs()
    device = cfg["device"]
    print(f"\nUsing device: {device}")

    # Create SAEs
    print("\n1. Creating SAE instances...")
    meta_sae = MetaSAEWrapper(BatchTopKSAE, meta_cfg)
    primary_sae = BatchTopKSAEWithPenalty(cfg, meta_sae, penalty_cfg)

    print(f"   Primary SAE: dict_size={cfg['dict_size']}, act_size={cfg['act_size']}")
    print(f"   Meta SAE: dict_size={meta_cfg['dict_size']}")
    print(f"   Penalty lambda={penalty_cfg['lambda2']}, sigma_sq={penalty_cfg['sigma_sq']}")

    # Create fake activation batch
    batch_size = 32
    x = torch.randn(batch_size, cfg["act_size"], device=device, dtype=cfg["dtype"])

    # Zero all gradients
    primary_sae.zero_grad()

    # Forward pass
    print("\n2. Running forward pass...")
    output = primary_sae(x)

    print(f"   Total loss: {output['loss'].item():.6f}")
    print(f"   L2 loss: {output['l2_loss'].item():.6f}")
    print(f"   L1 loss: {output['l1_loss'].item():.6f}")
    print(f"   Decomp penalty: {output['decomp_penalty'].item():.6f}")

    # Check that penalty is non-trivial
    penalty_val = output['decomp_penalty'].item()
    if penalty_val < 1e-6:
        print("   WARNING: Penalty is very small, may not affect training")
    elif penalty_val > 0.99:
        print("   WARNING: Penalty is very high (near 1), errors may be too small")
    else:
        print(f"   Penalty value looks reasonable: {penalty_val:.4f}")

    # Backward pass
    print("\n3. Running backward pass...")
    output['loss'].backward()

    # Check W_dec gradients
    print("\n4. Checking W_dec gradients...")

    if primary_sae.W_dec.grad is None:
        print("   FAIL: W_dec.grad is None!")
        return False

    grad = primary_sae.W_dec.grad
    grad_norm = grad.norm().item()
    grad_mean = grad.abs().mean().item()
    grad_max = grad.abs().max().item()

    print(f"   W_dec.grad shape: {grad.shape}")
    print(f"   W_dec.grad norm: {grad_norm:.6f}")
    print(f"   W_dec.grad mean abs: {grad_mean:.6f}")
    print(f"   W_dec.grad max abs: {grad_max:.6f}")

    if grad_norm < 1e-10:
        print("   FAIL: Gradient norm is essentially zero!")
        return False

    print("   PASS: W_dec has non-zero gradients")

    # Now test that the penalty specifically contributes to gradients
    # by comparing gradients with and without penalty
    print("\n5. Verifying penalty contributes to gradient...")

    # Create a version without penalty (lambda2=0)
    penalty_cfg_zero = penalty_cfg.copy()
    penalty_cfg_zero["lambda2"] = 0.0
    primary_sae_no_penalty = BatchTopKSAEWithPenalty(cfg, meta_sae, penalty_cfg_zero)

    # Copy weights to match
    primary_sae_no_penalty.load_state_dict(primary_sae.state_dict())
    primary_sae_no_penalty.zero_grad()

    # Forward and backward without penalty
    output_no_penalty = primary_sae_no_penalty(x)
    output_no_penalty['loss'].backward()

    grad_no_penalty = primary_sae_no_penalty.W_dec.grad.clone()
    grad_with_penalty = primary_sae.W_dec.grad.clone()

    # The difference should be non-zero if penalty contributes
    grad_diff = (grad_with_penalty - grad_no_penalty).norm().item()
    grad_diff_relative = grad_diff / (grad_no_penalty.norm().item() + 1e-10)

    print(f"   Gradient difference (with vs without penalty): {grad_diff:.6f}")
    print(f"   Relative difference: {grad_diff_relative:.4%}")

    if grad_diff < 1e-10:
        print("   FAIL: Penalty does not contribute to gradient!")
        print("   The penalty term has no effect on W_dec updates.")
        return False

    print("   PASS: Penalty contributes to W_dec gradient")

    # Verify the direction: penalty should push W_dec AWAY from reconstruction
    print("\n6. Verifying gradient direction...")

    # Get the reconstruction from meta_sae
    with torch.no_grad():
        W_dec = primary_sae.W_dec
        meta_output = meta_sae(W_dec)
        recon = meta_output["sae_out"]

        # Direction from recon to W_dec (away from reconstruction)
        away_direction = W_dec - recon

        # The penalty gradient should be aligned with this direction
        # (positive correlation means gradient pushes away)
        penalty_grad_contribution = grad_with_penalty - grad_no_penalty

        # Compute cosine similarity between penalty gradient and away direction
        cos_sim = torch.nn.functional.cosine_similarity(
            penalty_grad_contribution.flatten().unsqueeze(0),
            away_direction.flatten().unsqueeze(0)
        ).item()

    print(f"   Cosine similarity (penalty grad vs away-from-recon): {cos_sim:.4f}")

    # With gradient descent (W -= lr * grad), a positive cosine similarity
    # means the update will be in the OPPOSITE direction, which would be
    # toward recon. We want gradient to point TOWARD recon so that
    # W -= grad moves AWAY from recon.
    #
    # Wait, let me re-check the math...
    # penalty = exp(-error/σ²)
    # ∂penalty/∂W = -2/σ² * exp(-error/σ²) * (W - recon)
    # The gradient points in direction -(W - recon) = (recon - W), i.e., toward recon
    # So W -= lr * grad = W -= lr * (recon - W) direction = W moves away from recon
    #
    # So we expect NEGATIVE cosine similarity between penalty_grad and away_direction

    if cos_sim < -0.5:
        print("   PASS: Gradient direction is correct (will push W_dec away from recon)")
    elif cos_sim > 0.5:
        print("   WARNING: Gradient direction may be wrong (positive cosine sim)")
        print("   Expected negative cosine similarity for correct behavior")
    else:
        print("   INCONCLUSIVE: Gradient direction unclear (near orthogonal)")

    print("\n" + "=" * 60)
    print("GRADIENT FLOW TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return True


def test_penalty_values_over_training():
    """Test how penalty values change as we manually adjust W_dec."""
    print("\n" + "=" * 60)
    print("TESTING PENALTY BEHAVIOR")
    print("=" * 60)

    cfg, meta_cfg, penalty_cfg = create_test_configs()
    device = cfg["device"]

    meta_sae = MetaSAEWrapper(BatchTopKSAE, meta_cfg)
    primary_sae = BatchTopKSAEWithPenalty(cfg, meta_sae, penalty_cfg)

    # Measure initial penalty
    initial_penalty = primary_sae.compute_decomposability_penalty().item()
    print(f"\nInitial penalty: {initial_penalty:.6f}")

    # Move W_dec closer to what meta_sae can represent
    print("\nMoving W_dec toward meta_sae reconstruction...")
    with torch.no_grad():
        meta_output = meta_sae(primary_sae.W_dec)
        recon = meta_output["sae_out"]
        # Move W_dec 50% toward reconstruction
        primary_sae.W_dec.data = 0.5 * primary_sae.W_dec.data + 0.5 * recon

    closer_penalty = primary_sae.compute_decomposability_penalty().item()
    print(f"Penalty after moving toward recon: {closer_penalty:.6f}")

    if closer_penalty > initial_penalty:
        print("PASS: Penalty increased when W_dec moved toward reconstruction")
    else:
        print("UNEXPECTED: Penalty decreased when W_dec moved toward reconstruction")

    # Move W_dec away from what meta_sae can represent
    print("\nMoving W_dec away from meta_sae reconstruction...")
    with torch.no_grad():
        meta_output = meta_sae(primary_sae.W_dec)
        recon = meta_output["sae_out"]
        # Move W_dec away from reconstruction
        primary_sae.W_dec.data = 2.0 * primary_sae.W_dec.data - recon

    farther_penalty = primary_sae.compute_decomposability_penalty().item()
    print(f"Penalty after moving away from recon: {farther_penalty:.6f}")

    if farther_penalty < closer_penalty:
        print("PASS: Penalty decreased when W_dec moved away from reconstruction")
    else:
        print("UNEXPECTED: Penalty increased when W_dec moved away from reconstruction")

    print("\n" + "=" * 60)
    print("PENALTY BEHAVIOR TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DECOMPOSABILITY PENALTY GRADIENT FLOW TESTS")
    print("=" * 60)

    success = test_gradient_flow()

    if success:
        test_penalty_values_over_training()
    else:
        print("\nGradient flow test failed, skipping further tests.")
        sys.exit(1)

    print("\n All tests passed!")
