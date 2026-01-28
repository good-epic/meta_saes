# %%
# =============================================================================
# DECODER COMPARISON: Joint vs Solo Primary SAEs
# =============================================================================
# Compare decoder matrices between joint-trained and solo-trained primary SAEs
# to see if the composability penalty actually changes decoder vectors.

import sys
import os

# Non-interactive backend for script execution
if not hasattr(sys, 'ps1'):
    import matplotlib
    matplotlib.use('Agg')

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================
PRINT_TEXT = True  # Set to True to print detailed text readouts
GRID_SEARCH_DIR = Path(__file__).parent.parent / "outputs" / "grid_search_20260116_212533"
OUTPUT_DIR = Path(__file__).parent / "decoder_comparison_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# GPU settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hungarian matching - uses lapjv (fast)
RUN_HUNGARIAN = True

# %%
# =============================================================================
# LOAD ALL RUNS AND FILTER BY ARCHITECTURE
# =============================================================================

def load_run_data(run_dir):
    """Load metrics and check if SAE files exist for a run."""
    metrics_path = run_dir / "metrics.json"
    params_path = run_dir / "params.json"
    joint_sae_path = run_dir / "joint_primary_sae.pt"
    solo_sae_path = run_dir / "solo_primary_sae.pt"

    if not all(p.exists() for p in [metrics_path, joint_sae_path, solo_sae_path]):
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    if metrics.get("sequential_meta") is None:
        return None

    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "config": metrics["config"],
    }

# Load all runs
all_runs = []
for run_dir in GRID_SEARCH_DIR.iterdir():
    if run_dir.is_dir():
        data = load_run_data(run_dir)
        if data is not None:
            all_runs.append(data)

if PRINT_TEXT:
    print(f"Loaded {len(all_runs)} valid runs")
    print(f"Using device: {DEVICE}")
    print(f"Hungarian matching: {RUN_HUNGARIAN}")
    sys.stdout.flush()

# Filter to same-architecture runs
jumprelu_runs = [r for r in all_runs
                 if r["config"]["primary_sae_type"] == "jumprelu"
                 and r["config"]["meta_sae_type"] == "jumprelu"]

batchtopk_runs = [r for r in all_runs
                  if r["config"]["primary_sae_type"] == "batchtopk"
                  and r["config"]["meta_sae_type"] == "batchtopk"]

if PRINT_TEXT:
    print(f"JumpReLU-JumpReLU runs: {len(jumprelu_runs)}")
    print(f"BatchTopK-BatchTopK runs: {len(batchtopk_runs)}")
    sys.stdout.flush()

# %%
# =============================================================================
# GPU-ACCELERATED MATCHING FUNCTIONS
# =============================================================================

def cosine_similarity_matrix_gpu(A, B):
    """Compute cosine similarity matrix between rows of A and B on GPU."""
    # Normalize rows
    A_norm = A / (torch.norm(A, dim=1, keepdim=True) + 1e-10)
    B_norm = B / (torch.norm(B, dim=1, keepdim=True) + 1e-10)
    return A_norm @ B_norm.T

def l2_distance_matrix_gpu(A, B):
    """Compute L2 distance matrix between rows of A and B on GPU."""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    A_sq = torch.sum(A**2, dim=1, keepdim=True)
    B_sq = torch.sum(B**2, dim=1, keepdim=True)
    dist_sq = A_sq + B_sq.T - 2 * A @ B.T
    dist_sq = torch.clamp(dist_sq, min=0)  # numerical stability
    return torch.sqrt(dist_sq)

def hungarian_matching(sim_matrix_np):
    """
    Optimal 1:1 matching using Hungarian algorithm (lapjv - fast).
    Returns: (row_indices, col_indices) of matched pairs
    """
    from lapjv import lapjv
    # lapjv minimizes cost, so negate similarity
    cost_matrix = -sim_matrix_np
    # lapjv returns (row_to_col, col_to_row, cost)
    row_to_col, _, _ = lapjv(cost_matrix)
    row_ind = np.arange(len(row_to_col))
    col_ind = row_to_col
    return row_ind, col_ind

def closest_match_A_to_B(sim_matrix):
    """
    For each row in A, find the closest (most similar) row in B.
    Returns: tensor of column indices (length = num rows in A)
    """
    return torch.argmax(sim_matrix, dim=1)

def closest_match_B_to_A(sim_matrix):
    """
    For each column in B, find the closest (most similar) row in A.
    Returns: tensor of row indices (length = num cols in B)
    """
    return torch.argmax(sim_matrix, dim=0)

# %%
# =============================================================================
# METRICS COMPUTATION (GPU-ACCELERATED)
# =============================================================================

def compute_matching_metrics(solo_decoder_np, joint_decoder_np, device=DEVICE):
    """
    Compute all metrics for comparing solo vs joint decoder matrices.

    Args:
        solo_decoder_np: (num_features, hidden_dim) decoder matrix from solo SAE (numpy)
        joint_decoder_np: (num_features, hidden_dim) decoder matrix from joint SAE (numpy)
        device: torch device to use

    Returns:
        dict with all metrics for each matching method
    """
    # Move to GPU
    solo_decoder = torch.from_numpy(solo_decoder_np).float().to(device)
    joint_decoder = torch.from_numpy(joint_decoder_np).float().to(device)

    # Compute similarity and distance matrices on GPU
    sim_matrix = cosine_similarity_matrix_gpu(solo_decoder, joint_decoder)
    dist_matrix = l2_distance_matrix_gpu(solo_decoder, joint_decoder)

    # Compute norms
    solo_norms = torch.norm(solo_decoder, dim=1)
    joint_norms = torch.norm(joint_decoder, dim=1)

    results = {}

    # === Hungarian Matching (1:1) ===
    if RUN_HUNGARIAN:
        sim_np = sim_matrix.cpu().numpy()
        hung_row_idx, hung_col_idx = hungarian_matching(sim_np)

        hung_cosines = sim_np[hung_row_idx, hung_col_idx]
        hung_l2_dists = dist_matrix[hung_row_idx, hung_col_idx].cpu().numpy()
        hung_solo_norms = solo_norms[hung_row_idx].cpu().numpy()
        hung_joint_norms = joint_norms[hung_col_idx].cpu().numpy()

        results["hungarian"] = {
            "cosine_similarities": hung_cosines,
            "l2_distances": hung_l2_dists,
            "solo_norms": hung_solo_norms,
            "joint_norms": hung_joint_norms,
            "norm_ratios": hung_joint_norms / (hung_solo_norms + 1e-10),
        }

    # === Solo → Joint (each solo finds closest joint) ===
    s2j_joint_idx = closest_match_A_to_B(sim_matrix)
    s2j_solo_idx = torch.arange(len(solo_decoder), device=device)

    s2j_cosines = sim_matrix[s2j_solo_idx, s2j_joint_idx].cpu().numpy()
    s2j_l2_dists = dist_matrix[s2j_solo_idx, s2j_joint_idx].cpu().numpy()
    s2j_solo_norms = solo_norms.cpu().numpy()
    s2j_joint_norms = joint_norms[s2j_joint_idx].cpu().numpy()

    # Coverage: how many unique joint vectors are matched?
    s2j_unique_joint = len(torch.unique(s2j_joint_idx))
    s2j_orphan_joint = len(joint_decoder) - s2j_unique_joint

    results["solo_to_joint"] = {
        "cosine_similarities": s2j_cosines,
        "l2_distances": s2j_l2_dists,
        "solo_norms": s2j_solo_norms,
        "joint_norms": s2j_joint_norms,
        "norm_ratios": s2j_joint_norms / (s2j_solo_norms + 1e-10),
        "unique_targets": s2j_unique_joint,
        "orphan_targets": s2j_orphan_joint,
    }

    # === Joint → Solo (each joint finds closest solo) ===
    j2s_solo_idx = closest_match_B_to_A(sim_matrix)
    j2s_joint_idx = torch.arange(len(joint_decoder), device=device)

    j2s_cosines = sim_matrix[j2s_solo_idx, j2s_joint_idx].cpu().numpy()
    j2s_l2_dists = dist_matrix[j2s_solo_idx, j2s_joint_idx].cpu().numpy()
    j2s_solo_norms = solo_norms[j2s_solo_idx].cpu().numpy()
    j2s_joint_norms = joint_norms.cpu().numpy()

    # Coverage: how many unique solo vectors are matched?
    j2s_unique_solo = len(torch.unique(j2s_solo_idx))
    j2s_orphan_solo = len(solo_decoder) - j2s_unique_solo

    results["joint_to_solo"] = {
        "cosine_similarities": j2s_cosines,
        "l2_distances": j2s_l2_dists,
        "solo_norms": j2s_solo_norms,
        "joint_norms": j2s_joint_norms,
        "norm_ratios": j2s_joint_norms / (j2s_solo_norms + 1e-10),
        "unique_targets": j2s_unique_solo,
        "orphan_targets": j2s_orphan_solo,
    }

    return results

def compute_match_quality_thresholds(cosines, thresholds=[0.9, 0.95, 0.99]):
    """Compute fraction of matches above various cosine similarity thresholds."""
    return {t: np.mean(cosines >= t) for t in thresholds}

def compute_dead_feature_overlap(solo_sae, joint_sae):
    """
    Compute overlap in dead features between solo and joint SAEs.
    Dead features have zero or near-zero decoder norms.
    """
    # Get decoder weights (nested under state_dict)
    solo_decoder = solo_sae["state_dict"]["W_dec"].numpy()
    joint_decoder = joint_sae["state_dict"]["W_dec"].numpy()

    # Compute norms
    solo_norms = np.linalg.norm(solo_decoder, axis=1)
    joint_norms = np.linalg.norm(joint_decoder, axis=1)

    # Dead threshold (very small norm)
    threshold = 1e-6
    solo_dead = solo_norms < threshold
    joint_dead = joint_norms < threshold

    both_dead = np.sum(solo_dead & joint_dead)
    solo_only_dead = np.sum(solo_dead & ~joint_dead)
    joint_only_dead = np.sum(~solo_dead & joint_dead)
    both_alive = np.sum(~solo_dead & ~joint_dead)

    return {
        "both_dead": both_dead,
        "solo_only_dead": solo_only_dead,
        "joint_only_dead": joint_only_dead,
        "both_alive": both_alive,
        "total_solo_dead": np.sum(solo_dead),
        "total_joint_dead": np.sum(joint_dead),
    }

# %%
# =============================================================================
# PROCESS ALL RUNS
# =============================================================================

def process_runs(runs, arch_name):
    """Process all runs for a given architecture and return aggregated results."""
    all_results = []

    for i, run in enumerate(runs):
        if PRINT_TEXT:
            print(f"Processing {arch_name} run {i+1}/{len(runs)}: {run['run_dir'].name}")
            sys.stdout.flush()

        # Load SAE state dicts
        joint_sae = torch.load(run["run_dir"] / "joint_primary_sae.pt", map_location="cpu")
        solo_sae = torch.load(run["run_dir"] / "solo_primary_sae.pt", map_location="cpu")

        # Extract decoder matrices
        joint_decoder = joint_sae["state_dict"]["W_dec"].numpy()
        solo_decoder = solo_sae["state_dict"]["W_dec"].numpy()

        # Compute matching metrics
        metrics = compute_matching_metrics(solo_decoder, joint_decoder)

        # Compute dead feature overlap
        dead_overlap = compute_dead_feature_overlap(solo_sae, joint_sae)

        # Add config info
        result = {
            "run_dir": str(run["run_dir"]),
            "lambda2": run["config"]["lambda2"],
            "sigma_sq": run["config"]["sigma_sq"],
            "meta_dict_size": run["config"]["meta_dict_size"],
            "metrics": metrics,
            "dead_overlap": dead_overlap,
            # Add summary stats
            "joint_l2": run["metrics"]["joint"]["l2"],
            "solo_l2": run["metrics"]["solo"]["l2"],
            "joint_decomp": run["metrics"]["joint"]["decomp"],
        }

        all_results.append(result)

    return all_results

# Process both architectures
if PRINT_TEXT:
    print("\n" + "="*60)
    print("Processing JumpReLU runs...")
    print("="*60)
    sys.stdout.flush()
jumprelu_results = process_runs(jumprelu_runs, "JumpReLU")

if PRINT_TEXT:
    print("\n" + "="*60)
    print("Processing BatchTopK runs...")
    print("="*60)
    sys.stdout.flush()
batchtopk_results = process_runs(batchtopk_runs, "BatchTopK")

# %%
# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def summarize_results(results, arch_name):
    """Create summary DataFrame from results."""
    rows = []
    for r in results:
        row = {
            "lambda2": r["lambda2"],
            "sigma_sq": r["sigma_sq"],
            "meta_dict_size": r["meta_dict_size"],
            "joint_l2": r["joint_l2"],
            "solo_l2": r["solo_l2"],
            "joint_decomp": r["joint_decomp"],
        }

        # Add summary stats for each matching method
        methods = ["solo_to_joint", "joint_to_solo"]
        if "hungarian" in r["metrics"]:
            methods.insert(0, "hungarian")

        for method in methods:
            m = r["metrics"][method]
            prefix = method[:4]  # hung, solo, join

            row[f"{prefix}_cosine_mean"] = np.mean(m["cosine_similarities"])
            row[f"{prefix}_cosine_median"] = np.median(m["cosine_similarities"])
            row[f"{prefix}_cosine_std"] = np.std(m["cosine_similarities"])
            row[f"{prefix}_l2_mean"] = np.mean(m["l2_distances"])
            row[f"{prefix}_l2_median"] = np.median(m["l2_distances"])

            # Threshold stats
            thresholds = compute_match_quality_thresholds(m["cosine_similarities"])
            for t, frac in thresholds.items():
                row[f"{prefix}_frac_above_{t}"] = frac

            # Coverage (only for non-Hungarian)
            if "orphan_targets" in m:
                row[f"{prefix}_orphan_count"] = m["orphan_targets"]
                row[f"{prefix}_unique_targets"] = m["unique_targets"]

        # Dead feature stats
        d = r["dead_overlap"]
        row["dead_both"] = d["both_dead"]
        row["dead_solo_only"] = d["solo_only_dead"]
        row["dead_joint_only"] = d["joint_only_dead"]

        rows.append(row)

    df = pd.DataFrame(rows)
    return df

jumprelu_df = summarize_results(jumprelu_results, "JumpReLU")
batchtopk_df = summarize_results(batchtopk_results, "BatchTopK")

if PRINT_TEXT:
    print("\n" + "="*60)
    print("JumpReLU Summary")
    print("="*60)
    cols = ["lambda2", "sigma_sq", "solo_cosine_mean", "solo_l2_mean", "solo_frac_above_0.9", "solo_frac_above_0.99"]
    if "hung_cosine_mean" in jumprelu_df.columns:
        cols = ["lambda2", "sigma_sq", "hung_cosine_mean", "hung_l2_mean", "hung_frac_above_0.9", "hung_frac_above_0.99"]
    print(jumprelu_df[cols].to_string())

    print("\n" + "="*60)
    print("BatchTopK Summary")
    print("="*60)
    print(batchtopk_df[cols].to_string())
    sys.stdout.flush()

# %%
# =============================================================================
# PLOTTING: METRICS VS LAMBDA2
# =============================================================================

def plot_metrics_vs_lambda2(df, results, arch_name):
    """Plot key metrics vs lambda2 for a given architecture."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{arch_name}: Decoder Similarity vs Lambda2", fontsize=14)

    # Determine which methods are available
    methods = [("solo", "Solo→Joint"), ("join", "Joint→Solo")]
    if "hung_cosine_mean" in df.columns:
        methods.insert(0, ("hung", "Hungarian"))

    # Plot 1: Mean cosine similarity by matching method
    ax = axes[0, 0]
    for method, label in methods:
        means = df.groupby("lambda2")[f"{method}_cosine_mean"].mean()
        stds = df.groupby("lambda2")[f"{method}_cosine_mean"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values, marker='o', label=label, capsize=3)
    ax.set_xlabel("Lambda2")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Mean Cosine Similarity vs Lambda2")
    ax.legend()
    ax.set_xscale('symlog', linthresh=0.001)

    # Plot 2: Mean L2 distance by matching method
    ax = axes[0, 1]
    for method, label in methods:
        means = df.groupby("lambda2")[f"{method}_l2_mean"].mean()
        stds = df.groupby("lambda2")[f"{method}_l2_mean"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values, marker='o', label=label, capsize=3)
    ax.set_xlabel("Lambda2")
    ax.set_ylabel("Mean L2 Distance")
    ax.set_title("Mean L2 Distance vs Lambda2")
    ax.legend()
    ax.set_xscale('symlog', linthresh=0.001)

    # Plot 3: Fraction above 0.99 cosine similarity
    ax = axes[0, 2]
    for method, label in methods:
        means = df.groupby("lambda2")[f"{method}_frac_above_0.99"].mean()
        stds = df.groupby("lambda2")[f"{method}_frac_above_0.99"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values, marker='o', label=label, capsize=3)
    ax.set_xlabel("Lambda2")
    ax.set_ylabel("Fraction > 0.99 Cosine Sim")
    ax.set_title("High-Quality Matches vs Lambda2")
    ax.legend()
    ax.set_xscale('symlog', linthresh=0.001)

    # Plot 4: Orphan counts (coverage)
    ax = axes[1, 0]
    for method, label in [("solo", "Solo→Joint orphans"), ("join", "Joint→Solo orphans")]:
        means = df.groupby("lambda2")[f"{method}_orphan_count"].mean()
        stds = df.groupby("lambda2")[f"{method}_orphan_count"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values, marker='o', label=label, capsize=3)
    ax.set_xlabel("Lambda2")
    ax.set_ylabel("Orphan Count")
    ax.set_title("Unmatched Features vs Lambda2")
    ax.legend()
    ax.set_xscale('symlog', linthresh=0.001)

    # Plot 5: Joint decomp (the penalty value) vs lambda2
    ax = axes[1, 1]
    means = df.groupby("lambda2")["joint_decomp"].mean()
    stds = df.groupby("lambda2")["joint_decomp"].std()
    ax.errorbar(means.index, means.values, yerr=stds.values, marker='o', capsize=3)
    ax.set_xlabel("Lambda2")
    ax.set_ylabel("Joint Decomp")
    ax.set_title("Composability Penalty vs Lambda2")
    ax.set_xscale('symlog', linthresh=0.001)

    # Plot 6: Joint L2 vs Solo L2
    ax = axes[1, 2]
    scatter = ax.scatter(df["solo_l2"], df["joint_l2"], c=df["lambda2"], cmap='viridis', alpha=0.7)
    ax.plot([df["solo_l2"].min(), df["solo_l2"].max()],
            [df["solo_l2"].min(), df["solo_l2"].max()], 'k--', alpha=0.5)
    ax.set_xlabel("Solo L2")
    ax.set_ylabel("Joint L2")
    ax.set_title("Reconstruction Quality")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Lambda2")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{arch_name.lower()}_metrics_vs_lambda2.png", dpi=150)
    plt.close()

    if PRINT_TEXT:
        print(f"Saved {arch_name.lower()}_metrics_vs_lambda2.png")
        sys.stdout.flush()

plot_metrics_vs_lambda2(jumprelu_df, jumprelu_results, "JumpReLU")
plot_metrics_vs_lambda2(batchtopk_df, batchtopk_results, "BatchTopK")

# %%
# =============================================================================
# PLOTTING: DISTRIBUTION HISTOGRAMS
# =============================================================================

def plot_distributions_by_lambda2(results, arch_name):
    """Plot distribution histograms for different lambda2 values."""

    # Group results by lambda2
    by_lambda2 = defaultdict(list)
    for r in results:
        by_lambda2[r["lambda2"]].append(r)

    lambda2_values = sorted(by_lambda2.keys())
    n_lambda2 = len(lambda2_values)

    # Determine which methods are available
    sample_result = results[0]
    methods = [("solo_to_joint", "Solo→Joint"), ("joint_to_solo", "Joint→Solo")]
    if "hungarian" in sample_result["metrics"]:
        methods.insert(0, ("hungarian", "Hungarian"))
    n_methods = len(methods)

    # Plot cosine similarity distributions
    fig, axes = plt.subplots(n_lambda2, n_methods, figsize=(5*n_methods, 4*n_lambda2))
    if n_lambda2 == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f"{arch_name}: Cosine Similarity Distributions by Lambda2", fontsize=14)

    for i, l2 in enumerate(lambda2_values):
        runs = by_lambda2[l2]

        for j, (method, label) in enumerate(methods):
            ax = axes[i, j]

            # Aggregate all cosine similarities for this lambda2
            all_cosines = np.concatenate([r["metrics"][method]["cosine_similarities"] for r in runs])

            ax.hist(all_cosines, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(x=0.9, color='r', linestyle='--', alpha=0.5, label='0.9')
            ax.axvline(x=0.99, color='g', linestyle='--', alpha=0.5, label='0.99')
            ax.set_xlabel("Cosine Similarity")
            ax.set_ylabel("Count")
            ax.set_title(f"λ2={l2}, {label}")
            if i == 0 and j == 0:
                ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{arch_name.lower()}_cosine_distributions.png", dpi=150)
    plt.close()

    # Plot L2 distance distributions
    fig, axes = plt.subplots(n_lambda2, n_methods, figsize=(5*n_methods, 4*n_lambda2))
    if n_lambda2 == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f"{arch_name}: L2 Distance Distributions by Lambda2", fontsize=14)

    for i, l2 in enumerate(lambda2_values):
        runs = by_lambda2[l2]

        for j, (method, label) in enumerate(methods):
            ax = axes[i, j]

            all_l2_dists = np.concatenate([r["metrics"][method]["l2_distances"] for r in runs])

            ax.hist(all_l2_dists, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel("L2 Distance")
            ax.set_ylabel("Count")
            ax.set_title(f"λ2={l2}, {label}")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{arch_name.lower()}_l2_distributions.png", dpi=150)
    plt.close()

    # Plot norm ratio distributions
    fig, axes = plt.subplots(n_lambda2, n_methods, figsize=(5*n_methods, 4*n_lambda2))
    if n_lambda2 == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f"{arch_name}: Norm Ratio (Joint/Solo) Distributions by Lambda2", fontsize=14)

    for i, l2 in enumerate(lambda2_values):
        runs = by_lambda2[l2]

        for j, (method, label) in enumerate(methods):
            ax = axes[i, j]

            all_ratios = np.concatenate([r["metrics"][method]["norm_ratios"] for r in runs])
            # Clip extreme ratios for visualization
            all_ratios = np.clip(all_ratios, 0, 3)

            ax.hist(all_ratios, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='1.0')
            ax.set_xlabel("Norm Ratio (Joint/Solo)")
            ax.set_ylabel("Count")
            ax.set_title(f"λ2={l2}, {label}")
            if i == 0 and j == 0:
                ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{arch_name.lower()}_norm_ratio_distributions.png", dpi=150)
    plt.close()

    if PRINT_TEXT:
        print(f"Saved {arch_name.lower()}_*_distributions.png")
        sys.stdout.flush()

plot_distributions_by_lambda2(jumprelu_results, "JumpReLU")
plot_distributions_by_lambda2(batchtopk_results, "BatchTopK")

# %%
# =============================================================================
# SAVE RESULTS
# =============================================================================

jumprelu_df.to_csv(OUTPUT_DIR / "jumprelu_summary.csv", index=False)
batchtopk_df.to_csv(OUTPUT_DIR / "batchtopk_summary.csv", index=False)

if PRINT_TEXT:
    print(f"\nSaved summary CSVs to {OUTPUT_DIR}")
    print("\nDone!")

# %%
# =============================================================================
# KEY FINDINGS PRINTOUT
# =============================================================================

if PRINT_TEXT:
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    for arch_name, df in [("JumpReLU", jumprelu_df), ("BatchTopK", batchtopk_df)]:
        print(f"\n{arch_name}:")
        print("-" * 40)

        # Use solo_to_joint as the main metric (or hungarian if available)
        metric_prefix = "hung" if "hung_cosine_mean" in df.columns else "solo"

        # Effect of lambda2 on cosine similarity
        low_l2 = df[df["lambda2"] == 0.0][f"{metric_prefix}_cosine_mean"].mean()
        high_l2 = df[df["lambda2"] == df["lambda2"].max()][f"{metric_prefix}_cosine_mean"].mean()
        print(f"  {metric_prefix.title()} cosine sim: λ2=0 → {low_l2:.4f}, λ2=max → {high_l2:.4f}")

        # Effect on high-quality matches
        low_l2_frac = df[df["lambda2"] == 0.0][f"{metric_prefix}_frac_above_0.99"].mean()
        high_l2_frac = df[df["lambda2"] == df["lambda2"].max()][f"{metric_prefix}_frac_above_0.99"].mean()
        print(f"  Frac > 0.99 cosine: λ2=0 → {low_l2_frac:.4f}, λ2=max → {high_l2_frac:.4f}")

        # Reconstruction quality tradeoff
        low_l2_recon = df[df["lambda2"] == 0.0]["joint_l2"].mean()
        high_l2_recon = df[df["lambda2"] == df["lambda2"].max()]["joint_l2"].mean()
        print(f"  Joint L2 error: λ2=0 → {low_l2_recon:.6f}, λ2=max → {high_l2_recon:.6f}")

    sys.stdout.flush()
