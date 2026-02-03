# %%
# =============================================================================
# DECODER COMPARISON: Joint vs Solo Primary SAEs
# =============================================================================
# Compare decoder matrices between joint-trained and solo-trained primary SAEs
# to see if the composability penalty actually changes decoder vectors.

import sys
import os
import time

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
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
PRINT_TEXT = True  # Set to True to print detailed text readouts
GRID_SEARCH_DIR = Path(__file__).parent.parent / "outputs" / "grid_search_20260128_223720"
OUTPUT_DIR = GRID_SEARCH_DIR / "decoder_comparison_outputs"
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
print(f"\nScanning {GRID_SEARCH_DIR}...")
sys.stdout.flush()

run_dirs = [d for d in GRID_SEARCH_DIR.iterdir() if d.is_dir()]
all_runs = []

for run_dir in tqdm(run_dirs, desc="Loading runs", unit="run"):
    data = load_run_data(run_dir)
    if data is not None:
        all_runs.append(data)

print(f"\nLoaded {len(all_runs)} valid runs from {len(run_dirs)} directories")
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

def compute_matching_metrics(solo_decoder_np, joint_decoder_np, device=DEVICE, run_name=""):
    """
    Compute all metrics for comparing solo vs joint decoder matrices.

    Args:
        solo_decoder_np: (num_features, hidden_dim) decoder matrix from solo SAE (numpy)
        joint_decoder_np: (num_features, hidden_dim) decoder matrix from joint SAE (numpy)
        device: torch device to use
        run_name: name of run for progress display

    Returns:
        dict with all metrics for each matching method
    """
    # Move to GPU
    print(f"    Moving to {device}...", end=" ")
    sys.stdout.flush()
    solo_decoder = torch.from_numpy(solo_decoder_np).float().to(device)
    joint_decoder = torch.from_numpy(joint_decoder_np).float().to(device)
    print("done")
    sys.stdout.flush()

    # Compute similarity and distance matrices on GPU
    print(f"    Computing similarity matrix ({solo_decoder.shape[0]}x{joint_decoder.shape[0]})...", end=" ")
    sys.stdout.flush()
    sim_matrix = cosine_similarity_matrix_gpu(solo_decoder, joint_decoder)
    dist_matrix = l2_distance_matrix_gpu(solo_decoder, joint_decoder)
    print("done")
    sys.stdout.flush()

    # Compute norms
    solo_norms = torch.norm(solo_decoder, dim=1)
    joint_norms = torch.norm(joint_decoder, dim=1)

    results = {}

    # === Hungarian Matching (1:1) ===
    if RUN_HUNGARIAN:
        print(f"    Running Hungarian matching (lapjv)...", end=" ")
        sys.stdout.flush()
        start_hung = time.time()
        sim_np = sim_matrix.cpu().numpy()
        hung_row_idx, hung_col_idx = hungarian_matching(sim_np)
        hung_time = time.time() - start_hung
        print(f"done ({hung_time:.1f}s)")

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

    print(f"\n{'='*60}")
    print(f"Processing {len(runs)} {arch_name} runs")
    print(f"{'='*60}")
    sys.stdout.flush()

    for i, run in enumerate(runs):
        run_name = run['run_dir'].name
        start_time = time.time()

        print(f"\n[{i+1}/{len(runs)}] Starting: {run_name}")
        sys.stdout.flush()

        # Load SAE state dicts
        print(f"  Loading SAE files...", end=" ")
        sys.stdout.flush()
        joint_sae = torch.load(run["run_dir"] / "joint_primary_sae.pt", map_location="cpu")
        solo_sae = torch.load(run["run_dir"] / "solo_primary_sae.pt", map_location="cpu")
        print("done")
        sys.stdout.flush()

        # Extract decoder matrices
        joint_decoder = joint_sae["state_dict"]["W_dec"].numpy()
        solo_decoder = solo_sae["state_dict"]["W_dec"].numpy()
        print(f"  Decoder shape: {solo_decoder.shape}")
        sys.stdout.flush()

        # Compute matching metrics (includes Hungarian matching)
        print(f"  Computing matching metrics...")
        sys.stdout.flush()
        metrics = compute_matching_metrics(solo_decoder, joint_decoder, run_name=run_name)

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
            "joint_l0": run["metrics"]["joint"]["l0"],
            "solo_l0": run["metrics"]["solo"]["l0"],
        }

        all_results.append(result)

        elapsed = time.time() - start_time
        print(f"[{i+1}/{len(runs)}] Completed: {run_name} ({elapsed:.1f}s)")
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"Finished processing all {len(runs)} {arch_name} runs")
    print(f"{'='*60}")
    sys.stdout.flush()

    return all_results

# Process both architectures
jumprelu_results = process_runs(jumprelu_runs, "JumpReLU")
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
            "joint_l0": r["joint_l0"],
            "solo_l0": r["solo_l0"],
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

print("\n" + "="*60)
print("GENERATING SUMMARY STATISTICS")
print("="*60)
sys.stdout.flush()

jumprelu_df = summarize_results(jumprelu_results, "JumpReLU")
batchtopk_df = summarize_results(batchtopk_results, "BatchTopK")

if PRINT_TEXT:
    print("\nJumpReLU Summary:")
    print("-" * 40)
    cols = ["lambda2", "sigma_sq", "solo_cosine_mean", "solo_l2_mean", "solo_frac_above_0.9", "solo_frac_above_0.99"]
    if "hung_cosine_mean" in jumprelu_df.columns:
        cols = ["lambda2", "sigma_sq", "hung_cosine_mean", "hung_l2_mean", "hung_frac_above_0.9", "hung_frac_above_0.99"]
    print(jumprelu_df[cols].to_string())

    print("\nBatchTopK Summary:")
    print("-" * 40)
    print(batchtopk_df[cols].to_string())
    sys.stdout.flush()

# %%
# =============================================================================
# PLOTTING: METRICS VS LAMBDA2
# =============================================================================

print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)
sys.stdout.flush()

def plot_metrics_vs_lambda2(df, results, arch_name):
    """Plot key metrics vs lambda2 for a given architecture."""
    # Get L0 range for title
    l0_min = df["solo_l0"].min()
    l0_max = df["solo_l0"].max()
    l0_info = f" (L0: {l0_min:.0f}-{l0_max:.0f})" if l0_min != l0_max else f" (L0≈{l0_min:.0f})"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{arch_name}: Decoder Similarity vs Lambda2{l0_info}", fontsize=14)

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

    print(f"  Saved {arch_name.lower()}_metrics_vs_lambda2.png")
    sys.stdout.flush()

print("\nGenerating metrics vs lambda2 plots...")
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

    # Get L0 range for title
    all_l0 = [r["solo_l0"] for r in results]
    l0_min, l0_max = min(all_l0), max(all_l0)
    l0_info = f" (L0: {l0_min:.0f}-{l0_max:.0f})" if l0_min != l0_max else f" (L0≈{l0_min:.0f})"

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
    fig.suptitle(f"{arch_name}: Cosine Similarity Distributions by Lambda2{l0_info}", fontsize=14)

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

    print(f"  Saved {arch_name.lower()}_*_distributions.png (3 files)")
    sys.stdout.flush()

print("\nGenerating distribution histograms...")
sys.stdout.flush()
plot_distributions_by_lambda2(jumprelu_results, "JumpReLU")
plot_distributions_by_lambda2(batchtopk_results, "BatchTopK")

# %%
# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\nSaving summary CSVs...")
sys.stdout.flush()
jumprelu_df.to_csv(OUTPUT_DIR / "jumprelu_summary.csv", index=False)
batchtopk_df.to_csv(OUTPUT_DIR / "batchtopk_summary.csv", index=False)
print(f"  Saved jumprelu_summary.csv, batchtopk_summary.csv")

# %%
# =============================================================================
# KEY FINDINGS PRINTOUT
# =============================================================================

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
sys.stdout.flush()

if PRINT_TEXT:

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

# %%
# =============================================================================
# L0 EVOLUTION PLOTS FOR JUMPRELU (DUAL Y-AXIS)
# =============================================================================
# Plot L0 and l0_coeff over training for JumpReLU runs

print("\nGenerating L0 evolution plots for JumpReLU...")
sys.stdout.flush()

def plot_l0_evolution(run, phase="solo_primary"):
    """Load training metrics and plot L0 and l0_coeff evolution."""
    run_dir = Path(run["run_dir"])
    metrics_file = run_dir / f"training_{phase}_metrics.json"

    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        data = json.load(f)

    metrics = data.get("metrics", [])
    if not metrics:
        return None

    steps = [m["step"] for m in metrics]
    l0_values = [m["l0_norm"] for m in metrics]
    l0_coeff_values = [m.get("l0_coeff") for m in metrics]

    # Check if we have l0_coeff data (JumpReLU only)
    has_l0_coeff = any(v is not None for v in l0_coeff_values)

    return {
        "steps": steps,
        "l0": l0_values,
        "l0_coeff": l0_coeff_values if has_l0_coeff else None,
        "lambda2": run["config"]["lambda2"],
        "sigma_sq": run["config"]["sigma_sq"],
    }

def plot_jumprelu_l0_evolution(runs, title_suffix=""):
    """Plot L0 evolution for all JumpReLU runs with dual y-axis."""
    # Group by lambda2
    by_lambda2 = defaultdict(list)
    for run in runs:
        data = plot_l0_evolution(run, "solo_primary")
        if data and data["l0_coeff"] is not None:
            by_lambda2[run["config"]["lambda2"]].append(data)

    if not by_lambda2:
        print("No JumpReLU L0 evolution data found")
        return

    lambda2_values = sorted(by_lambda2.keys())
    n_lambda2 = len(lambda2_values)

    fig, axes = plt.subplots(n_lambda2, 2, figsize=(14, 4*n_lambda2))
    if n_lambda2 == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"JumpReLU L0 & Sparsity Coefficient Evolution{title_suffix}", fontsize=14)

    for i, l2 in enumerate(lambda2_values):
        runs_data = by_lambda2[l2]

        # Left plot: L0 over training
        ax = axes[i, 0]
        for data in runs_data:
            ax.plot(data["steps"], data["l0"], alpha=0.7,
                   label=f"σ²={data['sigma_sq']}")
        ax.axhline(y=64, color='r', linestyle='--', alpha=0.5, label='Target L0=64')
        ax.set_xlabel("Training Step")
        ax.set_ylabel("L0 (Active Features)")
        ax.set_title(f"λ2={l2}: L0 over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right plot: Dual y-axis with L0 and l0_coeff
        ax1 = axes[i, 1]
        ax2 = ax1.twinx()

        colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))

        for j, data in enumerate(runs_data):
            color = colors[j]
            # L0 on left axis
            line1, = ax1.plot(data["steps"], data["l0"], color=color,
                             linestyle='-', alpha=0.7, label=f"L0 (σ²={data['sigma_sq']})")
            # l0_coeff on right axis
            l0_coeff = [v if v is not None else np.nan for v in data["l0_coeff"]]
            line2, = ax2.plot(data["steps"], l0_coeff, color=color,
                             linestyle='--', alpha=0.7, label=f"coeff (σ²={data['sigma_sq']})")

        ax1.axhline(y=64, color='r', linestyle=':', alpha=0.5)
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("L0 (Active Features)", color='black')
        ax2.set_ylabel("L0 Coefficient", color='gray')
        ax1.set_title(f"λ2={l2}: L0 & Coefficient (Dual Axis)")
        ax1.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "jumprelu_l0_evolution.png", dpi=150)
    plt.close()

    print("  Saved jumprelu_l0_evolution.png")
    sys.stdout.flush()

# Also plot joint training L0 evolution
def plot_jumprelu_joint_l0_evolution(runs):
    """Plot L0 evolution during joint training for JumpReLU runs."""
    by_lambda2 = defaultdict(list)
    for run in runs:
        data = plot_l0_evolution(run, "joint_primary")
        if data and data["l0_coeff"] is not None:
            by_lambda2[run["config"]["lambda2"]].append(data)

    if not by_lambda2:
        print("No JumpReLU joint training L0 evolution data found")
        return

    lambda2_values = sorted(by_lambda2.keys())
    n_lambda2 = len(lambda2_values)

    fig, axes = plt.subplots(n_lambda2, 2, figsize=(14, 4*n_lambda2))
    if n_lambda2 == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("JumpReLU Joint Training: L0 & Sparsity Coefficient Evolution", fontsize=14)

    for i, l2 in enumerate(lambda2_values):
        runs_data = by_lambda2[l2]

        ax = axes[i, 0]
        for data in runs_data:
            ax.plot(data["steps"], data["l0"], alpha=0.7,
                   label=f"σ²={data['sigma_sq']}")
        ax.axhline(y=64, color='r', linestyle='--', alpha=0.5, label='Target L0=64')
        ax.set_xlabel("Training Step")
        ax.set_ylabel("L0 (Active Features)")
        ax.set_title(f"Joint Training λ2={l2}: L0 over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax1 = axes[i, 1]
        ax2 = ax1.twinx()

        colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))

        for j, data in enumerate(runs_data):
            color = colors[j]
            ax1.plot(data["steps"], data["l0"], color=color,
                    linestyle='-', alpha=0.7, label=f"L0 (σ²={data['sigma_sq']})")
            l0_coeff = [v if v is not None else np.nan for v in data["l0_coeff"]]
            ax2.plot(data["steps"], l0_coeff, color=color,
                    linestyle='--', alpha=0.7, label=f"coeff (σ²={data['sigma_sq']})")

        ax1.axhline(y=64, color='r', linestyle=':', alpha=0.5)
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("L0 (Active Features)", color='black')
        ax2.set_ylabel("L0 Coefficient", color='gray')
        ax1.set_title(f"Joint Training λ2={l2}: L0 & Coefficient (Dual Axis)")
        ax1.grid(True, alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "jumprelu_joint_l0_evolution.png", dpi=150)
    plt.close()

    print("  Saved jumprelu_joint_l0_evolution.png")
    sys.stdout.flush()

# Plot for JumpReLU runs (using the all_runs data with full config)
jumprelu_runs_full = [r for r in all_runs
                      if r["config"]["primary_sae_type"] == "jumprelu"]

plot_jumprelu_l0_evolution(jumprelu_runs_full, " (Solo Primary Training)")
plot_jumprelu_joint_l0_evolution(jumprelu_runs_full)

# %%
# =============================================================================
# BATCHTOPK vs JUMPRELU COMPARISON
# =============================================================================

print("\nGenerating BatchTopK vs JumpReLU comparison plots...")
sys.stdout.flush()

# Combine both DataFrames with architecture label
if len(jumprelu_df) > 0 and len(batchtopk_df) > 0:
    jumprelu_df["architecture"] = "JumpReLU"
    batchtopk_df["architecture"] = "BatchTopK"
    combined_df = pd.concat([jumprelu_df, batchtopk_df], ignore_index=True)

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("BatchTopK vs JumpReLU: Decoder Similarity Comparison", fontsize=14)

    # Determine metric prefix
    metric_prefix = "hung" if "hung_cosine_mean" in combined_df.columns else "solo"

    # 1. Mean cosine similarity by architecture and lambda2
    ax = axes[0, 0]
    for arch in ["BatchTopK", "JumpReLU"]:
        subset = combined_df[combined_df["architecture"] == arch]
        means = subset.groupby("lambda2")[f"{metric_prefix}_cosine_mean"].mean()
        stds = subset.groupby("lambda2")[f"{metric_prefix}_cosine_mean"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values, marker='o',
                   label=arch, capsize=3)
    ax.set_xlabel("Lambda2")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Cosine Similarity: Joint vs Solo Decoders")
    ax.legend()
    ax.set_xscale('symlog', linthresh=0.001)
    ax.grid(True, alpha=0.3)

    # 2. Fraction above 0.99 by architecture
    ax = axes[0, 1]
    for arch in ["BatchTopK", "JumpReLU"]:
        subset = combined_df[combined_df["architecture"] == arch]
        means = subset.groupby("lambda2")[f"{metric_prefix}_frac_above_0.99"].mean()
        stds = subset.groupby("lambda2")[f"{metric_prefix}_frac_above_0.99"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values, marker='o',
                   label=arch, capsize=3)
    ax.set_xlabel("Lambda2")
    ax.set_ylabel("Fraction > 0.99 Cosine")
    ax.set_title("High-Quality Matches (>0.99 cosine)")
    ax.legend()
    ax.set_xscale('symlog', linthresh=0.001)
    ax.grid(True, alpha=0.3)

    # 3. Mean L2 distance
    ax = axes[0, 2]
    for arch in ["BatchTopK", "JumpReLU"]:
        subset = combined_df[combined_df["architecture"] == arch]
        means = subset.groupby("lambda2")[f"{metric_prefix}_l2_mean"].mean()
        stds = subset.groupby("lambda2")[f"{metric_prefix}_l2_mean"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values, marker='o',
                   label=arch, capsize=3)
    ax.set_xlabel("Lambda2")
    ax.set_ylabel("Mean L2 Distance")
    ax.set_title("L2 Distance: Joint vs Solo Decoders")
    ax.legend()
    ax.set_xscale('symlog', linthresh=0.001)
    ax.grid(True, alpha=0.3)

    # 4. Joint L2 reconstruction quality
    ax = axes[1, 0]
    for arch in ["BatchTopK", "JumpReLU"]:
        subset = combined_df[combined_df["architecture"] == arch]
        means = subset.groupby("lambda2")["joint_l2"].mean()
        stds = subset.groupby("lambda2")["joint_l2"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values, marker='o',
                   label=arch, capsize=3)
    ax.set_xlabel("Lambda2")
    ax.set_ylabel("Joint L2 Error")
    ax.set_title("Reconstruction Quality (Lower = Better)")
    ax.legend()
    ax.set_xscale('symlog', linthresh=0.001)
    ax.grid(True, alpha=0.3)

    # 5. Joint decomp penalty
    ax = axes[1, 1]
    for arch in ["BatchTopK", "JumpReLU"]:
        subset = combined_df[combined_df["architecture"] == arch]
        means = subset.groupby("lambda2")["joint_decomp"].mean()
        stds = subset.groupby("lambda2")["joint_decomp"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values, marker='o',
                   label=arch, capsize=3)
    ax.set_xlabel("Lambda2")
    ax.set_ylabel("Joint Decomp Penalty")
    ax.set_title("Composability Penalty (Lower = Less Composable)")
    ax.legend()
    ax.set_xscale('symlog', linthresh=0.001)
    ax.grid(True, alpha=0.3)

    # 6. Scatter: Cosine sim vs Lambda2, colored by architecture
    ax = axes[1, 2]
    for arch, marker in [("BatchTopK", 'o'), ("JumpReLU", 's')]:
        subset = combined_df[combined_df["architecture"] == arch]
        ax.scatter(subset["lambda2"], subset[f"{metric_prefix}_cosine_mean"],
                  marker=marker, alpha=0.7, label=arch, s=60)
    ax.set_xlabel("Lambda2")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("All Runs: Cosine Similarity")
    ax.legend()
    ax.set_xscale('symlog', linthresh=0.001)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "batchtopk_vs_jumprelu_comparison.png", dpi=150)
    plt.close()

    print("  Saved batchtopk_vs_jumprelu_comparison.png")

    if PRINT_TEXT:
        # Print summary comparison
        print("\nSummary: BatchTopK vs JumpReLU")
        print("-" * 50)
        for arch in ["BatchTopK", "JumpReLU"]:
            subset = combined_df[combined_df["architecture"] == arch]
            print(f"\n{arch}:")
            print(f"  Mean cosine similarity: {subset[f'{metric_prefix}_cosine_mean'].mean():.4f}")
            print(f"  Frac > 0.99 cosine: {subset[f'{metric_prefix}_frac_above_0.99'].mean():.4f}")
            print(f"  Mean L2 distance: {subset[f'{metric_prefix}_l2_mean'].mean():.4f}")
            print(f"  Mean joint L2: {subset['joint_l2'].mean():.6f}")

        sys.stdout.flush()

    # Save combined results
    combined_df.to_csv(OUTPUT_DIR / "combined_comparison.csv", index=False)
    print(f"  Saved combined_comparison.csv")

# %%
# =============================================================================
# INDEPENDENT SAE BASELINE COMPARISON
# =============================================================================
# Compare independent SAE fits to establish baseline variance

INDEPENDENT_SAE_DIR = Path(__file__).parent.parent / "outputs" / "independent_saes"

print("\n" + "="*60)
print("INDEPENDENT SAE BASELINE ANALYSIS")
print("="*60)
sys.stdout.flush()

def load_independent_saes():
    """Load all independent SAE runs."""
    independent_saes = []

    if not INDEPENDENT_SAE_DIR.exists():
        print(f"Independent SAE directory not found: {INDEPENDENT_SAE_DIR}")
        return []

    run_dirs = sorted([d for d in INDEPENDENT_SAE_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")])

    for run_dir in tqdm(run_dirs, desc="Loading independent SAEs"):
        sae_path = run_dir / "primary_sae.pt"
        if sae_path.exists():
            sae_data = torch.load(sae_path, map_location="cpu")
            independent_saes.append({
                "run_dir": run_dir,
                "name": run_dir.name,
                "decoder": sae_data["state_dict"]["W_dec"].numpy(),
                "seed": sae_data.get("seed"),
                "metrics": sae_data.get("metrics", {}),
            })

    return independent_saes

def compute_pairwise_independent_metrics(independent_saes):
    """Compute metrics for all pairs of independent SAEs."""
    from itertools import combinations

    n_saes = len(independent_saes)
    n_pairs = n_saes * (n_saes - 1) // 2
    print(f"Computing {n_pairs} pairwise comparisons for {n_saes} independent SAEs...")
    sys.stdout.flush()

    all_cosines = []
    all_l2_dists = []

    pairs = list(combinations(range(n_saes), 2))

    for i, (idx1, idx2) in enumerate(tqdm(pairs, desc="Pairwise comparisons")):
        sae1 = independent_saes[idx1]
        sae2 = independent_saes[idx2]

        # Compute matching metrics (reuse existing function)
        metrics = compute_matching_metrics(
            sae1["decoder"],
            sae2["decoder"],
            run_name=f"{sae1['name']} vs {sae2['name']}"
        )

        # Use Hungarian matching results
        if "hungarian" in metrics:
            all_cosines.extend(metrics["hungarian"]["cosine_similarities"])
            all_l2_dists.extend(metrics["hungarian"]["l2_distances"])

    return {
        "cosine_similarities": np.array(all_cosines),
        "l2_distances": np.array(all_l2_dists),
        "n_pairs": n_pairs,
        "n_saes": n_saes,
    }

# Load independent SAEs
independent_saes = load_independent_saes()

if len(independent_saes) >= 2:
    print(f"Loaded {len(independent_saes)} independent SAEs")

    # Compute pairwise metrics
    independent_metrics = compute_pairwise_independent_metrics(independent_saes)

    print(f"\nIndependent SAE Baseline Statistics:")
    print(f"  Cosine similarity: mean={np.mean(independent_metrics['cosine_similarities']):.4f}, "
          f"std={np.std(independent_metrics['cosine_similarities']):.4f}")
    print(f"  L2 distance: mean={np.mean(independent_metrics['l2_distances']):.4f}, "
          f"std={np.std(independent_metrics['l2_distances']):.4f}")
    sys.stdout.flush()

    # =========================================================================
    # BASELINE HISTOGRAMS (Independent SAEs only)
    # =========================================================================
    print("\nGenerating baseline histograms...")
    sys.stdout.flush()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Independent SAE Baseline: {independent_metrics['n_pairs']} Pairwise Comparisons", fontsize=14)

    # Cosine similarity histogram
    ax = axes[0]
    ax.hist(independent_metrics["cosine_similarities"], bins=50, density=True,
            alpha=0.7, edgecolor='black', label='Independent pairs')
    ax.axvline(x=0.9, color='r', linestyle='--', alpha=0.5, label='0.9')
    ax.axvline(x=0.99, color='g', linestyle='--', alpha=0.5, label='0.99')
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Cosine Similarity Distribution")
    ax.legend()

    # L2 distance histogram
    ax = axes[1]
    ax.hist(independent_metrics["l2_distances"], bins=50, density=True,
            alpha=0.7, edgecolor='black', label='Independent pairs')
    ax.set_xlabel("L2 Distance")
    ax.set_ylabel("Density")
    ax.set_title("L2 Distance Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "independent_baseline_histograms.png", dpi=150)
    plt.close()
    print("  Saved independent_baseline_histograms.png")

    # =========================================================================
    # COMPARISON: Joint vs Solo vs Independent Baseline
    # =========================================================================
    print("\nGenerating comparison histograms...")
    sys.stdout.flush()

    # Aggregate joint vs solo cosines from JumpReLU runs (or BatchTopK if no JumpReLU)
    if len(jumprelu_results) > 0:
        comparison_results = jumprelu_results
        comparison_name = "JumpReLU"
    else:
        comparison_results = batchtopk_results
        comparison_name = "BatchTopK"

    # Get all cosine similarities from joint vs solo comparisons
    joint_vs_solo_cosines = []
    joint_vs_solo_l2 = []
    for r in comparison_results:
        if "hungarian" in r["metrics"]:
            joint_vs_solo_cosines.extend(r["metrics"]["hungarian"]["cosine_similarities"])
            joint_vs_solo_l2.extend(r["metrics"]["hungarian"]["l2_distances"])

    joint_vs_solo_cosines = np.array(joint_vs_solo_cosines)
    joint_vs_solo_l2 = np.array(joint_vs_solo_l2)

    # Define common bins
    cosine_bins = np.linspace(0, 1, 51)
    l2_max = max(np.percentile(joint_vs_solo_l2, 99), np.percentile(independent_metrics["l2_distances"], 99))
    l2_bins = np.linspace(0, l2_max, 51)

    # Side-by-side comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Decoder Similarity: Joint vs Solo ({comparison_name}) vs Independent Baseline", fontsize=14)

    # Top row: Overlaid histograms
    ax = axes[0, 0]
    ax.hist(joint_vs_solo_cosines, bins=cosine_bins, density=True, alpha=0.6,
            label=f'Joint vs Solo ({comparison_name})', color='blue', edgecolor='blue')
    ax.hist(independent_metrics["cosine_similarities"], bins=cosine_bins, density=True, alpha=0.6,
            label='Independent vs Independent', color='orange', edgecolor='orange')
    ax.axvline(x=0.9, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=0.99, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Cosine Similarity: Overlaid Distributions")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(joint_vs_solo_l2, bins=l2_bins, density=True, alpha=0.6,
            label=f'Joint vs Solo ({comparison_name})', color='blue', edgecolor='blue')
    ax.hist(independent_metrics["l2_distances"], bins=l2_bins, density=True, alpha=0.6,
            label='Independent vs Independent', color='orange', edgecolor='orange')
    ax.set_xlabel("L2 Distance")
    ax.set_ylabel("Density")
    ax.set_title("L2 Distance: Overlaid Distributions")
    ax.legend()

    # Bottom row: Differential histograms
    # Compute histogram values
    joint_solo_cosine_hist, _ = np.histogram(joint_vs_solo_cosines, bins=cosine_bins, density=True)
    independent_cosine_hist, _ = np.histogram(independent_metrics["cosine_similarities"], bins=cosine_bins, density=True)
    cosine_diff = joint_solo_cosine_hist - independent_cosine_hist
    cosine_bin_centers = (cosine_bins[:-1] + cosine_bins[1:]) / 2

    ax = axes[1, 0]
    colors = ['green' if d > 0 else 'red' for d in cosine_diff]
    ax.bar(cosine_bin_centers, cosine_diff, width=cosine_bins[1]-cosine_bins[0],
           color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.99, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density Difference")
    ax.set_title("Differential: (Joint vs Solo) - (Independent vs Independent)\nGreen = more joint/solo pairs, Red = more independent pairs")

    joint_solo_l2_hist, _ = np.histogram(joint_vs_solo_l2, bins=l2_bins, density=True)
    independent_l2_hist, _ = np.histogram(independent_metrics["l2_distances"], bins=l2_bins, density=True)
    l2_diff = joint_solo_l2_hist - independent_l2_hist
    l2_bin_centers = (l2_bins[:-1] + l2_bins[1:]) / 2

    ax = axes[1, 1]
    colors = ['green' if d > 0 else 'red' for d in l2_diff]
    ax.bar(l2_bin_centers, l2_diff, width=l2_bins[1]-l2_bins[0],
           color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel("L2 Distance")
    ax.set_ylabel("Density Difference")
    ax.set_title("Differential: (Joint vs Solo) - (Independent vs Independent)\nGreen = more joint/solo pairs, Red = more independent pairs")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "joint_vs_solo_vs_independent_comparison.png", dpi=150)
    plt.close()
    print("  Saved joint_vs_solo_vs_independent_comparison.png")

    # Print interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    # Compare means
    joint_solo_cosine_mean = np.mean(joint_vs_solo_cosines)
    independent_cosine_mean = np.mean(independent_metrics["cosine_similarities"])

    print(f"\nCosine Similarity:")
    print(f"  Joint vs Solo mean:        {joint_solo_cosine_mean:.4f}")
    print(f"  Independent vs Indep mean: {independent_cosine_mean:.4f}")
    print(f"  Difference:                {joint_solo_cosine_mean - independent_cosine_mean:+.4f}")

    if joint_solo_cosine_mean > independent_cosine_mean:
        print("  → Joint/Solo decoders are MORE similar than independent fits")
        print("    (Joint training preserves decoder structure)")
    else:
        print("  → Joint/Solo decoders are LESS similar than independent fits")
        print("    (Joint training changes decoder more than random variation)")

    joint_solo_l2_mean = np.mean(joint_vs_solo_l2)
    independent_l2_mean = np.mean(independent_metrics["l2_distances"])

    print(f"\nL2 Distance:")
    print(f"  Joint vs Solo mean:        {joint_solo_l2_mean:.4f}")
    print(f"  Independent vs Indep mean: {independent_l2_mean:.4f}")
    print(f"  Difference:                {joint_solo_l2_mean - independent_l2_mean:+.4f}")

    if joint_solo_l2_mean < independent_l2_mean:
        print("  → Joint/Solo decoders are MORE similar than independent fits")
    else:
        print("  → Joint/Solo decoders are LESS similar than independent fits")

    # Fraction above thresholds
    joint_solo_frac_99 = np.mean(joint_vs_solo_cosines >= 0.99)
    independent_frac_99 = np.mean(independent_metrics["cosine_similarities"] >= 0.99)

    print(f"\nFraction with cosine >= 0.99:")
    print(f"  Joint vs Solo:        {joint_solo_frac_99:.4f} ({joint_solo_frac_99*100:.1f}%)")
    print(f"  Independent vs Indep: {independent_frac_99:.4f} ({independent_frac_99*100:.1f}%)")

    sys.stdout.flush()

else:
    print(f"Need at least 2 independent SAEs for comparison, found {len(independent_saes)}")

print("\n" + "="*60)
print("DECODER COMPARISON COMPLETE")
print(f"All outputs saved to: {OUTPUT_DIR}")
print("="*60)
sys.stdout.flush()
