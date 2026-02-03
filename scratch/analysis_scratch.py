# %% [markdown]
# # Grid Search Analysis
#
# Interactive analysis of meta-SAE grid search results.
# Run cells with `# %%` markers interactively.

# %%
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Use non-interactive backend when running as script (not in notebook)
import sys
if not hasattr(sys, 'ps1'):  # Not in interactive mode
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION
# =============================================================================
PRINT_TEXT = False  # Set to True to print detailed text readouts

# %%
# =============================================================================
# 1. LOAD DATA
# =============================================================================

grid_dir = Path("../outputs/grid_search_20260128_223720")

# Find all metrics.json files
metrics_files = list(grid_dir.glob("*/metrics.json"))
if PRINT_TEXT:
    print(f"Found {len(metrics_files)} completed runs with metrics")

# Load all metrics into a list of dicts
records = []
skipped = 0
for mf in metrics_files:
    with open(mf) as f:
        data = json.load(f)

    # Skip runs with missing sequential_meta data
    if data.get("sequential_meta") is None:
        skipped += 1
        continue

    # Flatten the nested structure
    record = {
        "run_id": mf.parent.name,
        # Config params
        "lambda2": data["config"]["lambda2"],
        "sigma_sq": data["config"]["sigma_sq"],
        "meta_dict_size": data["config"]["meta_dict_size"],
        "primary_sae_type": data["config"]["primary_sae_type"],
        "meta_sae_type": data["config"]["meta_sae_type"],
        # Joint metrics
        "joint_l2": data["joint"]["l2"],
        "joint_l0": data["joint"]["l0"],
        "joint_loss": data["joint"]["loss"],
        "joint_decomp": data["joint"]["decomp"],
        # Solo metrics
        "solo_l2": data["solo"]["l2"],
        "solo_l0": data["solo"]["l0"],
        "solo_loss": data["solo"]["loss"],
        # Sequential meta metrics
        "seq_meta_l2": data["sequential_meta"]["l2"],
        "seq_meta_l0": data["sequential_meta"]["l0"],
        "seq_meta_loss": data["sequential_meta"]["loss"],
    }
    records.append(record)

if skipped > 0 and PRINT_TEXT:
    print(f"Skipped {skipped} runs with missing sequential_meta data")

df_all = pd.DataFrame(records)

# Derive joint_meta_l2 from joint_decomp
# decomp = exp(-error/σ²) → error = -σ² * ln(decomp)
# Note: This is approximate since decomp is avg(exp(-e_i/σ²)), not exp(-avg(e_i)/σ²)
# But it gives us a usable proxy for the meta-SAE reconstruction error
df_all["joint_meta_l2_approx"] = -df_all["sigma_sq"] * np.log(df_all["joint_decomp"].clip(lower=1e-10))

if PRINT_TEXT:
    print(f"\nFull DataFrame shape: {df_all.shape}")

# =============================================================================
# CREATE BOTH DATASETS
# =============================================================================
# df_all = all runs (JumpReLU L0≈640 + BatchTopK L0=32) - NOT directly comparable
# df_batchtopk = BatchTopK only (consistent L0=32)

df_batchtopk = df_all[df_all["primary_sae_type"] == "batchtopk"].copy()
df_jumprelu = df_all[df_all["primary_sae_type"] == "jumprelu"].copy()

if PRINT_TEXT:
    print(f"All runs: {len(df_all)}")
    print(f"  BatchTopK primary: {len(df_batchtopk)} runs (L0=32)")
    print(f"  JumpReLU primary: {len(df_jumprelu)} runs (L0≈640)")

# For backward compatibility with rest of script, df = batchtopk filtered
df = df_batchtopk

# %%
# =============================================================================
# 2. DATA OVERVIEW - Hyperparameter value counts
# =============================================================================

if PRINT_TEXT:
    print("=" * 60)
    print("HYPERPARAMETER VALUE COUNTS")
    print("=" * 60)

if PRINT_TEXT:
    for col in ["lambda2", "sigma_sq", "meta_dict_size", "primary_sae_type", "meta_sae_type"]:
        print(f"\n{col}:")
        print(df[col].value_counts().sort_index())

# Expected total combinations
n_lambda2 = df["lambda2"].nunique()
n_sigma_sq = df["sigma_sq"].nunique()
n_meta_dict = df["meta_dict_size"].nunique()
n_primary_type = df["primary_sae_type"].nunique()
n_meta_type = df["meta_sae_type"].nunique()
expected = n_lambda2 * n_sigma_sq * n_meta_dict * n_primary_type * n_meta_type
if PRINT_TEXT:
    print(f"\nExpected combinations: {n_lambda2} x {n_sigma_sq} x {n_meta_dict} x {n_primary_type} x {n_meta_type} = {expected}")
    print(f"Actual runs: {len(df)}")

# %%
# =============================================================================
# 3. SUMMARY STATISTICS
# =============================================================================

if PRINT_TEXT:
    print("=" * 60)
    print("SUMMARY STATISTICS FOR KEY METRICS")
    print("=" * 60)

key_metrics = ["joint_l2", "solo_l2", "seq_meta_l2", "joint_l0", "solo_l0", "joint_decomp"]
if PRINT_TEXT:
    print(df[key_metrics].describe().round(4))

# %%
# =============================================================================
# 4. PRIMARY SAE QUALITY: Joint vs Solo
# =============================================================================
#
# Key question: Does adding the composability penalty (lambda2 > 0) hurt
# the primary SAE's reconstruction quality?

# Compute degradation metrics
df["l2_degradation"] = df["joint_l2"] - df["solo_l2"]
df["l2_degradation_pct"] = 100 * (df["joint_l2"] - df["solo_l2"]) / df["solo_l2"]

if PRINT_TEXT:
    print("=" * 60)
    print("PRIMARY SAE QUALITY: Joint vs Solo")
    print("=" * 60)
    print(f"\nL2 Degradation (joint - solo):")
    print(f"  Mean: {df['l2_degradation'].mean():.6f}")
    print(f"  Std:  {df['l2_degradation'].std():.6f}")
    print(f"  Min:  {df['l2_degradation'].min():.6f}")
    print(f"  Max:  {df['l2_degradation'].max():.6f}")
    print(f"\nL2 Degradation %:")
    print(f"  Mean: {df['l2_degradation_pct'].mean():.2f}%")
    print(f"  Std:  {df['l2_degradation_pct'].std():.2f}%")

# %%
# Plot: Joint L2 vs Solo L2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter with y=x reference
ax = axes[0]
scatter = ax.scatter(df["solo_l2"], df["joint_l2"], c=df["lambda2"],
                     cmap="viridis", alpha=0.7, edgecolors='white', linewidth=0.5)
# Reference line y=x
lims = [min(df["solo_l2"].min(), df["joint_l2"].min()) * 0.95,
        max(df["solo_l2"].max(), df["joint_l2"].max()) * 1.05]
ax.plot(lims, lims, 'r--', alpha=0.5, label='y=x (no degradation)')
ax.set_xlabel("Solo L2 (baseline)")
ax.set_ylabel("Joint L2 (with penalty)")
ax.set_title("Primary SAE Reconstruction: Joint vs Solo")
ax.legend()
plt.colorbar(scatter, ax=ax, label="lambda2")

# Degradation % by lambda2
ax = axes[1]
for lam in sorted(df["lambda2"].unique()):
    subset = df[df["lambda2"] == lam]
    ax.hist(subset["l2_degradation_pct"], bins=20, alpha=0.5, label=f"λ2={lam}")
ax.axvline(0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel("L2 Degradation %")
ax.set_ylabel("Count")
ax.set_title("Distribution of Reconstruction Degradation")
ax.legend()

plt.tight_layout()
plt.savefig("../outputs/analysis_joint_vs_solo.png", dpi=150)
plt.show()

# %%
# =============================================================================
# 5. COMPOSABILITY METRICS
# =============================================================================
#
# Sequential meta L2: How well can a meta-SAE reconstruct the primary SAE's
# decoder vectors AFTER training? Higher = harder to reconstruct = less
# composable = potentially more monosemantic.

if PRINT_TEXT:
    print("=" * 60)
    print("COMPOSABILITY METRICS: Sequential Meta-SAE L2")
    print("=" * 60)

    print("\nSequential Meta L2 by lambda2:")
    print(df.groupby("lambda2")["seq_meta_l2"].agg(["mean", "std", "min", "max"]).round(4))

    print("\nSequential Meta L2 by sigma_sq:")
    print(df.groupby("sigma_sq")["seq_meta_l2"].agg(["mean", "std", "min", "max"]).round(4))

# %%
# Plot: Sequential Meta L2 distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By lambda2
ax = axes[0]
lambda2_vals = sorted(df["lambda2"].unique())
data_by_lambda = [df[df["lambda2"] == lam]["seq_meta_l2"].values for lam in lambda2_vals]
bp = ax.boxplot(data_by_lambda, labels=[str(l) for l in lambda2_vals], patch_artist=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(lambda2_vals)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xlabel("lambda2")
ax.set_ylabel("Sequential Meta L2")
ax.set_title("Meta-SAE Reconstruction Error by lambda2\n(Higher = Less Composable)")

# By sigma_sq
ax = axes[1]
sigma_vals = sorted(df["sigma_sq"].unique())
data_by_sigma = [df[df["sigma_sq"] == sig]["seq_meta_l2"].values for sig in sigma_vals]
bp = ax.boxplot(data_by_sigma, labels=[str(s) for s in sigma_vals], patch_artist=True)
colors = plt.cm.plasma(np.linspace(0, 1, len(sigma_vals)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xlabel("sigma_sq")
ax.set_ylabel("Sequential Meta L2")
ax.set_title("Meta-SAE Reconstruction Error by sigma_sq")

plt.tight_layout()
plt.savefig("../outputs/analysis_composability.png", dpi=150)
plt.show()

# %%
# =============================================================================
# 6. THE CORE TRADEOFF: Joint Primary L2 vs Joint Meta L2 (PRIMARY GRAPH)
# =============================================================================
#
# This is the KEY tradeoff plot - both metrics come from joint training.
# joint_l2 = how well primary SAE reconstructs activations
# joint_meta_l2_approx = how well meta-SAE reconstructs primary decoder vectors
#   (derived from decomp: error ≈ -σ² * ln(decomp))
#
# We want: LOW joint_l2 (good reconstruction) AND HIGH joint_meta_l2 (hard to compose)
# Ideal runs are in the BOTTOM-RIGHT of the plot.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Color by lambda2
ax = axes[0]
scatter = ax.scatter(df["joint_l2"], df["joint_meta_l2_approx"],
                     c=df["lambda2"], cmap="viridis",
                     alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel("Joint Primary L2 (lower = better reconstruction)")
ax.set_ylabel("Joint Meta L2 (approx) (higher = less composable)")
ax.set_title("JOINT Training: Primary L2 vs Meta L2\n(Colored by lambda2)")
plt.colorbar(scatter, ax=ax, label="lambda2")

# Add arrow showing ideal direction (bottom-right)
ax.annotate('', xy=(df["joint_l2"].min(), df["joint_meta_l2_approx"].max()),
            xytext=(df["joint_l2"].max(), df["joint_meta_l2_approx"].min()),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.5))
ax.text(df["joint_l2"].mean(), df["joint_meta_l2_approx"].max() * 0.95,
        "← Ideal direction", color='red', fontsize=10, alpha=0.7)

# Color by meta_dict_size
ax = axes[1]
scatter = ax.scatter(df["joint_l2"], df["joint_meta_l2_approx"],
                     c=df["meta_dict_size"], cmap="coolwarm",
                     alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel("Joint Primary L2 (lower = better reconstruction)")
ax.set_ylabel("Joint Meta L2 (approx) (higher = less composable)")
ax.set_title("JOINT Training: Primary L2 vs Meta L2\n(Colored by meta_dict_size)")
plt.colorbar(scatter, ax=ax, label="meta_dict_size")

plt.tight_layout()
plt.savefig("../outputs/batchtopk_tradeoff_joint.png", dpi=150)
plt.show()

# %%
# =============================================================================
# 6a-ALL. SAME PLOT FOR ALL DATA (JumpReLU + BatchTopK)
# =============================================================================
# NOTE: L0 differs dramatically between architectures, so this is for reference only

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
scatter = ax.scatter(df_all["joint_l2"], df_all["joint_meta_l2_approx"],
                     c=df_all["lambda2"], cmap="viridis",
                     alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel("Joint Primary L2 (lower = better reconstruction)")
ax.set_ylabel("Joint Meta L2 (approx) (higher = less composable)")
ax.set_title("ALL DATA: Primary L2 vs Meta L2\n(Colored by lambda2)\n[L0 varies: BatchTopK=32, JumpReLU≈640]")
plt.colorbar(scatter, ax=ax, label="lambda2")

ax = axes[1]
# Color by primary_sae_type to show the two clusters
colors = df_all["primary_sae_type"].map({"batchtopk": 0, "jumprelu": 1})
scatter = ax.scatter(df_all["joint_l2"], df_all["joint_meta_l2_approx"],
                     c=colors, cmap="Set1",
                     alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel("Joint Primary L2 (lower = better reconstruction)")
ax.set_ylabel("Joint Meta L2 (approx) (higher = less composable)")
ax.set_title("ALL DATA: Primary L2 vs Meta L2\n(Colored by primary_sae_type)\n[Red=BatchTopK L0=32, Blue=JumpReLU L0≈640]")

plt.tight_layout()
plt.savefig("../outputs/all_tradeoff_joint.png", dpi=150)
plt.show()

# %%
# Also show joint_decomp directly for reference (BatchTopK only)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Color by lambda2
ax = axes[0]
scatter = ax.scatter(df["joint_l2"], df["joint_decomp"],
                     c=df["lambda2"], cmap="viridis",
                     alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel("Joint Primary L2 (lower = better reconstruction)")
ax.set_ylabel("Joint Decomp Penalty (lower = harder to compose)")
ax.set_title("JOINT Training: Primary L2 vs Decomp Penalty\n(Colored by lambda2)")
plt.colorbar(scatter, ax=ax, label="lambda2")

# Color by sigma_sq (important since decomp depends on sigma_sq)
ax = axes[1]
scatter = ax.scatter(df["joint_l2"], df["joint_decomp"],
                     c=df["sigma_sq"], cmap="plasma",
                     alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel("Joint Primary L2 (lower = better reconstruction)")
ax.set_ylabel("Joint Decomp Penalty (lower = harder to compose)")
ax.set_title("JOINT Training: Primary L2 vs Decomp Penalty\n(Colored by sigma_sq)")
plt.colorbar(scatter, ax=ax, label="sigma_sq")

plt.tight_layout()
plt.savefig("../outputs/batchtopk_tradeoff_decomp.png", dpi=150)
plt.show()

# %%
# =============================================================================
# 6b. SEQUENTIAL TRADEOFF (for comparison)
# =============================================================================
#
# This compares solo primary SAE with the meta-SAE trained on it sequentially.
# Both metrics come from the sequential pipeline.
# Higher seq_meta_l2 = meta-SAE struggles to reconstruct = less composable

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Color by lambda2 (note: lambda2 shouldn't affect sequential training much)
ax = axes[0]
scatter = ax.scatter(df["solo_l2"], df["seq_meta_l2"],
                     c=df["lambda2"], cmap="viridis",
                     alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel("Solo Primary L2 (lower = better reconstruction)")
ax.set_ylabel("Sequential Meta L2 (higher = less composable)")
ax.set_title("SEQUENTIAL Training: Reconstruction vs Composability\n(Colored by lambda2 - should have no effect)")
plt.colorbar(scatter, ax=ax, label="lambda2")

# Color by meta_dict_size
ax = axes[1]
scatter = ax.scatter(df["solo_l2"], df["seq_meta_l2"],
                     c=df["meta_dict_size"], cmap="coolwarm",
                     alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel("Solo Primary L2 (lower = better reconstruction)")
ax.set_ylabel("Sequential Meta L2 (higher = less composable)")
ax.set_title("SEQUENTIAL Training\n(Colored by meta_dict_size)")
plt.colorbar(scatter, ax=ax, label="meta_dict_size")

plt.tight_layout()
plt.savefig("../outputs/analysis_tradeoff_sequential.png", dpi=150)
plt.show()

# %%
# =============================================================================
# 7. HYPERPARAMETER EFFECTS - Detailed breakdown
# =============================================================================

def plot_param_effects(df, param, metrics, figsize=(15, 4)):
    """Plot effect of a hyperparameter on multiple metrics."""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    param_vals = sorted(df[param].unique())

    for ax, metric in zip(axes, metrics):
        data = [df[df[param] == val][metric].values for val in param_vals]
        bp = ax.boxplot(data, labels=[str(v) for v in param_vals], patch_artist=True)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(param_vals)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by {param}")

    plt.tight_layout()
    return fig

# %%
# Effect of lambda2 on JOINT training metrics (the correct comparison)
if PRINT_TEXT:
    print("=" * 60)
    print("EFFECT OF LAMBDA2 ON JOINT TRAINING")
    print("=" * 60)

fig = plot_param_effects(df, "lambda2", ["joint_l2", "joint_meta_l2_approx", "joint_decomp"])
plt.savefig("../outputs/batchtopk_lambda2_effects.png", dpi=150)
plt.show()

if PRINT_TEXT:
    print("\nMean JOINT metrics by lambda2:")
    print(df.groupby("lambda2")[["joint_l2", "joint_meta_l2_approx", "joint_decomp"]].mean().round(4))

# %%
# Effect of sigma_sq on JOINT training metrics
if PRINT_TEXT:
    print("=" * 60)
    print("EFFECT OF SIGMA_SQ ON JOINT TRAINING")
    print("=" * 60)

fig = plot_param_effects(df, "sigma_sq", ["joint_l2", "joint_meta_l2_approx", "joint_decomp"])
plt.savefig("../outputs/batchtopk_sigma_sq_effects.png", dpi=150)
plt.show()

if PRINT_TEXT:
    print("\nMean JOINT metrics by sigma_sq:")
    print(df.groupby("sigma_sq")[["joint_l2", "joint_meta_l2_approx", "joint_decomp"]].mean().round(4))

# %%
# Effect of meta_dict_size
if PRINT_TEXT:
    print("=" * 60)
    print("EFFECT OF META_DICT_SIZE")
    print("=" * 60)

fig = plot_param_effects(df, "meta_dict_size", ["joint_l2", "seq_meta_l2", "seq_meta_l0"])
plt.savefig("../outputs/batchtopk_meta_dict_size_effects.png", dpi=150)
plt.show()

if PRINT_TEXT:
    print("\nMean metrics by meta_dict_size:")
    print(df.groupby("meta_dict_size")[["joint_l2", "solo_l2", "seq_meta_l2", "seq_meta_l0"]].mean().round(4))

# %%
# Effect of META SAE architecture (primary is all BatchTopK now)
if PRINT_TEXT:
    print("=" * 60)
    print("EFFECT OF META SAE ARCHITECTURE (Primary = BatchTopK)")
    print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Meta SAE type effect on joint metrics
ax = axes[0]
for mtype in df["meta_sae_type"].unique():
    subset = df[df["meta_sae_type"] == mtype]
    ax.hist(subset["joint_l2"], bins=15, alpha=0.5, label=mtype)
ax.set_xlabel("Joint Primary L2")
ax.set_ylabel("Count")
ax.set_title("Meta SAE Type: Effect on Primary L2")
ax.legend()

ax = axes[1]
for mtype in df["meta_sae_type"].unique():
    subset = df[df["meta_sae_type"] == mtype]
    ax.hist(subset["joint_meta_l2_approx"], bins=15, alpha=0.5, label=mtype)
ax.set_xlabel("Joint Meta L2 (approx)")
ax.set_ylabel("Count")
ax.set_title("Meta SAE Type: Effect on Meta L2")
ax.legend()

plt.tight_layout()
plt.savefig("../outputs/batchtopk_meta_architecture_effects.png", dpi=150)
plt.show()

if PRINT_TEXT:
    print("\nMean JOINT metrics by meta_sae_type (primary=BatchTopK):")
    print(df.groupby("meta_sae_type")[["joint_l2", "joint_meta_l2_approx", "joint_decomp"]].mean().round(4))

# %%
# =============================================================================
# 8. INTERACTION EFFECTS: lambda2 x sigma_sq (JOINT METRICS)
# =============================================================================

if PRINT_TEXT:
    print("=" * 60)
    print("INTERACTION: LAMBDA2 x SIGMA_SQ (JOINT TRAINING)")
    print("=" * 60)

# Create pivot tables
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Joint L2
pivot_joint = df.pivot_table(values="joint_l2", index="lambda2", columns="sigma_sq", aggfunc="mean")
ax = axes[0]
sns.heatmap(pivot_joint, annot=True, fmt=".4f", cmap="RdYlGn_r", ax=ax)
ax.set_title("Joint Primary L2 (lower = better)")

# Joint Meta L2 (the key metric!)
pivot_meta = df.pivot_table(values="joint_meta_l2_approx", index="lambda2", columns="sigma_sq", aggfunc="mean")
ax = axes[1]
sns.heatmap(pivot_meta, annot=True, fmt=".4f", cmap="RdYlGn", ax=ax)
ax.set_title("Joint Meta L2 (higher = less composable = better)")

# Joint Decomp
pivot_decomp = df.pivot_table(values="joint_decomp", index="lambda2", columns="sigma_sq", aggfunc="mean")
ax = axes[2]
sns.heatmap(pivot_decomp, annot=True, fmt=".4f", cmap="RdYlGn_r", ax=ax)
ax.set_title("Joint Decomp (lower = less composable)")

plt.tight_layout()
plt.savefig("../outputs/batchtopk_lambda_sigma_interaction.png", dpi=150)
plt.show()

# %%
# =============================================================================
# 9. META SAE TYPE COMPARISON (Primary = BatchTopK)
# =============================================================================

if PRINT_TEXT:
    print("=" * 60)
    print("META SAE TYPE COMPARISON (Primary = BatchTopK)")
    print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Joint L2 by meta SAE type
ax = axes[0]
meta_types = sorted(df["meta_sae_type"].unique())
data = [df[df["meta_sae_type"] == mt]["joint_l2"].values for mt in meta_types]
bp = ax.boxplot(data, tick_labels=meta_types, patch_artist=True)
colors = plt.cm.Set2(np.linspace(0, 1, len(meta_types)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xlabel("Meta SAE Type")
ax.set_ylabel("Joint Primary L2")
ax.set_title("Primary Reconstruction by Meta SAE Type")

# Joint Meta L2 by meta SAE type
ax = axes[1]
data = [df[df["meta_sae_type"] == mt]["joint_meta_l2_approx"].values for mt in meta_types]
bp = ax.boxplot(data, tick_labels=meta_types, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xlabel("Meta SAE Type")
ax.set_ylabel("Joint Meta L2 (approx)")
ax.set_title("Meta-SAE Reconstruction Error by Meta SAE Type")

plt.tight_layout()
plt.savefig("../outputs/batchtopk_meta_type_comparison.png", dpi=150)
plt.show()

if PRINT_TEXT:
    print("\nMean JOINT metrics by meta_sae_type (primary=BatchTopK):")
    print(df.groupby("meta_sae_type")[["joint_l2", "joint_meta_l2_approx", "joint_decomp"]].mean().round(4))

# %%
# =============================================================================
# 10. PARETO FRONTIER ANALYSIS (JOINT TRAINING)
# =============================================================================
#
# Find runs that are Pareto-optimal for the JOINT training tradeoff.
# We want LOW joint_l2 AND HIGH joint_meta_l2 (low primary error, high meta error).

if PRINT_TEXT:
    print("=" * 60)
    print("PARETO FRONTIER ANALYSIS (JOINT TRAINING)")
    print("=" * 60)

def is_pareto_optimal(costs):
    """
    Find Pareto-optimal points.
    costs: (n_points, n_objectives) array where LOWER is better for all objectives
    Returns: boolean array of length n_points
    """
    is_optimal = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_optimal[i]:
            # Keep points that are not dominated by point i
            is_optimal[is_optimal] = np.any(costs[is_optimal] < c, axis=1) | np.all(costs[is_optimal] == c, axis=1)
            is_optimal[i] = True
    return is_optimal

# For Pareto: we want LOW joint_l2 and HIGH joint_meta_l2_approx
# So we minimize [joint_l2, -joint_meta_l2_approx]
costs = np.column_stack([df["joint_l2"].values, -df["joint_meta_l2_approx"].values])
pareto_mask = is_pareto_optimal(costs)

df["is_pareto_joint"] = pareto_mask
pareto_df = df[df["is_pareto_joint"]].copy()

if PRINT_TEXT:
    print(f"\nFound {pareto_mask.sum()} Pareto-optimal runs out of {len(df)}")

# Plot Pareto frontier
fig, ax = plt.subplots(figsize=(10, 7))

# Non-Pareto points
non_pareto = df[~df["is_pareto_joint"]]
ax.scatter(non_pareto["joint_l2"], non_pareto["joint_meta_l2_approx"],
           c='lightgray', alpha=0.5, s=40, label="Dominated")

# Pareto points colored by lambda2
scatter = ax.scatter(pareto_df["joint_l2"], pareto_df["joint_meta_l2_approx"],
                     c=pareto_df["lambda2"], cmap="viridis",
                     s=100, edgecolors='red', linewidth=2, label="Pareto-optimal")
plt.colorbar(scatter, ax=ax, label="lambda2")

# Connect Pareto points with line (sorted by joint_l2)
pareto_sorted = pareto_df.sort_values("joint_l2")
ax.plot(pareto_sorted["joint_l2"], pareto_sorted["joint_meta_l2_approx"],
        'r--', alpha=0.5, linewidth=1)

ax.set_xlabel("Joint Primary L2 (lower = better reconstruction)")
ax.set_ylabel("Joint Meta L2 (approx) (higher = less composable)")
ax.set_title("Pareto Frontier: JOINT Training\n(Ideal = bottom-right)")
ax.legend()

plt.tight_layout()
plt.savefig("../outputs/batchtopk_pareto.png", dpi=150)
plt.show()

# %%
# Show Pareto-optimal runs
if PRINT_TEXT:
    print("\nPareto-optimal runs (sorted by joint_l2):")
pareto_cols = ["run_id", "lambda2", "sigma_sq", "meta_dict_size",
               "primary_sae_type", "meta_sae_type", "joint_l2", "joint_meta_l2_approx"]
if PRINT_TEXT:
    print(pareto_sorted[pareto_cols].to_string(index=False))

# %%
# =============================================================================
# 11. BEST RUNS BY DIFFERENT CRITERIA
# =============================================================================

if PRINT_TEXT:
    print("=" * 60)
    print("BEST RUNS BY DIFFERENT CRITERIA")
    print("=" * 60)

display_cols = ["run_id", "lambda2", "sigma_sq", "meta_dict_size",
                "primary_sae_type", "meta_sae_type",
                "joint_l2", "seq_meta_l2", "l2_degradation_pct"]

# Best by lowest joint_l2 (best reconstruction)
if PRINT_TEXT:
    print("\n--- Top 5 by LOWEST Joint L2 (best reconstruction) ---")
    print(df.nsmallest(5, "joint_l2")[display_cols].to_string(index=False))

# Best by highest seq_meta_l2 (least composable)
if PRINT_TEXT:
    print("\n--- Top 5 by HIGHEST Sequential Meta L2 (least composable) ---")
    print(df.nlargest(5, "seq_meta_l2")[display_cols].to_string(index=False))

# Best by lowest degradation (joint training doesn't hurt)
if PRINT_TEXT:
    print("\n--- Top 5 by LOWEST L2 Degradation % (joint ≈ solo) ---")
    print(df.nsmallest(5, "l2_degradation_pct")[display_cols].to_string(index=False))

# %%
# =============================================================================
# 12. COMPOSITE SCORE RANKING
# =============================================================================
#
# Create a composite score that balances both objectives.
# Since we want LOW joint_l2 and HIGH seq_meta_l2, we normalize each
# and create: score = (1 - norm_joint_l2) + norm_seq_meta_l2

if PRINT_TEXT:
    print("=" * 60)
    print("COMPOSITE SCORE RANKING")
    print("=" * 60)

# Normalize metrics to [0, 1]
df["norm_joint_l2"] = (df["joint_l2"] - df["joint_l2"].min()) / (df["joint_l2"].max() - df["joint_l2"].min())
df["norm_seq_meta_l2"] = (df["seq_meta_l2"] - df["seq_meta_l2"].min()) / (df["seq_meta_l2"].max() - df["seq_meta_l2"].min())

# Composite score: higher is better
# Weight can be adjusted - here we weight equally
df["composite_score"] = (1 - df["norm_joint_l2"]) + df["norm_seq_meta_l2"]

if PRINT_TEXT:
    print("\n--- Top 10 by COMPOSITE SCORE (equal weight) ---")
top_composite = df.nlargest(10, "composite_score")
if PRINT_TEXT:
    print(top_composite[display_cols + ["composite_score"]].to_string(index=False))

# %%
# Also show with different weightings
if PRINT_TEXT:
    print("\n--- Top 5 emphasizing RECONSTRUCTION (70% joint_l2, 30% meta_l2) ---")
df["composite_recon"] = 0.7 * (1 - df["norm_joint_l2"]) + 0.3 * df["norm_seq_meta_l2"]
if PRINT_TEXT:
    print(df.nlargest(5, "composite_recon")[display_cols].to_string(index=False))

    print("\n--- Top 5 emphasizing COMPOSABILITY (30% joint_l2, 70% meta_l2) ---")
df["composite_decomp"] = 0.3 * (1 - df["norm_joint_l2"]) + 0.7 * df["norm_seq_meta_l2"]
if PRINT_TEXT:
    print(df.nlargest(5, "composite_decomp")[display_cols].to_string(index=False))

# %%
# =============================================================================
# 13. DOES LAMBDA2 ACTUALLY HELP? (JOINT TRAINING METRICS)
# =============================================================================
#
# Key question: Compared to lambda2=0 baseline, does adding the penalty
# actually reduce joint_decomp (make features harder to compose)?

if PRINT_TEXT:
    print("=" * 60)
    print("DOES THE COMPOSABILITY PENALTY (lambda2 > 0) HELP?")
    print("=" * 60)

# Compare lambda2=0 to lambda2>0
baseline = df[df["lambda2"] == 0.0]
with_penalty = df[df["lambda2"] > 0.0]

if PRINT_TEXT:
    print(f"\nBaseline (lambda2=0): {len(baseline)} runs")
    print(f"  Mean joint_l2: {baseline['joint_l2'].mean():.6f}")
    print(f"  Mean joint_meta_l2: {baseline['joint_meta_l2_approx'].mean():.6f}")
    print(f"  Mean joint_decomp: {baseline['joint_decomp'].mean():.6f}")

    print(f"\nWith penalty (lambda2>0): {len(with_penalty)} runs")
    print(f"  Mean joint_l2: {with_penalty['joint_l2'].mean():.6f}")
    print(f"  Mean joint_meta_l2: {with_penalty['joint_meta_l2_approx'].mean():.6f}")
    print(f"  Mean joint_decomp: {with_penalty['joint_decomp'].mean():.6f}")

    print(f"\nDifference:")
    print(f"  joint_l2: {with_penalty['joint_l2'].mean() - baseline['joint_l2'].mean():+.6f} (positive = worse reconstruction)")
    print(f"  joint_meta_l2: {with_penalty['joint_meta_l2_approx'].mean() - baseline['joint_meta_l2_approx'].mean():+.6f} (positive = less composable = GOOD)")
    print(f"  joint_decomp: {with_penalty['joint_decomp'].mean() - baseline['joint_decomp'].mean():+.6f} (negative = less composable = GOOD)")

# %%
# Statistical test
from scipy import stats

# For each metric, do a t-test comparing lambda2=0 vs lambda2>0
if PRINT_TEXT:
    print("\n--- Statistical Tests (t-test) on JOINT metrics ---")

if PRINT_TEXT:
    for metric in ["joint_l2", "joint_meta_l2_approx", "joint_decomp"]:
        t_stat, p_val = stats.ttest_ind(baseline[metric], with_penalty[metric])
        print(f"\n{metric}:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4e}")
        if p_val < 0.05:
            print(f"  -> Statistically significant difference (p < 0.05)")
        else:
            print(f"  -> NOT statistically significant (p >= 0.05)")

# %%
# Breakdown by lambda2 value
if PRINT_TEXT:
    print("\n--- Mean JOINT metrics by lambda2 value ---")
    print(df.groupby("lambda2")[["joint_l2", "joint_meta_l2_approx", "joint_decomp"]].mean().round(4))

# %%
# =============================================================================
# 14. CORRELATION ANALYSIS (JOINT METRICS)
# =============================================================================

if PRINT_TEXT:
    print("=" * 60)
    print("CORRELATION ANALYSIS (JOINT TRAINING)")
    print("=" * 60)

# Correlation matrix for key metrics
# Need to handle categorical columns
df_numeric = df.copy()
df_numeric["primary_sae_type_num"] = (df_numeric["primary_sae_type"] == "jumprelu").astype(int)
df_numeric["meta_sae_type_num"] = (df_numeric["meta_sae_type"] == "jumprelu").astype(int)

corr_cols_full = ["lambda2", "sigma_sq", "meta_dict_size",
                  "primary_sae_type_num", "meta_sae_type_num",
                  "joint_l2", "joint_decomp", "l2_degradation_pct"]

corr_matrix = df_numeric[corr_cols_full].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            xticklabels=corr_cols_full, yticklabels=corr_cols_full, ax=ax)
ax.set_title("Correlation Matrix (Joint Training Metrics)")
plt.tight_layout()
plt.savefig("../outputs/batchtopk_correlation.png", dpi=150)
plt.show()

# %%
# =============================================================================
# 15. SAVE RESULTS TABLE
# =============================================================================

if PRINT_TEXT:
    print("=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

# Save full results to CSV
output_path = "../outputs/grid_search_analysis_batchtopk.csv"
df.to_csv(output_path, index=False)
if PRINT_TEXT:
    print(f"\nBatchTopK results saved to: {output_path}")

# Save Pareto-optimal runs
pareto_path = "../outputs/pareto_optimal_runs_batchtopk.csv"
pareto_df.to_csv(pareto_path, index=False)
if PRINT_TEXT:
    print(f"Pareto-optimal runs saved to: {pareto_path}")

# Also save full dataset for reference
df_all.to_csv("../outputs/grid_search_analysis_all.csv", index=False)
if PRINT_TEXT:
    print(f"Full results (all architectures) saved to: ../outputs/grid_search_analysis_all.csv")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
