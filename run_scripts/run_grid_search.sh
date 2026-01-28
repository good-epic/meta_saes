#!/bin/bash
# =============================================================================
# Grid Search Runner for Meta-SAE Hyperparameter Tuning
# =============================================================================
#
# This script runs a grid search over meta-SAE hyperparameters.
#
# Usage:
#   ./run_scripts/run_grid_search.sh                    # Run with new defaults
#   ./run_scripts/run_grid_search.sh --dry-run          # Show what would run
#   ./run_scripts/run_grid_search.sh --resume           # Resume incomplete search
#
# Environment variables (can override):
#   NUM_WORKERS=2                    # Concurrent training jobs
#   MODEL_NAME=gpt2-large            # Model (gpt2-small, gpt2-medium, gpt2-large)
#   LAYER=20                         # Model layer to hook
#   DICT_SIZE=20480                  # Primary SAE dictionary size
#   META_DICT_SIZE=1800              # Meta SAE dictionary size
#   PRIMARY_TOP_K=64                 # Top-K / target L0
#   TARGET_L0=64                     # Dynamic L0 target for JumpReLU
#   BANDWIDTH=0.0001                 # JumpReLU bandwidth (lower = steeper)
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Setup paths
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------

NUM_WORKERS=${NUM_WORKERS:-2}

# Model settings
MODEL_NAME=${MODEL_NAME:-gpt2-large}
LAYER=${LAYER:-20}  # ~56% through GPT-2 Large (36 layers)

# SAE settings
DICT_SIZE=${DICT_SIZE:-20480}  # 16x residual stream (1280)
META_DICT_SIZE=${META_DICT_SIZE:-1800}
PRIMARY_TOP_K=${PRIMARY_TOP_K:-64}
META_TOP_K=${META_TOP_K:-8}
BANDWIDTH=${BANDWIDTH:-0.0001}
NUM_TOKENS=${NUM_TOKENS:-100000000}
TARGET_L0=${TARGET_L0:-64}  # Dynamic L0 targeting for JumpReLU

# JumpReLU threshold (leave empty to use default initialization)
JUMPRELU_INIT_THRESHOLD=${JUMPRELU_INIT_THRESHOLD:-}

# Grid parameters (defaults in grid_search.py will be used if not overridden)
# lambda2: [0.0, 0.01, 0.1, 1.0]
# sigma_sq: [0.1]
# meta_dict_size: [1024]
# primary_sae_type: [batchtopk, jumprelu]
# meta_sae_type: [batchtopk, jumprelu]

# -----------------------------------------------------------------------------
# Parse command line arguments
# -----------------------------------------------------------------------------

DRY_RUN=""
RESUME=""
CUSTOM_OUTPUT=""
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --dry-run|--dry_run)
            DRY_RUN="--dry_run"
            ;;
        --resume)
            RESUME="--resume"
            ;;
        --output_dir=*)
            CUSTOM_OUTPUT="${arg#*=}"
            ;;
        *)
            # Pass through other arguments to grid_search.py
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# Output directory
GRID_OUTPUT_DIR="${OUTPUT_DIR}/grid_search_$(date +%Y%m%d_%H%M%S)"
if [[ -n "$CUSTOM_OUTPUT" ]]; then
    GRID_OUTPUT_DIR="$CUSTOM_OUTPUT"
fi

# If resuming without custom output, find most recent grid search
if [[ -n "$RESUME" && -z "$CUSTOM_OUTPUT" ]]; then
    LATEST_GRID=$(ls -dt "${OUTPUT_DIR}"/grid_search_* 2>/dev/null | head -1)
    if [[ -n "$LATEST_GRID" ]]; then
        GRID_OUTPUT_DIR="$LATEST_GRID"
        echo "Resuming from: $GRID_OUTPUT_DIR"
    fi
fi

# -----------------------------------------------------------------------------
# Build command
# -----------------------------------------------------------------------------

CMD="python grid_search.py"
CMD="$CMD --num_workers ${NUM_WORKERS}"
CMD="$CMD --output_dir ${GRID_OUTPUT_DIR}"
CMD="$CMD --model_name ${MODEL_NAME}"
CMD="$CMD --layer ${LAYER}"
CMD="$CMD --dict_size ${DICT_SIZE}"
CMD="$CMD --meta_dict_size ${META_DICT_SIZE}"
CMD="$CMD --primary_top_k ${PRIMARY_TOP_K}"
CMD="$CMD --meta_top_k ${META_TOP_K}"
CMD="$CMD --bandwidth ${BANDWIDTH}"
CMD="$CMD --num_tokens ${NUM_TOKENS}"
CMD="$CMD --target_l0 ${TARGET_L0}"

if [[ -n "$JUMPRELU_INIT_THRESHOLD" ]]; then
    CMD="$CMD --jumprelu_init_threshold ${JUMPRELU_INIT_THRESHOLD}"
fi

if [[ -n "$DRY_RUN" ]]; then
    CMD="$CMD --dry_run"
fi

if [[ -n "$RESUME" ]]; then
    CMD="$CMD --resume"
fi

CMD="$CMD $EXTRA_ARGS"

# -----------------------------------------------------------------------------
# Display configuration and run
# -----------------------------------------------------------------------------

echo "============================================================"
echo "Meta-SAE Grid Search (GPT-2 Large)"
echo "============================================================"
echo ""
echo "Paths:"
echo "  Project root:    ${PROJECT_ROOT}"
echo "  Output dir:      ${GRID_OUTPUT_DIR}"
echo ""
echo "Model settings:"
echo "  model_name:      ${MODEL_NAME}"
echo "  layer:           ${LAYER}"
echo ""
echo "SAE settings:"
echo "  dict_size:       ${DICT_SIZE}"
echo "  meta_dict_size:  ${META_DICT_SIZE}"
echo "  primary_top_k:   ${PRIMARY_TOP_K}"
echo "  meta_top_k:      ${META_TOP_K}"
echo "  bandwidth:       ${BANDWIDTH}"
echo "  target_l0:       ${TARGET_L0}"
if [[ -n "$JUMPRELU_INIT_THRESHOLD" ]]; then
echo "  jumprelu_init:   ${JUMPRELU_INIT_THRESHOLD}"
fi
echo ""
echo "Training:"
echo "  num_tokens:      ${NUM_TOKENS}"
echo "  num_workers:     ${NUM_WORKERS}"
echo ""
echo "Grid (defaults):"
echo "  lambda2:         [0.0, 0.01, 0.1, 1.0]"
echo "  sigma_sq:        [0.1, 1.0]"
echo "  meta_dict_size:  [${META_DICT_SIZE}]"
echo "  primary/meta:    [batchtopk, jumprelu] (matched)"
echo ""
echo "Total runs: 4 x 2 x 1 x 2 = 16 (matched architectures)"
echo "Workers: ${NUM_WORKERS} concurrent jobs"
echo "============================================================"
echo ""
echo "Command: $CMD"
echo ""

if [[ -n "$DRY_RUN" ]]; then
    echo "DRY RUN MODE - no jobs will be started"
    echo ""
fi

# Run
eval $CMD

echo ""
echo "============================================================"
echo "Grid search complete!"
echo "Results saved to: ${GRID_OUTPUT_DIR}"
echo ""
echo "To analyze results:"
echo "  cd scratch && python decoder_comparison.py"
echo "============================================================"
