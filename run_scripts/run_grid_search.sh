#!/bin/bash
# =============================================================================
# Grid Search Runner for Meta-SAE Hyperparameter Tuning
# =============================================================================
#
# This script runs a grid search over meta-SAE hyperparameters.
#
# For RunPod A40 (48GB VRAM, $0.20/hr):
#   - 8 concurrent workers is a safe default
#   - ~144 runs at ~10-15 min each = 3-4.5 hours
#   - Estimated cost: ~$0.60-$0.90
#
# Usage:
#   ./run_scripts/run_grid_search.sh              # Run with defaults
#   ./run_scripts/run_grid_search.sh --dry-run    # Show what would run
#   ./run_scripts/run_grid_search.sh --resume     # Resume incomplete search
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Setup paths
# -----------------------------------------------------------------------------

# Get the directory where this script lives, then find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Change to project root so Python imports work
cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

NUM_WORKERS=${NUM_WORKERS:-8}
GRID_OUTPUT_DIR="${OUTPUT_DIR}/grid_search_$(date +%Y%m%d_%H%M%S)"

# Check for flags
DRY_RUN=""
RESUME=""
CUSTOM_OUTPUT=""

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
    esac
done

# Use custom output dir if provided, otherwise use timestamped default
if [[ -n "$CUSTOM_OUTPUT" ]]; then
    GRID_OUTPUT_DIR="$CUSTOM_OUTPUT"
fi

# If resuming without custom output, try to find the most recent grid search
if [[ -n "$RESUME" && -z "$CUSTOM_OUTPUT" ]]; then
    LATEST_GRID=$(ls -dt "${OUTPUT_DIR}"/grid_search_* 2>/dev/null | head -1)
    if [[ -n "$LATEST_GRID" ]]; then
        GRID_OUTPUT_DIR="$LATEST_GRID"
        echo "Resuming from: $GRID_OUTPUT_DIR"
    fi
fi

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

echo "============================================================"
echo "Meta-SAE Grid Search"
echo "============================================================"
echo ""
echo "Paths:"
echo "  Project root:    ${PROJECT_ROOT}"
echo "  Output dir:      ${GRID_OUTPUT_DIR}"
echo ""
echo "Configuration:"
echo "  Workers:    ${NUM_WORKERS}"
echo ""
echo "Grid (Phase 1):"
echo "  lambda2:           [0.0, 0.001, 0.01, 0.1]"
echo "  sigma_sq:          [0.01, 0.1, 1.0]"
echo "  meta_dict_size:    [512, 1024, 2048]"
echo "  primary_sae_type:  [batchtopk, jumprelu]"
echo "  meta_sae_type:     [batchtopk, jumprelu]"
echo ""
echo "Total runs: 4 x 3 x 3 x 2 x 2 = 144"
echo "============================================================"
echo ""

if [[ -n "$DRY_RUN" ]]; then
    echo "DRY RUN MODE"
    echo ""
fi

# Run the grid search
python grid_search.py \
    --num_workers ${NUM_WORKERS} \
    --output_dir "${GRID_OUTPUT_DIR}" \
    ${DRY_RUN} \
    ${RESUME}

echo ""
echo "============================================================"
echo "Grid search complete!"
echo "Results saved to: ${GRID_OUTPUT_DIR}"
echo ""
echo "To analyze results, run:"
echo "  python analyze_grid_results.py --input_dir ${GRID_OUTPUT_DIR}"
echo "============================================================"
