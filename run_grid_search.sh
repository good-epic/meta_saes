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
#   ./run_grid_search.sh              # Run with defaults
#   ./run_grid_search.sh --dry-run    # Show what would run
#   ./run_grid_search.sh --resume     # Resume incomplete search
#
# =============================================================================

set -e

# Configuration
NUM_WORKERS=${NUM_WORKERS:-8}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/grid_search_$(date +%Y%m%d_%H%M%S)"}

# Check for dry run
DRY_RUN=""
RESUME=""
for arg in "$@"; do
    case $arg in
        --dry-run|--dry_run)
            DRY_RUN="--dry_run"
            ;;
        --resume)
            RESUME="--resume"
            ;;
    esac
done

echo "============================================================"
echo "Meta-SAE Grid Search"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Workers:    ${NUM_WORKERS}"
echo "  Output:     ${OUTPUT_DIR}"
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
    --output_dir "${OUTPUT_DIR}" \
    ${DRY_RUN} \
    ${RESUME}

echo ""
echo "============================================================"
echo "Grid search complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To analyze results, run:"
echo "  python analyze_grid_results.py --input_dir ${OUTPUT_DIR}"
echo "============================================================"
