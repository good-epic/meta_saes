#!/bin/bash
# =============================================================================
# Meta-SAE Assessment Script
# =============================================================================
# This script evaluates trained SAEs on reconstruction quality, sparsity,
# and functional similarity.
#
# Usage:
#   ./run_scripts/run_assessment.sh              # Run with defaults
#   ./run_scripts/run_assessment.sh --dry-run    # Print command without running
#
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Setup paths
# -----------------------------------------------------------------------------

# Get the directory where this script lives, then find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"

# Change to project root so Python imports work
cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Configuration (should match training settings)
# -----------------------------------------------------------------------------

# Model & Data
LAYER=8
SITE="resid_pre"
DATASET_PATH="HuggingFaceFW/fineweb"
DATASET_NAME="sample-10BT"

# SAE architecture (must match trained models)
DICT_SIZE=12288
META_DICT_SIZE=2048
PRIMARY_TOP_K=32
META_TOP_K=4

# Assessment settings
NUM_ASSESSMENT_BATCHES=1000
NUM_BATCHES_IN_BUFFER=3
MODEL_BATCH_SIZE=256
SEQ_LEN=128

# Checkpoint paths (in OUTPUT_DIR)
JOINT_PRIMARY_PATH="${OUTPUT_DIR}/joint_primary_sae.pt"
JOINT_META_PATH="${OUTPUT_DIR}/joint_meta_sae.pt"
SOLO_PRIMARY_PATH="${OUTPUT_DIR}/solo_primary_sae.pt"
SEQUENTIAL_META_PATH="${OUTPUT_DIR}/sequential_meta_sae.pt"

# -----------------------------------------------------------------------------
# Build command
# -----------------------------------------------------------------------------

CMD="python assess_meta_sae.py \
    --dataset_path ${DATASET_PATH} \
    --dataset_name ${DATASET_NAME} \
    --layer ${LAYER} \
    --site ${SITE} \
    --dict_size ${DICT_SIZE} \
    --meta_dict_size ${META_DICT_SIZE} \
    --primary_top_k ${PRIMARY_TOP_K} \
    --meta_top_k ${META_TOP_K} \
    --num_assessment_batches ${NUM_ASSESSMENT_BATCHES} \
    --num_batches_in_buffer ${NUM_BATCHES_IN_BUFFER} \
    --model_batch_size ${MODEL_BATCH_SIZE} \
    --seq_len ${SEQ_LEN} \
    --joint_primary_path ${JOINT_PRIMARY_PATH} \
    --joint_meta_path ${JOINT_META_PATH} \
    --solo_primary_path ${SOLO_PRIMARY_PATH} \
    --sequential_meta_path ${SEQUENTIAL_META_PATH} \
    --assess_joint_saes \
    --assess_sequential_saes"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

echo "============================================================"
echo "Meta-SAE Assessment"
echo "============================================================"
echo ""
echo "Paths:"
echo "  Project root:    ${PROJECT_ROOT}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo ""
echo "Settings:"
echo "  Layer:              ${LAYER}"
echo "  Primary SAE:        ${DICT_SIZE} features, top-k=${PRIMARY_TOP_K}"
echo "  Meta SAE:           ${META_DICT_SIZE} features, top-k=${META_TOP_K}"
echo "  Assessment batches: ${NUM_ASSESSMENT_BATCHES}"
echo ""
echo "Checkpoints:"
echo "  Joint primary:      ${JOINT_PRIMARY_PATH}"
echo "  Joint meta:         ${JOINT_META_PATH}"
echo "  Solo primary:       ${SOLO_PRIMARY_PATH}"
echo "  Sequential meta:    ${SEQUENTIAL_META_PATH}"
echo ""
echo "============================================================"

if [[ "$1" == "--dry-run" ]]; then
    echo "DRY RUN - Command that would be executed:"
    echo ""
    echo "$CMD"
    echo ""
else
    echo "Starting assessment..."
    echo ""
    eval $CMD
fi
