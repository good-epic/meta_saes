#!/bin/bash
# =============================================================================
# Meta-SAE Training Script
# =============================================================================
# This script trains SAEs with the decomposability penalty to encourage
# atomic feature learning.
#
# Usage:
#   ./run_scripts/run_training.sh              # Run with defaults
#   ./run_scripts/run_training.sh --dry-run    # Print command without running
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

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Change to project root so Python imports work
cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Model & Data
MODEL_NAME="gpt2-small"
LAYER=8
SITE="resid_pre"
DATASET_PATH="HuggingFaceFW/fineweb"
DATASET_NAME="sample-10BT"

# Primary SAE architecture
DICT_SIZE=12288          # Number of features in primary SAE
PRIMARY_TOP_K=32         # Sparsity: ~32 active features per activation

# Meta SAE architecture
META_DICT_SIZE=2048      # Smaller dictionary for "atomic" features
META_TOP_K=4             # Very sparse: ~4 meta-features per primary feature

# Training budget
NUM_TOKENS=100000000     # 100M tokens (adjust based on compute budget)
BATCH_SIZE=4096          # Activations per SAE training step
LR=3e-4                  # Learning rate

# Decomposability penalty (THE KEY HYPERPARAMETERS)
LAMBDA2=0.01             # Penalty weight (try: 0.001, 0.01, 0.1)
SIGMA_SQ=0.1             # Penalty bandwidth (try: 0.01, 0.1, 1.0)

# Alternating training schedule
N_PRIMARY_STEPS=10       # Primary SAE steps per cycle
N_META_STEPS=5           # Meta SAE steps per cycle

# Memory management
NUM_BATCHES_IN_BUFFER_JOINT=5
NUM_BATCHES_IN_BUFFER_SEQ=3
MODEL_BATCH_SIZE=256     # Sequences per model forward pass
SEQ_LEN=128              # Tokens per sequence

# Output paths (relative to OUTPUT_DIR)
JOINT_PRIMARY_PATH="${OUTPUT_DIR}/joint_primary_sae.pt"
JOINT_META_PATH="${OUTPUT_DIR}/joint_meta_sae.pt"
SOLO_PRIMARY_PATH="${OUTPUT_DIR}/solo_primary_sae.pt"
SEQUENTIAL_META_PATH="${OUTPUT_DIR}/sequential_meta_sae.pt"

# -----------------------------------------------------------------------------
# Build command
# -----------------------------------------------------------------------------

CMD="python train_meta_sae.py \
    --dataset_path ${DATASET_PATH} \
    --dataset_name ${DATASET_NAME} \
    --layer ${LAYER} \
    --site ${SITE} \
    --dict_size ${DICT_SIZE} \
    --meta_dict_size ${META_DICT_SIZE} \
    --primary_top_k ${PRIMARY_TOP_K} \
    --meta_top_k ${META_TOP_K} \
    --num_tokens ${NUM_TOKENS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --lambda2 ${LAMBDA2} \
    --sigma_sq ${SIGMA_SQ} \
    --n_primary_steps ${N_PRIMARY_STEPS} \
    --n_meta_steps ${N_META_STEPS} \
    --num_batches_in_buffer_joint ${NUM_BATCHES_IN_BUFFER_JOINT} \
    --num_batches_in_buffer_sequential ${NUM_BATCHES_IN_BUFFER_SEQ} \
    --model_batch_size ${MODEL_BATCH_SIZE} \
    --seq_len ${SEQ_LEN} \
    --joint_primary_path ${JOINT_PRIMARY_PATH} \
    --joint_meta_path ${JOINT_META_PATH} \
    --solo_primary_path ${SOLO_PRIMARY_PATH} \
    --sequential_meta_path ${SEQUENTIAL_META_PATH} \
    --train_joint_saes \
    --train_sequential_saes"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

echo "============================================================"
echo "Meta-SAE Training"
echo "============================================================"
echo ""
echo "Paths:"
echo "  Project root:    ${PROJECT_ROOT}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo ""
echo "Key settings:"
echo "  Model:           ${MODEL_NAME} layer ${LAYER}"
echo "  Dataset:         ${DATASET_PATH}/${DATASET_NAME}"
echo "  Primary SAE:     ${DICT_SIZE} features, top-k=${PRIMARY_TOP_K}"
echo "  Meta SAE:        ${META_DICT_SIZE} features, top-k=${META_TOP_K}"
echo "  Penalty:         lambda2=${LAMBDA2}, sigma_sq=${SIGMA_SQ}"
echo "  Training:        ${NUM_TOKENS} tokens"
echo ""
echo "============================================================"

if [[ "$1" == "--dry-run" ]]; then
    echo "DRY RUN - Command that would be executed:"
    echo ""
    echo "$CMD"
    echo ""
else
    echo "Starting training..."
    echo ""
    eval $CMD
fi
