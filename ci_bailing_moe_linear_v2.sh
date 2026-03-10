#!/bin/bash
# BailingMoE Linear V2 CI Training Script for Megatron-LM
# Based on: pretrain_bailing_moe_linear_v2.sh
#
# CI version: runs 10 training steps (--exit-interval 10) with no checkpoint saving.
# Intended for 8x H200 GPU validation.

set -exo pipefail

# ============================================================================
# Environment Setup
# ============================================================================
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

# Validate GPU count
if [ "$GPUS_PER_NODE" -eq 0 ]; then
    echo "ERROR: No GPUs detected. GPUS_PER_NODE is 0."
    exit 1
fi

WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANDOM_PORT=$(( RANDOM % 10000 + 20000 ))
MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}
GPU_NUM=$((${GPUS_PER_NODE}*${WORLD_SIZE}))

echo "---> WORLD_SIZE: ${WORLD_SIZE}, NODE_RANK: ${NODE_RANK}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"

LAUNCHER=" \
    torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    "

# ============================================================================
# Environment Variables
# ============================================================================
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# NCCL settings for optimal distributed training
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0

# Flash Linear Attention library path
# CI: supports both pip-installed fla and FLASH_LINEAR_PATH directory
FLASH_LINEAR_PATH="${FLASH_LINEAR_PATH:-}"
if [ -n "$FLASH_LINEAR_PATH" ]; then
    if [ ! -d "$FLASH_LINEAR_PATH" ]; then
        echo "ERROR: FLASH_LINEAR_PATH does not exist: $FLASH_LINEAR_PATH"
        echo "Please set FLASH_LINEAR_PATH environment variable to the correct path."
        exit 1
    fi
    export PYTHONPATH=${FLASH_LINEAR_PATH}:$PYTHONPATH
else
    if ! python -c "import fla" 2>/dev/null; then
        echo "ERROR: flash-linear-attention not found."
        echo "Either set FLASH_LINEAR_PATH or pip install flash-linear-attention."
        exit 1
    fi
fi

# Transformer Engine debug (optional)
# export NVTE_DEBUG=1
# export NVTE_DEBUG_LEVEL=2

# ============================================================================
# GPU-specific settings
# ============================================================================
DEVICE_MODEL=$(nvidia-smi -i 0 -q | grep "Product Name" | awk -F: '{ print $2 }')
DEVICE_MODEL=$(echo "$DEVICE_MODEL" | xargs)

if [[ $DEVICE_MODEL == NVIDIA* ]]; then
    DEVICE_MODEL=${DEVICE_MODEL#"NVIDIA"}
    DEVICE_MODEL=$(echo "$DEVICE_MODEL" | sed 's/^ *//')
fi

if [ "$DEVICE_MODEL" = "A800-SXM4-80GB" ] || [ "$DEVICE_MODEL" = "A100-SXM4-80GB" ]; then
    # Ampere GPUs do not support multicast
    export UB_SKIPMC=1
fi

# ============================================================================
# Paths Configuration
# ============================================================================
JOB_DIR="${JOB_DIR:-/path/to/experiment/bailing-moe-linear-v2-megatron}"
if [ "$JOB_DIR" = "/path/to/experiment/bailing-moe-linear-v2-megatron" ]; then
    echo "ERROR: JOB_DIR is not set or using placeholder path."
    echo "Please set JOB_DIR environment variable to a valid directory."
    exit 1
fi

CHECKPOINT_PATH=${JOB_DIR}/checkpoints
TENSORBOARD_LOGS_PATH=${JOB_DIR}/tensorboard
DATA_CACHE_PATH=${JOB_DIR}/data_cache

# Create directories
mkdir -p ${JOB_DIR}
mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${TENSORBOARD_LOGS_PATH}
mkdir -p ${DATA_CACHE_PATH}

# Validate directories are writable
if [ ! -w "${JOB_DIR}" ]; then
    echo "ERROR: JOB_DIR is not writable: ${JOB_DIR}"
    exit 1
fi
if [ ! -w "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: CHECKPOINT_PATH is not writable: ${CHECKPOINT_PATH}"
    exit 1
fi
if [ ! -w "${TENSORBOARD_LOGS_PATH}" ]; then
    echo "ERROR: TENSORBOARD_LOGS_PATH is not writable: ${TENSORBOARD_LOGS_PATH}"
    exit 1
fi
if [ ! -w "${DATA_CACHE_PATH}" ]; then
    echo "ERROR: DATA_CACHE_PATH is not writable: ${DATA_CACHE_PATH}"
    exit 1
fi

# Save script and environment info
if [[ ${NODE_RANK} -eq 0 ]]; then
    cp -r ${0} ${JOB_DIR}/
    pip list > ${JOB_DIR}/pip_list.txt
    python -m torch.utils.collect_env > ${JOB_DIR}/collect_env.txt
fi

LOG_PATH="${JOB_DIR}/log_${NODE_RANK}.txt"

# ============================================================================
# Model Architecture Arguments
# ============================================================================
GPT_MODEL_ARGS=(
    # Custom layer spec for BailingMoE Linear V2
    # Note: --spec requires two arguments: module_path and symbol_name
    --spec megatron.core.models.gpt.bailing_moe_linear_v2_layer_specs bailing_moe_linear_v2_block_spec

    # Basic architecture
    --num-layers 20
    --hidden-size 2048
    --num-attention-heads 16
    --num-query-groups 16
    --ffn-hidden-size 5120

    # Attention configuration
    --group-query-attention
    --qk-layernorm
    --use-flash-attn
    --attention-dropout 0
    --hidden-dropout 0

    # Position embeddings
    --position-embedding-type rope
    --rotary-base 10000
    --rotary-percent 0.5
    --rotary-interleaved
    --max-position-embeddings 4096

    # Normalization
    --normalization RMSNorm
    --norm-epsilon 1e-06

    # Activation
    --swiglu
    --disable-bias-linear

    # Vocabulary
    --vocab-size 157184
    --make-vocab-size-divisible-by 128
    --untie-embeddings-and-output-weights

    # MLA dimension args (for hybrid Linear Attention + MLA layers)
    --q-lora-rank 256
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128

    # Implementation
    --transformer-impl transformer_engine
)

# ============================================================================
# Linear Attention Configuration
# ============================================================================
# Pattern: 4 linear attention layers + 1 standard attention layer (layer_group_size=5)
# Layers 0-3: Linear Attention
# Layer 4: Standard Attention
# Layers 5-8: Linear Attention
# Layer 9: Standard Attention
# ... and so on
LINEAR_ATTN_ARGS=(
    --linear-attention-freq "([1]*4+[0]*1)*4"
    --linear-attn-norm-group-size 4
    --linear-attn-norm-group-type group_diff
)

# ============================================================================
# MoE Configuration
# ============================================================================
# Group-Limited TopK Router: 256 experts in 8 groups, select 4 groups, then 8 experts
MOE_ARGS=(
    --num-experts 256
    --expert-model-parallel-size 8
    --moe-router-topk 8

    # Group-Limited TopK Router configuration
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-router-topk-scaling-factor 2.5
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-bias-zero-mean-update
    --moe-router-dtype fp32

    # Token dispatcher
    --moe-token-dispatcher-type flex
    --moe-grouped-gemm

    # Router z-loss (matching reference moe-z-loss-coeff)
    --moe-z-loss-coeff 0.0000035

    # Expert configuration
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 2048

    # Layer frequency: [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # First layer is dense, rest are MoE
    --moe-layer-freq "([0]+[1]*19)"
)

# ============================================================================
# Multi-Token Prediction (MTP) Configuration
# ============================================================================
MTP_ARGS=(
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --mtp-loss-scaling-per-layer
)

# ============================================================================
# Training Configuration
# ============================================================================
TRAINING_ARGS=(
    # Batch sizes
    --micro-batch-size 2
    --global-batch-size 5120
    --seq-length 4096

    # Training iterations
    --train-iters 95367

    # CI: exit after 10 steps
    --exit-interval 10

    # Optimizer
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --weight-decay 0.1
    --clip-grad 1.0

    # Learning rate
    --lr 0.000336
    --lr-decay-style constant
    --min-lr 0.000336
    --lr-warmup-iters 2000

    # Initialization
    --init-method-std 0.006
    --seed 42

    # Precision
    --bf16
)

# ============================================================================
# Parallelism Configuration
# ============================================================================
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --no-gradient-accumulation-fusion

    # Activation recomputation
    --recompute-granularity selective

    # Communication overlap
    --overlap-param-gather
    --overlap-grad-reduce
)

# ============================================================================
# Data Configuration
# ============================================================================
DATA_PATH="${DATA_PATH:-/path/to/your/dataset_text_document}"
if [ "$DATA_PATH" = "/path/to/your/dataset_text_document" ]; then
    echo "ERROR: DATA_PATH is not set or using placeholder path."
    echo "Please set DATA_PATH environment variable to a valid dataset path."
    exit 1
fi
if [ ! -e "$DATA_PATH.bin" ] || [ ! -e "$DATA_PATH.idx" ]; then
    echo "ERROR: Dataset files not found at: $DATA_PATH"
    echo "Expected files: ${DATA_PATH}.bin and ${DATA_PATH}.idx"
    exit 1
fi

TOKENIZER_MODEL="${TOKENIZER_MODEL:-/path/to/your/tokenizer}"
if [ "$TOKENIZER_MODEL" = "/path/to/your/tokenizer" ]; then
    echo "ERROR: TOKENIZER_MODEL is not set or using placeholder path."
    echo "Please set TOKENIZER_MODEL environment variable to a valid tokenizer path."
    exit 1
fi
if [ ! -e "$TOKENIZER_MODEL" ]; then
    echo "ERROR: Tokenizer not found at: $TOKENIZER_MODEL"
    exit 1
fi

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --data-cache-path ${DATA_CACHE_PATH}
    --split 999,1,0

    # Tokenizer
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"

    # Data loader
    --dataloader-type single
    --no-create-attention-mask-in-dataloader
    --eod-mask-loss
)

# ============================================================================
# Checkpoint Loading (optional)
# ============================================================================
# Set LOAD_PATH to load a pre-converted Megatron checkpoint for continued training.
# Convert from HF safetensors using:
#   python tools/checkpoint/convert_bailing_moe_linear_v2_hf.py \
#       --input-dir /path/to/hf/safetensors --output-dir /path/to/megatron/ckpt
LOAD_PATH="${LOAD_PATH:-/data/checkpoints/bailing-moe-linear-v2}"

CHECKPOINT_LOAD_ARGS=()
if [ -n "$LOAD_PATH" ]; then
    CHECKPOINT_LOAD_ARGS=(
        --load ${LOAD_PATH}
        --ckpt-format torch
        --no-load-rng
        --no-load-optim
    )
fi

# ============================================================================
# Checkpointing and Logging (CI: no checkpoint saving)
# ============================================================================
EVAL_AND_LOGGING_ARGS=(

    # Evaluation
    --eval-interval 1490
    --eval-iters 1

    # Logging
    --log-interval 1
    --log-throughput
    --tensorboard-dir ${TENSORBOARD_LOGS_PATH}
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --log-validation-ppl-to-tensorboard
)

# ============================================================================
# Kernel Configuration
# ============================================================================
KERNEL_ARGS=(
    --attention-backend auto
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
)

# ============================================================================
# Build Command
# ============================================================================
CMD="${LAUNCHER} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LINEAR_ATTN_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MTP_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${KERNEL_ARGS[@]} \
    ${CHECKPOINT_LOAD_ARGS[@]} \
"

# ============================================================================
# Execute
# ============================================================================
echo "=========================================="
echo "Starting BailingMoE Linear V2 CI Training"
echo "=========================================="
echo ${CMD}
echo "=========================================="

${CMD} 2>&1 | tee ${LOG_PATH}
