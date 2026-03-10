#!/bin/bash
# BailingMoE Linear V2 Training Script for Megatron-LM
# Based on reference: ling_2.5_megatron.sh
#
# This script adapts the reference configuration to work with current Megatron-LM
# See IMPLEMENTATION_DIFFERENCES.md for detailed comparison

set -ex

# ============================================================================
# Environment Setup
# ============================================================================
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANDOM_PORT=$[$RANDOM + 20000]
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

# NCCL settings
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0

# Flash Linear Attention library path
export PYTHONPATH=/path/to/flash-linear-attention:$PYTHONPATH

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
JOB_DIR="/path/to/experiment/bailing-moe-linear-v2-megatron"
CHECKPOINT_PATH=${JOB_DIR}/checkpoints
TENSORBOARD_LOGS_PATH=${JOB_DIR}/tensorboard
DATA_CACHE_PATH=${JOB_DIR}/data_cache

# Create directories
mkdir -p ${JOB_DIR}
mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${TENSORBOARD_LOGS_PATH}
mkdir -p ${DATA_CACHE_PATH}

# Save script and environment info
if [[ $RANK -eq 0 ]]; then
    cp -r ${0} ${JOB_DIR}/
    pip list > ${JOB_DIR}/pip_list.txt
    python -m torch.utils.collect_env > ${JOB_DIR}/collect_env.txt
fi

LOG_PATH="${JOB_DIR}/log_${NODE_RANK}.txt"

# ============================================================================
# Model Architecture Arguments
# ============================================================================
GPT_MODEL_ARGS=(
    # Basic architecture
    --num-layers 20
    --hidden-size 2048
    --num-attention-heads 16
    --num-query-groups 4
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
# Multi-Latent Attention (MLA) Configuration
# ============================================================================
# Standard attention layers (every 5th layer) use MLA instead of vanilla MHA
MLA_ARGS=(
    --multi-latent-attention
    --q-lora-rank 256
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
)

# ============================================================================
# MoE Configuration
# ============================================================================
MOE_ARGS=(
    --num-experts 256
    --expert-model-parallel-size 8
    --moe-router-topk 8

    # Token dispatcher
    --moe-token-dispatcher-type flex
    --moe-grouped-gemm

    # Router z-loss (matching reference moe-z-loss-coeff)
    --moe-z-loss-coeff 0.0000035
    --moe-router-score-function sigmoid
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-bias-zero-mean-update
    --moe-router-topk-scaling-factor 2.5

    # Router precision and initialization (matching reference)
    --moe-router-dtype fp32

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
    --fp8-param-gather
)

# ============================================================================
# Parallelism Configuration
# ============================================================================
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer

    # Activation recomputation
    --recompute-granularity selective
    --recompute-method uniform
    --recompute-num-layers 1

    # Communication overlap
    --overlap-param-gather
    --overlap-grad-reduce
)

# ============================================================================
# Data Configuration
# ============================================================================
DATA_ARGS=(
    --data-path "/path/to/your/dataset_text_document"
    --data-cache-path ${DATA_CACHE_PATH}
    --split 999,1,0

    # Tokenizer
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "/path/to/your/tokenizer"

    # Data loader
    --dataloader-type single
    --no-create-attention-mask-in-dataloader
    --eod-mask-loss
)

# ============================================================================
# Checkpointing and Logging
# ============================================================================
EVAL_AND_LOGGING_ARGS=(
    # Checkpointing
    --save ${CHECKPOINT_PATH}
    --load ${CHECKPOINT_PATH}
    --save-interval 1490
    --ckpt-format torch_dist
    --async-save
    --no-save-rng
    --no-load-rng
    --no-load-optim

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
    --cross-entropy-loss-fusion
)

# ============================================================================
# Build Command
# ============================================================================
CMD="${LAUNCHER} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LINEAR_ATTN_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MTP_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${KERNEL_ARGS[@]} \
"

# ============================================================================
# Execute
# ============================================================================
echo "=========================================="
echo "Starting BailingMoE Linear V2 Training"
echo "=========================================="
echo ${CMD}
echo "=========================================="

${CMD} 2>&1 | tee ${LOG_PATH}
