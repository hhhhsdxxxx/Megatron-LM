#!/bin/bash
# ===========================================================================
# Dump dataloader script — 基于 ci_bailing_moe_linear_v2.sh 的训练参数
#
# 用法:
#   bash run_dump_dataloader.sh
#
# 必须设置的环境变量:
#   DATA_PATH       — 数据集路径 (与 ci 脚本一致)
#
# 可选环境变量:
#   DUMP_DP_SIZE          — 训练时的总 DP 并行度 (TP=1, PP=1 → dp_size = 总GPU数)
#   DUMP_DP_RANK          — 模拟哪个 DP rank (-1 = 所有 rank, 默认)
#   DUMP_NUM_STEPS        — dump 多少步 (0 = 全部)
#   DUMP_CONSUMED_SAMPLES — 已消耗 sample 数 (用于断点续跑)
#   DUMP_NUM_WORKERS      — DataLoader worker 数
#   DUMP_OUTPUT_DIR       — dump 输出目录
#   JOB_DIR               — 实验目录 (用于 data_cache_path)
# ===========================================================================

set -exo pipefail

# ============================================================================
# Dump 专用参数
# ============================================================================
DUMP_DP_SIZE=${DUMP_DP_SIZE:-8}              # CI 默认 8 卡
DUMP_DP_RANK=${DUMP_DP_RANK:--1}            # -1 = dump 所有 rank
DUMP_NUM_STEPS=${DUMP_NUM_STEPS:-3}         # 0 = dump 全部 step
DUMP_CONSUMED_SAMPLES=${DUMP_CONSUMED_SAMPLES:-0}
DUMP_NUM_WORKERS=${DUMP_NUM_WORKERS:-0}
DUMP_OUTPUT_DIR=${DUMP_OUTPUT_DIR:-"/models/megatron_dump/output"}

# ============================================================================
# 路径配置 (与 ci_bailing_moe_linear_v2.sh 保持一致)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

JOB_DIR="${JOB_DIR:-/models/megatron_dump/dump_dataloader_job}"
DATA_CACHE_PATH=${JOB_DIR}/data_cache
mkdir -p ${DATA_CACHE_PATH}

# Flash Linear Attention library path
FLASH_LINEAR_PATH="${FLASH_LINEAR_PATH:-}"
if [ -n "$FLASH_LINEAR_PATH" ]; then
    export PYTHONPATH=${FLASH_LINEAR_PATH}:${SCRIPT_DIR}:$PYTHONPATH
else
    export PYTHONPATH=${SCRIPT_DIR}:$PYTHONPATH
fi

# ---- 数据路径 ----
DATA_PATH="${DATA_PATH:-/models/datasets/nemotron-cc-v2.1_megatron_indexed/High-Quality-Translated-To-English_text_document.bin}"
if [ "$DATA_PATH" = "/path/to/your/dataset_text_document" ]; then
    echo "ERROR: DATA_PATH is not set or using placeholder path."
    echo "Please set DATA_PATH environment variable to a valid dataset path."
    exit 1
fi
# CI 脚本使用 DATA_DIR 下的固定文件名
DATA_DIR=$(dirname "$DATA_PATH")
DATA_PATH="${DATA_DIR}/High-Quality-Translated-To-English_text_document"
if [ ! -e "$DATA_PATH.bin" ] || [ ! -e "$DATA_PATH.idx" ]; then
    echo "ERROR: Dataset files not found at: $DATA_PATH"
    echo "Expected files: ${DATA_PATH}.bin and ${DATA_PATH}.idx"
    exit 1
fi

# ============================================================================
# 环境变量
# ============================================================================
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ============================================================================
# 执行 dump_dataloader.py
# 所有模型/训练/数据参数与 ci_bailing_moe_linear_v2.sh 完全对齐
# ============================================================================
echo "=========================================="
echo "Starting Megatron Data Pipeline Dump"
echo "  dp_size=${DUMP_DP_SIZE}, dp_rank=${DUMP_DP_RANK}"
echo "  num_steps=${DUMP_NUM_STEPS}, output=${DUMP_OUTPUT_DIR}"
echo "=========================================="

python dump_dataloader.py \
    \
    --spec megatron.core.models.gpt.bailing_moe_linear_v2_layer_specs bailing_moe_linear_v2_block_spec \
    \
    --num-layers 20 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --num-query-groups 16 \
    --ffn-hidden-size 5120 \
    \
    --qk-layernorm \
    --use-flash-attn \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    \
    --position-embedding-type rope \
    --rotary-base 10000 \
    --rotary-percent 0.5 \
    --max-position-embeddings 4096 \
    \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    \
    --swiglu \
    --disable-bias-linear \
    \
    --vocab-size 157184 \
    --make-vocab-size-divisible-by 128 \
    --untie-embeddings-and-output-weights \
    \
    --multi-latent-attention \
    --q-lora-rank 256 \
    --kv-lora-rank 512 \
    --qk-head-dim 128 \
    --qk-pos-emb-head-dim 64 \
    --v-head-dim 128 \
    \
    --transformer-impl transformer_engine \
    \
    --linear-attention-freq "([1]*4+[0]*1)*4" \
    --linear-attn-norm-group-size 4 \
    --linear-attn-norm-group-type group_diff \
    \
    --num-experts 256 \
    --expert-model-parallel-size 8 \
    --moe-router-topk 8 \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-bias-update-rate 1e-3 \
    --moe-router-bias-zero-mean-update \
    --moe-router-dtype fp32 \
    --moe-token-dispatcher-type flex \
    --moe-grouped-gemm \
    --moe-z-loss-coeff 0.0000035 \
    --moe-ffn-hidden-size 512 \
    --moe-shared-expert-intermediate-size 2048 \
    --moe-layer-freq "([0]+[1]*19)" \
    \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    \
    --micro-batch-size 2 \
    --global-batch-size 5120 \
    --seq-length 4096 \
    --train-iters 95367 \
    --seed 42 \
    \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --lr 0.000336 \
    --lr-decay-style constant \
    --min-lr 0.000336 \
    --lr-warmup-iters 2000 \
    --init-method-std 0.006 \
    \
    --bf16 \
    \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --no-gradient-accumulation-fusion \
    \
    --data-path "$DATA_PATH" \
    --data-cache-path ${DATA_CACHE_PATH} \
    --split 999,1,0 \
    --dataloader-type single \
    --no-create-attention-mask-in-dataloader \
    --no-mmap-bin-files \
    --eod-mask-loss \
    \
    --eval-interval 1490 \
    --eval-iters 1 \
    \
    --attention-backend auto \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --cross-entropy-loss-fusion \
    \
    --dump-output-dir  "${DUMP_OUTPUT_DIR}" \
    --dump-dp-size     ${DUMP_DP_SIZE} \
    --dump-dp-rank     ${DUMP_DP_RANK} \
    --dump-num-steps   ${DUMP_NUM_STEPS} \
    --dump-consumed-samples ${DUMP_CONSUMED_SAMPLES} \
    --dump-num-workers ${DUMP_NUM_WORKERS}
