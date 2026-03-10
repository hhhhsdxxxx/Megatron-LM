#!/bin/bash
set -ex
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANDOM_PORT=$[$RANDOM + 20000]
MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}
GPU_NUM=$((${GPUS_PER_NODE}*${WORLD_SIZE}))
echo "---> from pytorch runtime, WORLD_SIZE: ${WORLD_SIZE}, NODE_RANK: ${NODE_RANK}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"
LAUNCHER=" \
    torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    "

export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG_SUBSYS=INIT   # disable aistudio default nccl env

export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0

export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2  # 2 means DEBUG level


DEVICE_MODEL=$(nvidia-smi -i 0 -q | grep "Product Name" | awk -F: '{ print $2 }')
DEVICE_MODEL=$(echo "$DEVICE_MODEL" | xargs)  # drop white space

if [[ $DEVICE_MODEL == NVIDIA* ]]; then
    DEVICE_MODEL=${DEVICE_MODEL#"NVIDIA"}
    DEVICE_MODEL=$(echo "$DEVICE_MODEL" | sed 's/^ *//')
fi

if [ "$DEVICE_MODEL" = "A800-SXM4-80GB" ] || [ "$DEVICE_MODEL" = "A100-SXM4-80GB" ]; then
    # Ampere GPUs do not support multicast. If `--tp-comm-overlap` is set on Ampere-arch GPUs, this env must be set.
    export UB_SKIPMC=1
fi

JOB_DIR="/mnt/exp/moe-lite/yangqingyuan.yqy/tpu/moe-mini-v25-e256-0520-fp8-20T-hl-mla-baseline/"
CHECKPOINT_PATH=${JOB_DIR} #<Specify path>
TENSORBOARD_LOGS_PATH=/home/admin/logs/tfevent/runs

if [[ $RANK -eq 0 ]]; then
    cp -r ${0} ${JOB_DIR}
    pip list > ${JOB_DIR}/pip_list.txt
    python -m torch.utils.collect_env > ${JOB_DIR}/collect_env.txt
fi

LOG_PATH="${JOB_DIR}/log_${NODE_RANK}.txt"

MOE_ARGS=(
    --moe-enable-deepep
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --moe-grouped-gemm
    --moe-token-dispatcher-type flex
    --moe-router-dtype fp32
    --num-experts 256
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 2048
    --moe-router-score-function sigmoid
    --moe-router-topk 8
    --moe-router-enable-expert-bias
    --moe-router-topk-scaling-factor 2.5
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-z-loss-coeff 0.0000035
    --moe-router-bias-update-rate 1e-3
    --moe-layer-freq [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
)

MPT_ARGS=(
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
)

GPT_MODEL_ARGS=(
    --num-layers 20
    --layer-group-size 5
    --linear-attn-norm-group-size 4
    --linear-attn-norm-group-type group_diff
    --no-linear-silu
    --hidden-size 2048
    --ffn-hidden-size 5120
    --num-attention-heads 16
    --num-query-groups 16
    --multi-latent-attention
    --q-lora-rank 256
    --kv-lora-rank 512
    --max-position-embeddings 4096
    --vocab-size 157184
    --make-vocab-size-divisible-by 128
    --position-embedding-type "rope"
    --rotary-base 10000
    --rotary-percent 0.5
    --swiglu
    --untie-embeddings-and-output-weights
    --normalization "RMSNorm"
    --norm-epsilon "1e-06"
    --disable-bias-linear
    --transformer-impl "transformer_engine"
    --attention-dropout 0
    --hidden-dropout 0
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 5120
    --seq-length "4096"
    --train-iters 95367
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0

    --bf16
    --fp8-param-gather
    --fp8-recipe "blockwise"
    --fp8-format "e4m3"

    --lr "0.000336"
    --lr-decay-style constant
    --min-lr "0.000336"
    --lr-warmup-iters 2000
    --seed 42

    --skip-casting-dtype-for-param-pattern "^expert_bias$|.+\.expert_bias$|^local_tokens_per_expert$|.+\.local_tokens_per_expert$"
    --bias-zero-mean-update
)

MODEL_PARALLEL_ARGS=(
    --pipeline-model-parallel-size 1
    --tensor-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --recompute-granularity selective
    --recompute-modules moe

    --overlap-param-gather
    --overlap-grad-reduce
)

DATA_ARGS=(
    --data-path "0.007496 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2015-06_text_document 0.011284 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2020-40_text_document 0.014374 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2022-49_text_document 0.012031 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-17_text_document 0.016223 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2023-50_text_document 0.008039 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2014-35_text_document 0.010140 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-04_text_document 0.009286 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-30_text_document 0.009707 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-43_text_document 0.008900 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2021-25_text_document 0.006225 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2016-18_text_document 0.006878 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2014-49_text_document 0.008507 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-26_text_document 0.014332 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2023-06_text_document 0.008476 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2014-23_text_document 0.008892 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2021-49_text_document 0.007542 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2015-35_text_document 0.008495 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-30_text_document 0.009697 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-43_text_document 0.007678 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2013-48_text_document 0.007336 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2016-07_text_document 0.011405 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2021-39_text_document 0.007493 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2016-30_text_document 0.007808 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2013-20_text_document 0.008092 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2014-52_text_document 0.008755 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-34_text_document 0.009313 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-26_text_document 0.008676 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-13_text_document 0.009302 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-26_text_document 0.013625 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2022-40_text_document 0.007862 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2020-10_text_document 0.006527 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2016-22_text_document 0.008755 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2020-50_text_document 0.007680 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-51_text_document 0.008852 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-13_text_document 0.008020 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2015-22_text_document 0.009679 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2021-10_text_document 0.009729 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2016-44_text_document 0.007907 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2014-10_text_document 0.007441 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2015-32_text_document 0.008924 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-43_text_document 0.013010 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2022-27_text_document 0.007153 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2015-27_text_document 0.009013 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2020-45_text_document 0.009753 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-51_text_document 0.014132 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2023-14_text_document 0.008989 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-18_text_document 0.012012 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2021-17_text_document 0.007792 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2014-42_text_document 0.011763 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2021-04_text_document 0.008925 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-22_text_document 0.010359 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-09_text_document 0.009268 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-35_text_document 0.010435 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2020-05_text_document 0.007539 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2014-15_text_document 0.009139 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-09_text_document 0.008317 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-47_text_document 0.012801 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2021-31_text_document 0.007378 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-22_text_document 0.008279 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2014-41_text_document 0.007725 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-51_text_document 0.006076 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2015-40_text_document 0.008532 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-39_text_document 0.011422 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2022-05_text_document 0.007223 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2015-14_text_document 0.016813 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2023-40_text_document 0.007332 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2016-36_text_document 0.005484 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2016-26_text_document 0.008119 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2016-40_text_document 0.008298 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-34_text_document 0.008771 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-22_text_document 0.013152 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2021-43_text_document 0.008753 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-47_text_document 0.009630 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2016-50_text_document 0.008484 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-39_text_document 0.014387 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2022-21_text_document 0.009826 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2020-16_text_document 0.012003 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-13_text_document 0.009878 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2021-21_text_document 0.008535 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-04_text_document 0.009426 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2022-33_text_document 0.009492 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-05_text_document 0.010590 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2020-29_text_document 0.007460 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2015-48_text_document 0.008263 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-17_text_document 0.008827 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2017-30_text_document 0.008223 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-39_text_document 0.008236 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2020-24_text_document 0.008388 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2020-34_text_document 0.008101 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2015-18_text_document 0.007651 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2015-11_text_document 0.015204 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2023-23_text_document 0.009523 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2019-09_text_document 0.008898 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/fineweb/processed_data/CC-MAIN-2018-47_text_document 0.018598 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/Nemotron_processed_data/Medium-High-Quality-Translated-To-English_text_document 0.005618 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/Nemotron_processed_data/High-Quality-DQA_text_document 0.017845 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/Nemotron_processed_data/High-Quality_text_document 0.036612 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/Nemotron_processed_data/Medium-Quality_text_document 0.027574 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/Nemotron_processed_data/High-Quality-Translated-To-English_text_document 0.011591 /mnt/exp/moe-lite/yangqingyuan.yqy/dataset/Nemotron_processed_data/Medium-High-Quality_text_document"
    --data-cache-path "/personal/tpu/data_cache_path_formal"
    --tokenizer-type "HuggingFaceTokenizer"
    --tokenizer-model "/ossfs/workspace/bt2"
    --split 999,1,0
    --dataloader-type "single"
    --no-create-attention-mask-in-dataloader
    --eod-mask-loss
)

EVAL_AND_LOGGING_ARGS=(
    --save-interval 1490
    --eval-interval 1490
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --override-opt_param-scheduler
    --no-save-rng
    --no-load-optim
    --qk-layernorm
    --use-flash-attn
    --no-load-rng
    --ckpt-format "torch_dist"
    --async-save
    --eval-iters 1
    --log-interval 1
    --log-throughput
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --log-validation-ppl-to-tensorboard
)

KERNEL_ARGS=(
    --attention-backend auto
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --cross-entropy-loss-fusion
)


CMD="${LAUNCHER} pretrain_gpt.py \
    ${MOE_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${KERNEL_ARGS[@]} \
    ${MPT_ARGS[@]} \
    ${PROFILING_ARGS[@]} \
"

echo ${CMD}
export PYTHONPATH=/ossfs/workspace/flash-linear-attention:$PYTHONPATH
PYTHONPATH=/workspace/bin/Megatron-LM:$PYTHONPATH ${CMD} 2>&1 | tee ${LOG_PATH}
