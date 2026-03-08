# BailingMoE Linear V2 Training Example

This directory contains documentation and examples for training the BailingMoE Linear V2 model using Megatron-LM.

## Overview

BailingMoE Linear V2 is a 20-layer decoder-only transformer model that combines three advanced techniques:

- **Hybrid Attention**: Alternates between Linear Attention (using Gated Linear Attention from flash-linear-attention) and Standard Attention in a 4:1 pattern
- **Mixture of Experts (MoE)**: Uses 256 routed experts with Group-Limited TopK routing, plus 1 shared expert
- **Multi-Token Prediction (MTP)**: Predicts the next token at each position to improve training efficiency

### Key Features

- 20 decoder layers with hybrid attention pattern (layers 0-3, 5-8, 10-13, 15-18 use Linear Attention; layers 4, 9, 14, 19 use Standard Attention)
- First layer uses dense MLP, remaining 19 layers use sparse MoE
- Group-Limited TopK Router: 8 groups of 32 experts each, select 4 groups, then 8 experts total
- 1 shared expert with intermediate size 2048 (equivalent to 4 routed experts)
- 1 MTP layer for auxiliary next-token prediction

### Architecture Summary

```
Model Parameters:
- Layers: 20
- Hidden Size: 2048
- Attention Heads: 16 (4 KV heads with GQA)
- FFN Hidden Size: 5120 (dense layer)
- MoE FFN Hidden Size: 512 (per expert)
- Vocab Size: 157,184
- Max Sequence Length: 4096
```

## Prerequisites

### 1. Install flash-linear-attention

The model requires the flash-linear-attention library for efficient linear attention computation:

```bash
git clone https://github.com/sustcsonglin/flash-linear-attention.git
cd flash-linear-attention
pip install -e .
```

### 2. Set Environment Variables

Before running training, set the following environment variables:

```bash
# Path to flash-linear-attention installation
export FLASH_LINEAR_PATH=/path/to/flash-linear-attention

# Job directory for checkpoints and logs
export JOB_DIR=/path/to/experiment/bailing-moe-linear-v2-megatron

# Dataset path (without .bin/.idx extension)
export DATA_PATH=/path/to/your/dataset_text_document

# Tokenizer path
export TOKENIZER_MODEL=/path/to/your/tokenizer
```

### 3. Prepare Dataset

Ensure your dataset is in Megatron-LM's binary format:
- `${DATA_PATH}.bin` - Binary data file
- `${DATA_PATH}.idx` - Index file

You can use Megatron-LM's `preprocess_data.py` to convert text data to this format.

## Quick Start

**Note**: The training script `pretrain_bailing_moe_linear_v2.sh` is located in the Megatron-LM root directory. All commands below should be run from the root directory of the Megatron-LM repository.

### Single Node Training

For training on a single node with multiple GPUs:

```bash
# Navigate to Megatron-LM root directory
cd /path/to/Megatron-LM

# Set required paths
export FLASH_LINEAR_PATH=/path/to/flash-linear-attention
export JOB_DIR=/path/to/experiment/bailing-moe-linear-v2-megatron
export DATA_PATH=/path/to/your/dataset_text_document
export TOKENIZER_MODEL=/path/to/your/tokenizer

# Run training script from root directory
bash pretrain_bailing_moe_linear_v2.sh
```

The script will automatically detect the number of GPUs on your node.

### Multi-Node Training

For distributed training across multiple nodes:

```bash
# Navigate to Megatron-LM root directory on each node
cd /path/to/Megatron-LM

# On each node, set the following environment variables:
export WORLD_SIZE=4              # Total number of nodes
export RANK=0                    # Node rank (0, 1, 2, 3)
export MASTER_ADDR=192.168.1.1   # IP of rank 0 node
export MASTER_PORT=6000          # Communication port

# Set required paths (same on all nodes)
export FLASH_LINEAR_PATH=/path/to/flash-linear-attention
export JOB_DIR=/path/to/experiment/bailing-moe-linear-v2-megatron
export DATA_PATH=/path/to/your/dataset_text_document
export TOKENIZER_MODEL=/path/to/your/tokenizer

# Run training script from root directory on each node
bash pretrain_bailing_moe_linear_v2.sh
```

### Updating Paths in the Script

If you prefer to modify the script directly instead of using environment variables, edit `pretrain_bailing_moe_linear_v2.sh` in the root directory:

```bash
# Flash Linear Attention path (FLASH_LINEAR_PATH variable)
FLASH_LINEAR_PATH="${FLASH_LINEAR_PATH:-/your/actual/path/flash-linear-attention}"

# Job directory (JOB_DIR variable)
JOB_DIR="${JOB_DIR:-/your/actual/path/experiment/bailing-moe-linear-v2-megatron}"

# Dataset path (DATA_PATH variable)
DATA_PATH="${DATA_PATH:-/your/actual/path/dataset_text_document}"

# Tokenizer path (TOKENIZER_MODEL variable)
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/your/actual/path/tokenizer}"
```

## Configuration

### Key Parameters

The training script is pre-configured with parameters from the reference implementation. Here are the most important ones:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--num-layers` | 20 | Number of transformer layers |
| `--hidden-size` | 2048 | Hidden dimension size |
| `--num-attention-heads` | 16 | Number of attention heads |
| `--num-query-groups` | 4 | Number of KV heads (GQA) |
| `--ffn-hidden-size` | 5120 | Dense MLP hidden size |
| `--vocab-size` | 157184 | Vocabulary size |
| `--max-position-embeddings` | 4096 | Maximum sequence length |
| `--micro-batch-size` | 2 | Batch size per GPU |
| `--global-batch-size` | 5120 | Total batch size across all GPUs |
| `--seq-length` | 4096 | Sequence length |
| `--train-iters` | 95367 | Number of training iterations |
| `--lr` | 0.000336 | Learning rate |
| `--num-experts` | 256 | Number of MoE experts |
| `--expert-model-parallel-size` | 8 | Expert parallelism degree |
| `--moe-router-topk` | 8 | Number of experts to route to |
| `--moe-ffn-hidden-size` | 512 | Hidden size per expert |
| `--moe-shared-expert-intermediate-size` | 2048 | Shared expert size |
| `--mtp-num-layers` | 1 | Number of MTP layers |
| `--mtp-loss-scaling-factor` | 0.1 | MTP loss weight |

**Note on Bias Configuration**: The script uses `--disable-bias-linear` to disable bias in linear layers, which matches the reference implementation's configuration for optimal performance.

### Parameter Mapping

For a complete mapping between the reference configuration and Megatron-LM parameters, see the design document:
`docs/plans/2026-03-08-bailing-moe-linear-v2-training-design.md`

## Performance Tuning

### Memory Optimization

If you encounter out-of-memory errors, try these options:

1. **Reduce micro batch size**:
   ```bash
   --micro-batch-size 1
   ```

2. **Enable activation checkpointing**:
   ```bash
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 10
   ```

3. **Use sequence parallelism** (already enabled):
   ```bash
   --sequence-parallel
   ```

4. **Use distributed optimizer** (already enabled):
   ```bash
   --use-distributed-optimizer
   ```

### Throughput Optimization

To improve training throughput:

1. **Increase micro batch size** (if memory allows):
   ```bash
   --micro-batch-size 4
   ```

2. **Enable communication overlap** (already enabled):
   ```bash
   --overlap-param-gather
   --overlap-grad-reduce
   ```

3. **Use grouped GEMM for MoE** (already enabled):
   ```bash
   --moe-grouped-gemm
   ```

4. **Adjust tensor/pipeline parallelism**:
   ```bash
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 2
   ```

### Expert Parallelism Configuration

The script uses `--expert-model-parallel-size 8`, which divides 256 experts across 8 GPUs (32 experts per GPU).

Adjust this based on your GPU count:
- 8 GPUs: `--expert-model-parallel-size 8` (32 experts/GPU)
- 16 GPUs: `--expert-model-parallel-size 16` (16 experts/GPU)
- 32 GPUs: `--expert-model-parallel-size 32` (8 experts/GPU)

The expert parallelism size must evenly divide the number of experts (256).

## Monitoring

### TensorBoard Setup

The script automatically logs metrics to TensorBoard. To view them:

```bash
tensorboard --logdir ${JOB_DIR}/tensorboard --port 6006
```

Then open `http://localhost:6006` in your browser.

### Key Metrics to Watch

1. **Loss Metrics**:
   - `lm loss`: Main language modeling loss
   - `mtp loss`: Multi-token prediction loss
   - `loss`: Total loss (main + 0.1 * MTP)

2. **MoE Metrics**:
   - `moe aux loss`: Load balancing auxiliary loss
   - `moe load balance`: Expert load distribution
   - `moe routing weights`: Expert selection patterns

3. **Performance Metrics**:
   - `throughput`: Tokens per second
   - `learning rate`: Current learning rate
   - `grad norm`: Gradient norm (should be < 1.0 due to clipping)

4. **Memory Metrics**:
   - `memory allocated`: GPU memory usage
   - `memory reserved`: Reserved GPU memory

### Log Files

Training logs are saved to:
- `${JOB_DIR}/log_${NODE_RANK}.txt` - Training output for each node
- `${JOB_DIR}/pip_list.txt` - Python package versions (rank 0 only)
- `${JOB_DIR}/collect_env.txt` - Environment information (rank 0 only)

## Troubleshooting

### flash-linear-attention not found

**Error**: `ModuleNotFoundError: No module named 'fla'`

**Solution**:
1. Verify flash-linear-attention is installed:
   ```bash
   python -c "import fla; print(fla.__file__)"
   ```

2. Check PYTHONPATH includes flash-linear-attention:
   ```bash
   echo $PYTHONPATH
   ```

3. Reinstall if necessary:
   ```bash
   cd /path/to/flash-linear-attention
   pip install -e .
   ```

### Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce micro batch size: `--micro-batch-size 1`
2. Increase activation checkpointing: `--recompute-granularity full`
3. Reduce sequence length: `--seq-length 2048`
4. Increase expert parallelism (if you have more GPUs)

### MoE Load Imbalance

**Issue**: Some experts receive significantly more tokens than others

**Solutions**:
1. Increase aux loss coefficient:
   ```bash
   --moe-aux-loss-coeff 0.00001
   ```

2. Enable expert bias (already enabled):
   ```bash
   --moe-router-enable-expert-bias
   ```

3. Monitor `moe load balance` metric in TensorBoard

### Training Hangs or Crashes

**Issue**: Training stops responding or crashes without clear error

**Solutions**:
1. Check NCCL settings (already configured in script):
   ```bash
   export TORCH_NCCL_AVOID_RECORD_STREAMS=1
   export NCCL_NVLS_ENABLE=0
   ```

2. Verify all nodes can communicate:
   ```bash
   # On each node
   ping $MASTER_ADDR
   ```

3. Check GPU health:
   ```bash
   nvidia-smi
   ```

4. Enable NCCL debug logging:
   ```bash
   export NCCL_DEBUG=INFO
   ```

### Checkpoint Loading Fails

**Issue**: Cannot resume from checkpoint

**Solutions**:
1. Verify checkpoint directory exists and is readable:
   ```bash
   ls -la ${JOB_DIR}/checkpoints
   ```

2. Check checkpoint format matches:
   ```bash
   --ckpt-format torch_dist
   ```

3. If changing configuration, start from scratch or use `--no-load-optim`

## References

### Documentation
- [Design Document](../../docs/plans/2026-03-08-bailing-moe-linear-v2-training-design.md) - Complete implementation design
- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM) - Official Megatron-LM docs

### Papers
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434) - MoE architecture
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437) - Multi-token prediction and improvements
- [Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models](https://arxiv.org/abs/2401.04658) - Linear attention

### Libraries
- [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) - Efficient linear attention implementation
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Large-scale transformer training framework

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the design document for implementation details
3. Check Megatron-LM documentation for general training issues
4. Verify flash-linear-attention installation and compatibility
