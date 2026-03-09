# BailingMoE Linear V2 CI Design

## Goal

Create a GitHub Actions workflow that runs BailingMoE Linear V2 pretraining for 10 steps on 8x H200 GPUs with real data and weights. Pass criteria: exit code = 0.

## Approach

Independent GitHub Actions workflow (`.github/workflows/ci-bailing-moe-linear-v2.yml`) that mirrors the production `pretrain_bailing_moe_linear_v2.sh` with minimal CI-specific changes.

## Changes from Production Script

Only 3 modifications:

1. **Add** `--exit-interval 10` — stop after 10 training steps
2. **Remove** `--fp8-param-gather` — disable FP8
3. **Remove** checkpoint saving params (`--save`, `--async-save`, `--save-interval`, `--no-save-rng`)

All other parameters (model architecture, data, training config) remain identical.

## Environment Variables (configured in CI)

- `DATA_PATH` — dataset `.bin/.idx` path
- `TOKENIZER_MODEL` — HuggingFace tokenizer path
- `JOB_DIR` — output directory
- `FLASH_LINEAR_PATH` — flash-linear-attention library path (optional if pip installed)

## Dependencies

- `pip install flash-linear-attention` — installed fresh each run to stay in sync

## Workflow Steps

1. Checkout repository
2. Install flash-linear-attention via pip
3. Run `torchrun --nproc_per_node=8 pretrain_gpt.py` with full production config + `--exit-interval 10`
4. Success = exit code 0

## Key Architecture Parameters (unchanged)

- 20 layers, hidden=2048, 16 heads, 4 KV groups, FFN=5120
- 256 experts, EP=8, topk=8, grouped topk with 8 groups
- Linear attention pattern: `([1]*4+[0]*1)*4`
- MTP: 1 layer, loss scaling 0.1
- micro-batch-size=2, global-batch-size=5120, seq-length=4096
- HuggingFaceTokenizer
