# BailingMoE Linear V2 CI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a GitHub Actions workflow that smoke-tests BailingMoE Linear V2 pretraining (10 steps, real data, 8x H200).

**Architecture:** A single CI shell script (`ci_bailing_moe_linear_v2.sh`) mirrors the production training script with 3 minimal changes (add `--exit-interval 10`, remove FP8, remove checkpoint saving). A GitHub Actions workflow calls this script with environment variables for data/tokenizer paths.

**Tech Stack:** GitHub Actions, bash, torchrun, Megatron-LM, flash-linear-attention (pip)

---

### Task 1: Create CI training script

**Files:**
- Create: `ci_bailing_moe_linear_v2.sh`

**Step 1: Write the CI training script**

Copy `pretrain_bailing_moe_linear_v2.sh` and apply exactly 3 changes:

1. In `TRAINING_ARGS`, remove `--fp8-param-gather`
2. In `EVAL_AND_LOGGING_ARGS`, remove checkpoint saving params: `--save`, `--save-interval`, `--ckpt-format`, `--async-save`, `--no-save-rng`
3. Add `--exit-interval 10` to `TRAINING_ARGS`

The resulting `ci_bailing_moe_linear_v2.sh` keeps ALL other parameters identical (model arch, data, training, parallelism, logging).

Full diff from production script:

```diff
 TRAINING_ARGS=(
     ...
     --bf16
-    --fp8-param-gather
+    --exit-interval 10
 )
```

```diff
 EVAL_AND_LOGGING_ARGS=(
-    # Checkpointing
-    --save ${CHECKPOINT_PATH}
     --load ${CHECKPOINT_PATH}
-    --save-interval 1490
-    --ckpt-format torch_dist
-    --async-save
-    --no-save-rng
     --no-load-rng
     --no-load-optim
     ...
 )
```

**Step 2: Make script executable**

Run: `chmod +x ci_bailing_moe_linear_v2.sh`

**Step 3: Verify script syntax**

Run: `bash -n ci_bailing_moe_linear_v2.sh`
Expected: No output (no syntax errors)

**Step 4: Commit**

```bash
git add ci_bailing_moe_linear_v2.sh
git commit -m "ci: add BailingMoE Linear V2 smoke test training script"
```

---

### Task 2: Create GitHub Actions workflow

**Files:**
- Create: `.github/workflows/ci-bailing-moe-linear-v2.yml`

**Step 1: Write the workflow file**

```yaml
name: CI - BailingMoE Linear V2 Pretrain

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

env:
  DATA_PATH: ${{ vars.BAILING_DATA_PATH }}
  TOKENIZER_MODEL: ${{ vars.BAILING_TOKENIZER_MODEL }}
  JOB_DIR: ${{ runner.temp }}/bailing-moe-linear-v2-ci
  FLASH_LINEAR_PATH: ${{ vars.BAILING_FLASH_LINEAR_PATH }}

jobs:
  pretrain-smoke-test:
    name: BailingMoE Linear V2 - 10 Step Smoke Test
    runs-on: [self-hosted, 8xH200]
    timeout-minutes: 60
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install flash-linear-attention
        run: pip install flash-linear-attention

      - name: Run BailingMoE Linear V2 pretraining (10 steps)
        run: bash ci_bailing_moe_linear_v2.sh
```

**Step 2: Validate YAML syntax**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci-bailing-moe-linear-v2.yml'))"`
Expected: No error

**Step 3: Commit**

```bash
git add .github/workflows/ci-bailing-moe-linear-v2.yml
git commit -m "ci: add GitHub Actions workflow for BailingMoE Linear V2 smoke test"
```

---

### Task 3: Verify end-to-end locally (if GPUs available)

**Step 1: Dry-run the CI script arguments**

Run: `bash -x ci_bailing_moe_linear_v2.sh 2>&1 | head -50`

Verify the printed command includes `--exit-interval 10`, does NOT include `--fp8-param-gather`, does NOT include `--save` or `--async-save`.

**Step 2: Commit design doc**

```bash
git add docs/plans/2026-03-08-bailing-moe-linear-v2-ci-design.md docs/plans/2026-03-08-bailing-moe-linear-v2-ci-implementation.md
git commit -m "docs: add BailingMoE Linear V2 CI design and implementation plan"
```
