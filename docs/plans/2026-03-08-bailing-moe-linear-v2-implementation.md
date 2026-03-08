# BailingMoE Linear V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement BailingMoE Linear V2 model training in Megatron-LM with hybrid Linear/Standard Attention, MoE layers, and Multi-Token Prediction.

**Architecture:** Reuse Megatron-LM's existing LinearAttention, Group-Limited TopK Router, and MoE components. Create custom MTP layer spec with final_layernorm. Configure via training script with all parameters mapped from reference implementation.

**Tech Stack:** Megatron-LM, PyTorch, Transformer Engine, flash-linear-attention (fla), NCCL

---

## Task 1: Create Custom MTP Layer Spec with final_layernorm

**Files:**
- Create: `megatron/core/models/gpt/bailing_moe_linear_v2_layer_specs.py`

**Step 1: Create the layer spec file with imports**

Create file with basic imports and structure.

**Step 2: Define custom MTP layer submodules with final_layernorm**

Implement `get_bailing_moe_v2_mtp_layer_submodules()` function that returns MultiTokenPredictionLayerSubmodules with custom final_layernorm.

**Step 3: Verify the file compiles**

Run: `python -c "from megatron.core.models.gpt.bailing_moe_linear_v2_layer_specs import get_bailing_moe_v2_mtp_layer_submodules; print('Import successful')"`

Expected: "Import successful"

**Step 4: Commit**

```bash
git add megatron/core/models/gpt/bailing_moe_linear_v2_layer_specs.py
git commit -m "feat: add custom MTP layer spec for BailingMoE Linear V2"
```

---

## Task 2: Add Layer Spec Function for BailingMoE Linear V2

**Files:**
- Modify: `megatron/core/models/gpt/bailing_moe_linear_v2_layer_specs.py`

**Step 1: Add function to get layer spec based on layer index**

Implement `get_bailing_moe_linear_v2_layer_spec()` that:
- Determines attention type based on layer_group_size pattern (4 linear + 1 standard)
- Determines MLP type (layer 0 dense, rest MoE)
- Returns appropriate TransformerLayerSubmodules

**Step 2: Verify the function works**

Run: `python -c "from megatron.core.models.gpt.bailing_moe_linear_v2_layer_specs import get_bailing_moe_linear_v2_layer_spec; print('Function defined successfully')"`

Expected: "Function defined successfully"

**Step 3: Commit**

```bash
git add megatron/core/models/gpt/bailing_moe_linear_v2_layer_specs.py
git commit -m "feat: add layer spec function for BailingMoE Linear V2"
```

---

## Task 3: Create Training Script

**Files:**
- Create: `pretrain_bailing_moe_linear_v2.sh`

**Step 1: Create script with environment setup**

Create bash script with:
- GPU detection and distributed training setup
- Environment variables (NCCL, CUDA, flash-linear-attention path)
- Path configuration

**Step 2: Add model architecture arguments**

Add GPT_MODEL_ARGS with all basic architecture parameters.

**Step 3: Add Linear Attention, MoE, and MTP configuration**

Add LINEAR_ATTN_ARGS, MOE_ARGS, and MTP_ARGS sections.

**Step 4: Add training, parallelism, data, and logging configuration**

Add TRAINING_ARGS, MODEL_PARALLEL_ARGS, DATA_ARGS, EVAL_AND_LOGGING_ARGS, and KERNEL_ARGS.

**Step 5: Build command and execution**

Combine all argument arrays into final command.

**Step 6: Make script executable**

Run: `chmod +x pretrain_bailing_moe_linear_v2.sh`

**Step 7: Commit**

```bash
git add pretrain_bailing_moe_linear_v2.sh
git commit -m "feat: add training script for BailingMoE Linear V2"
```

---

## Task 4: Create Example Documentation

**Files:**
- Create: `examples/bailing_moe_linear_v2/README.md`

**Step 1: Create example directory**

Run: `mkdir -p examples/bailing_moe_linear_v2`

**Step 2: Create README with usage instructions**

Write comprehensive README covering:
- Prerequisites and installation
- Quick start guide
- Configuration reference
- Performance tuning
- Troubleshooting

**Step 3: Commit**

```bash
git add examples/bailing_moe_linear_v2/README.md
git commit -m "docs: add example documentation for BailingMoE Linear V2"
```

---

## Task 5: Create Validation Script

**Files:**
- Create: `examples/bailing_moe_linear_v2/validate_config.py`

**Step 1: Create validation script**

Implement Python script that validates:
- Linear attention pattern
- MoE pattern
- MoE configuration (group-limited topk)
- Architecture parameters

**Step 2: Make script executable**

Run: `chmod +x examples/bailing_moe_linear_v2/validate_config.py`

**Step 3: Test validation script**

Run: `python examples/bailing_moe_linear_v2/validate_config.py --help`

Expected: Help message displayed

**Step 4: Commit**

```bash
git add examples/bailing_moe_linear_v2/validate_config.py
git commit -m "feat: add configuration validation script"
```

---

## Success Criteria

- ✅ Custom MTP layer spec created with final_layernorm
- ✅ Layer spec function handles hybrid attention and MLP patterns
- ✅ Training script includes all parameter mappings
- ✅ Documentation provides clear usage instructions
- ✅ Validation script catches common configuration errors
- ✅ All files committed to git with clear commit messages

## Next Steps

After completing this implementation plan:

1. **Test Model Initialization**: Run a quick test to verify the model initializes correctly
2. **Validate Configuration**: Use the validation script to check all parameters
3. **Small-Scale Training**: Run training for a few iterations to verify everything works
4. **Performance Profiling**: Profile the training to identify any bottlenecks

## References

- Design Document: `docs/plans/2026-03-08-bailing-moe-linear-v2-training-design.md`
- Reference Code: `ref/ling2.5-pretrain/bailing_moe_modeling.py`
- Reference Script: `ref/ling2.5-pretrain/bailing_moe_linear_v2_megatron.sh`
