# BailingMoE Linear V2 Training Implementation Design

**Date**: 2026-03-08
**Author**: Claude (Brainstorming Session)
**Status**: Approved

## Overview

This document describes the design for implementing BailingMoE Linear V2 model training in Megatron-LM. The implementation follows a **minimal modification approach**, maximally reusing existing Megatron-LM components while ensuring full compatibility with the reference implementation.

## Reference Materials

- Reference Model Code: `ref/ling2.5-pretrain/bailing_moe_modeling.py`
- Reference Training Script: `ref/ling2.5-pretrain/bailing_moe_linear_v2_megatron.sh`

## Architecture Design

### Model Architecture

BailingMoE Linear V2 is a 20-layer decoder-only transformer with the following key features:

**1. Hybrid Attention Pattern**
- Pattern: 4 Linear Attention layers + 1 Standard Attention layer, repeated 4 times
- Total: 20 layers (layers 0-3, 5-8, 10-13, 15-18 use Linear Attention; layers 4, 9, 14, 19 use Standard Attention)
- Linear Attention uses Gated Linear Attention (GLA) from flash-linear-attention library

**2. Hybrid MLP Pattern**
- Layer 0: Dense MLP (intermediate_size = 5120)
- Layers 1-19: Sparse MoE
  - 256 routed experts (each with FFN hidden size = 512)
  - 1 shared expert (intermediate size = 2048)
  - Group-Limited TopK Router: 8 groups, select 4 groups, then select 8 experts

**3. Multi-Token Prediction (MTP)**
- 1 MTP layer after the main decoder layers
- Predicts the next token at each position
- Custom implementation with final_layernorm (matching reference code)

### Component Details

#### Linear Attention
- Uses `fla.ops.simple_gla` operators (chunk_simple_gla, fused_recurrent_simple_gla)
- Layer-dependent slope for decay control
- Gate projection with Group RMS Norm (group_size = 4)
- Partial rotary embeddings (50% of head_dim)

#### MoE Router (Group-Limited TopK)
- Score function: sigmoid
- Expert bias enabled with dynamic update
- Routing process:
  1. Divide 256 experts into 8 groups (32 experts per group)
  2. Select 4 groups based on sum of top-2 expert scores within each group
  3. Select 8 experts from the selected groups
- Scaling factor: 2.5 applied to routing weights

#### Shared Expert
- Single shared expert with intermediate_size = 2048
- Output added to routed expert outputs
- Equivalent to 4 routed experts in capacity (2048 / 512 = 4)

## Implementation Strategy

### Approach: Minimal Modification

**Principle**: Maximize reuse of Megatron-LM existing components, only add necessary customization.

### Component Reuse Matrix

| Component | Megatron-LM Implementation | Status | Notes |
|-----------|---------------------------|--------|-------|
| Linear Attention | `megatron/core/transformer/linear_attention.py` | ✅ Direct Use | Identical implementation |
| Group-Limited TopK Router | `megatron/core/transformer/moe/router.py` | ✅ Direct Use | Fully supports all features |
| Shared Experts | `megatron/core/transformer/moe/shared_experts.py` | ✅ Direct Use | Standard implementation |
| Group RMS Norm | `megatron/core/transformer/group_rms_norm.py` | ✅ Direct Use | Identical implementation |
| Rotary Embedding | `megatron/core/models/common/embeddings/rotary_pos_embedding.py` | ✅ Direct Use | Supports partial rotation |
| MTP Layer | `megatron/core/transformer/multi_token_prediction.py` | ⚠️ Custom Spec | Need to add final_layernorm |

### Implementation Differences

#### Only Significant Difference: MTP Layer final_layernorm

**Reference Code**:
```python
class BailingMoeV2MTPLayer(nn.Module):
    def forward(self, input_embeds, hidden_states, ...):
        # ... enorm, hnorm, eh_proj, attention, MoE, post_attention_layernorm
        hidden_states = self.final_layernorm(hidden_states)  # ← This line
        return outputs
```

**Megatron-LM**:
```python
class MultiTokenPredictionLayer(MegatronModule):
    def forward(self, hidden_states, input_ids, ...):
        # ... enorm, hnorm, eh_proj, transformer layer
        # No final_layernorm
        return outputs
```

**Solution**: Create custom MTP layer spec that adds final_layernorm after the transformer layer.

## Parameter Configuration Mapping

### Complete Mapping Table

| Reference Config | Megatron-LM Parameter | Value |
|-----------------|----------------------|-------|
| **Model Architecture** |
| num_layers: 20 | --num-layers | 20 |
| hidden_size: 2048 | --hidden-size | 2048 |
| num_attention_heads: 16 | --num-attention-heads | 16 |
| num_key_value_heads: 4 | --num-query-groups | 4 |
| intermediate_size: 5120 | --ffn-hidden-size | 5120 |
| vocab_size: 157184 | --vocab-size | 157184 |
| max_position_embeddings: 4096 | --max-position-embeddings | 4096 |
| **Attention Configuration** |
| layer_group_size: 5 | --linear-attention-freq | "([1]*4+[0]*1)*4" |
| use_qk_norm: True | --qk-layernorm | true |
| use_qkv_bias: True | (omit --disable-bias-linear) | - |
| attention_dropout: 0 | --attention-dropout | 0 |
| hidden_dropout: 0 | --hidden-dropout | 0 |
| **Rotary Embeddings** |
| partial_rotary_factor: 0.5 | --rotary-percent | 0.5 |
| rope_theta: 10000 | --rotary-base | 10000 |
| **MoE Configuration** |
| num_experts: 256 | --num-experts | 256 |
| num_experts_per_tok: 8 | --moe-router-topk | 8 |
| n_group: 8 | --moe-router-num-groups | 8 |
| topk_group: 4 | --moe-router-group-topk | 4 |
| moe_intermediate_size: 512 | --moe-ffn-hidden-size | 512 |
| num_shared_experts: 1 | --moe-shared-expert-intermediate-size | 2048 |
| routed_scaling_factor: 2.5 | --moe-router-topk-scaling-factor | 2.5 |
| first_k_dense_replace: 1 | --moe-layer-freq | "([0]+[1]*19)" |
| moe_aux_loss_coeff: 0.0000035 | --moe-aux-loss-coeff | 0.0000035 |
| score_function: sigmoid | --moe-router-score-function | sigmoid |
| enable_expert_bias: True | --moe-router-enable-expert-bias | true |
| **Linear Attention** |
| group_norm_size: 4 | --linear-attn-norm-group-size | 4 |
| **MTP Configuration** |
| num_nextn_predict_layers: 1 | --mtp-num-layers | 1 |
| mtp_loss_scaling_factor: 0.1 | --mtp-loss-scaling-factor | 0.1 |
| **Training Configuration** |
| micro_batch_size: 2 | --micro-batch-size | 2 |
| global_batch_size: 5120 | --global-batch-size | 5120 |
| seq_length: 4096 | --seq-length | 4096 |
| train_iters: 95367 | --train-iters | 95367 |
| lr: 0.000336 | --lr | 0.000336 |
| min_lr: 0.000336 | --min-lr | 0.000336 |
| lr_warmup_iters: 2000 | --lr-warmup-iters | 2000 |
| weight_decay: 0.1 | --weight-decay | 0.1 |
| clip_grad: 1.0 | --clip-grad | 1.0 |
| adam_beta1: 0.9 | --adam-beta1 | 0.9 |
| adam_beta2: 0.95 | --adam-beta2 | 0.95 |
| init_method_std: 0.006 | --init-method-std | 0.006 |

## Implementation Plan

### Files to Create

1. **Layer Spec File**: `megatron/core/models/gpt/bailing_moe_linear_v2_layer_specs.py`
   - Define custom MTP layer spec with final_layernorm
   - Define layer spec for BailingMoE Linear V2 model
   - Handle hybrid attention pattern (Linear + Standard)
   - Handle hybrid MLP pattern (Dense + MoE)

2. **Training Script**: `pretrain_bailing_moe_linear_v2.sh`
   - Complete parameter configuration
   - Environment setup (NCCL, CUDA, flash-linear-attention path)
   - GPU detection and distributed training setup
   - Logging and checkpointing configuration

3. **Example Directory**: `examples/bailing_moe_linear_v2/`
   - README with usage instructions
   - Configuration examples
   - Performance tuning guide

### Implementation Steps

**Step 1: Create Custom MTP Layer Spec**
- Extend `MultiTokenPredictionLayerSubmodules`
- Add `final_layernorm` to the spec
- Ensure compatibility with existing MTP infrastructure

**Step 2: Create BailingMoE Linear V2 Layer Spec**
- Define function `get_bailing_moe_linear_v2_layer_local_spec()`
- Use `ModuleSpec` to compose:
  - Linear Attention (for layers matching pattern)
  - Standard Attention (for layers matching pattern)
  - Dense MLP (for layer 0)
  - MoE (for layers 1-19)
- Return `TransformerLayerSubmodules` with proper configuration

**Step 3: Create Training Script**
- Copy structure from reference script
- Map all parameters according to the mapping table
- Add comments explaining each configuration section
- Include validation checks for critical parameters

**Step 4: Validation**
- Verify model initialization
- Check parameter counts match expected values
- Validate layer types at each position
- Test with small batch to ensure forward pass works

## Data Flow

### Training Forward Pass

```
Input IDs [batch, seq_len]
    ↓
Embedding Layer
    ↓
┌─────────────────────────────────────┐
│ Main Decoder Layers (20 layers)    │
│                                     │
│ Layer 0:  Linear Attn + Dense MLP  │
│ Layer 1:  Linear Attn + MoE        │
│ Layer 2:  Linear Attn + MoE        │
│ Layer 3:  Linear Attn + MoE        │
│ Layer 4:  Standard Attn + MoE      │
│ Layer 5:  Linear Attn + MoE        │
│ ...                                 │
│ Layer 19: Standard Attn + MoE      │
└─────────────────────────────────────┘
    ↓
Final RMS Norm
    ↓ (main_hidden_states)
    ├─────────────────────────────────┐
    ↓                                 ↓
LM Head                          MTP Layer
    ↓                                 ↓
Main Logits                      MTP Logits
    ↓                                 ↓
Main Loss                        MTP Loss (scaled by 0.1)
    └─────────────────────────────────┘
                    ↓
            Total Loss = Main Loss + 0.1 * MTP Loss
```

### MTP Layer Detail

```
Input: (input_embeds, hidden_states)
    ↓
enorm(input_embeds) + hnorm(hidden_states)
    ↓
eh_proj: [batch, seq, 2*hidden] → [batch, seq, hidden]
    ↓
input_layernorm
    ↓
Attention (Standard or Linear based on config)
    ↓
Residual Connection
    ↓
post_attention_layernorm
    ↓
MoE Layer
    ↓
Residual Connection
    ↓
final_layernorm  ← Custom addition
    ↓
Output
```

## Testing Strategy

### Unit Tests
- Test custom MTP layer spec creation
- Verify final_layernorm is properly added
- Test layer spec generation for different layer indices

### Integration Tests
- Initialize full model and verify layer types
- Check parameter counts:
  - Total parameters should match reference model
  - MoE parameters should be correctly distributed
- Verify forward pass with dummy data

### Training Tests
- Small-scale training run (few iterations)
- Verify loss computation (main + MTP)
- Check gradient flow through all components
- Validate checkpoint saving/loading

## Performance Considerations

### Memory Optimization
- Use `--sequence-parallel` for long sequences
- Enable `--use-distributed-optimizer` for large models
- Use `--recompute-granularity selective` for activation checkpointing

### Communication Optimization
- `--overlap-param-gather` and `--overlap-grad-reduce` for communication overlap
- `--moe-token-dispatcher-type flex` for efficient MoE communication
- `--moe-grouped-gemm` for optimized expert computation

### Computation Optimization
- `--use-flash-attn` for standard attention layers
- Flash-linear-attention operators for linear attention layers
- `--fp8-param-gather` for reduced precision communication

## Risks and Mitigations

### Risk 1: MTP Layer Behavior Difference
**Risk**: Custom final_layernorm might affect training dynamics
**Mitigation**:
- Compare loss curves with reference implementation
- Monitor gradient norms
- If issues arise, can disable MTP temporarily

### Risk 2: Flash-Linear-Attention Dependency
**Risk**: External library dependency might cause compatibility issues
**Mitigation**:
- Pin specific version in requirements
- Document installation steps clearly
- Provide fallback to standard attention if needed

### Risk 3: Parameter Mismatch
**Risk**: Subtle parameter differences might affect convergence
**Mitigation**:
- Detailed parameter mapping documentation
- Validation script to check all parameters
- Compare model outputs on same inputs

## Success Criteria

1. ✅ Model initializes successfully with correct architecture
2. ✅ All 20 decoder layers have correct attention and MLP types
3. ✅ MTP layer includes final_layernorm
4. ✅ Parameter count matches reference model (within 1%)
5. ✅ Training runs without errors for 100 iterations
6. ✅ Loss decreases over training iterations
7. ✅ Checkpoint saving and loading works correctly
8. ✅ Training script is well-documented and easy to use

## Future Enhancements

1. **Performance Profiling**: Detailed profiling to identify bottlenecks
2. **Hyperparameter Tuning**: Systematic tuning of learning rate, batch size, etc.
3. **Multi-Node Training**: Validation on multi-node setups
4. **Inference Optimization**: Optimize for inference performance
5. **Model Compression**: Explore quantization and pruning techniques

## References

- DeepSeek-V2 Paper: https://arxiv.org/pdf/2405.04434
- DeepSeek-V3 Paper: https://arxiv.org/pdf/2412.19437
- Lightning Attention-2 Paper: https://arxiv.org/abs/2401.04658
- Flash-Linear-Attention Library: https://github.com/sustcsonglin/flash-linear-attention
- Megatron-LM Documentation: https://github.com/NVIDIA/Megatron-LM

## Appendix: Key Design Decisions

### Decision 1: Use Megatron-LM's Group-Limited TopK Router
**Rationale**: Megatron-LM already implements this feature for DeepSeek-V3, fully compatible with reference code.

### Decision 2: Custom MTP Layer Spec with final_layernorm
**Rationale**: Ensures exact match with reference implementation, minimal code change.

### Decision 3: Single Training Script Approach
**Rationale**: Easier to understand and modify, follows reference script structure.

### Decision 4: Set num_shared_experts=1 when moe-shared-expert-intermediate-size is specified
**Rationale**: Simplifies configuration, matches common usage pattern.
