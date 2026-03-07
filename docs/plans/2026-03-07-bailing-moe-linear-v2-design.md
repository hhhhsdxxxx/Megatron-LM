# BailingMoeLinearV2 Integration Design

**Date:** 2026-03-07
**Author:** Claude (with user collaboration)
**Status:** Approved

## Executive Summary

This document describes the design for integrating the BailingMoeLinearV2 model architecture into Megatron-LM. The model combines:
- **Gated Linear Attention (GLA)** for efficient long-context modeling
- **Multi-Latent Attention (MLA)** for compressed KV cache
- **Mixture of Experts (MoE)** with sigmoid-based group-limited routing
- **Multi-Token Prediction (MTP)** with fused embedding variant

The design follows a **hybrid approach** that extends existing Megatron components while adding new specialized modules, maximizing code reuse and maintainability.

## Architecture Overview

### Model Structure

The model uses a **hybrid layer architecture**:
- **Linear Attention Layers**: Use GLA for most layers (efficient O(n) complexity)
- **Standard Attention Layers**: Use MLA every N layers (controlled by `layer_group_size`)
- **MoE Layers**: Replace dense FFN in specified layers
- **MTP Layers**: Additional prediction layers after main transformer

```
Layer Pattern (layer_group_size=5):
├── Layer 0-3: Linear Attention (GLA) + MoE
├── Layer 4: Standard Attention (MLA) + MoE
├── Layer 5-8: Linear Attention (GLA) + MoE
├── Layer 9: Standard Attention (MLA) + MoE
└── ...
└── MTP Layers (1-N additional layers)
```

### Key Design Decisions

1. **Reuse Megatron's TopKRouter**: No need for custom SigmoidGroupedRouter
   - Megatron already supports sigmoid score function
   - Supports group-limited topk routing
   - Supports expert bias with dynamic updates

2. **Reuse Megatron's MLA**: Existing MultiLatentAttention implementation
   - Only used in standard attention layers
   - Not used in linear attention layers

3. **External Dependency**: Use `flash-linear-attention` library
   - Provides optimized GLA kernels (chunk_simple_gla, fused_recurrent_simple_gla)
   - High-performance CUDA implementations

4. **Layer Spec Mechanism**: Dynamic layer type selection
   - Use Megatron's ModuleSpec system
   - Select attention type based on layer_idx and layer_group_size

## Component Design

### 1. Configuration Extensions

**New Parameters in `TransformerConfig`:**

```python
# Linear Attention
layer_group_size: int = 1  # Standard attention every N layers
use_linear_attention: bool = False
linear_attn_norm_group_size: int = 4  # GroupRMSNorm group size
linear_attn_norm_group_type: str = "group_diff"

# MoE Router (additions to existing params)
moe_router_num_groups: Optional[int] = None  # Expert groups
moe_router_group_topk: Optional[int] = None  # Groups per token
moe_router_topk_scaling_factor: float = 1.0  # Weight scaling
moe_router_bias_update_rate: float = 1e-3  # Bias update rate

# MTP
mtp_variant: str = "standard"  # 'standard' or 'fused'
mtp_loss_scaling_factor: float = 0.1
```

**Command-line Argument Mapping:**

```bash
--layer-group-size 5 → layer_group_size
--linear-attn-norm-group-size 4 → linear_attn_norm_group_size
--moe-router-num-groups 8 → moe_router_num_groups
--moe-router-group-topk 4 → moe_router_group_topk
--moe-router-topk-scaling-factor 2.5 → moe_router_topk_scaling_factor
--mtp-variant fused → mtp_variant
--mtp-loss-scaling-factor 0.1 → mtp_loss_scaling_factor
```

### 2. GroupRMSNorm Implementation

**File:** `megatron/core/transformer/group_rms_norm.py`

GroupRMSNorm divides the hidden dimension into groups and normalizes each group independently.

**Key Implementation Details:**

```python
class GroupRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, group_norm_size: int, eps: float = 1e-6):
        # hidden_size must be divisible by group_norm_size
        # Example: hidden_size=2048, group_norm_size=4
        #   -> 512 groups, each with 4 elements

    def forward(self, hidden_states):
        # Reshape: [..., hidden_size]
        #   -> [..., group_norm_size, hidden_size // group_norm_size]
        # Compute variance on last dim
        # Apply normalization per group
```

**Difference from Standard RMSNorm:**
- Standard RMSNorm: Variance computed over entire hidden_size
- GroupRMSNorm: Variance computed over each group independently

### 3. Linear Attention (GLA) Implementation

**File:** `megatron/core/transformer/linear_attention.py`

Implements Gated Linear Attention based on Lightning Attention-2.

**Architecture:**

```
Input
  ↓
QKV Projection
  ↓
QK Norm (optional)
  ↓
RoPE (unsqueeze_dim=2)
  ↓
GLA Kernel (chunk or fused_recurrent)
  ↓
GroupRMSNorm
  ↓
Gating (sigmoid(g_proj(input)) * output)
  ↓
Output Projection
```

**Key Components:**

1. **Slope Tensor**: Layer-dependent decay factor
   ```python
   base_slope = build_slope_tensor(num_heads)
   layer_factor = 1 - (layer_idx - 1) / max(num_layers - 1, 1) + 1e-5
   slope = -base_slope * layer_factor
   ```
   - Controls attention score decay rate
   - Varies by layer: deeper layers have smaller absolute slope
   - Uses `max(num_layers - 1, 1)` to prevent division by zero when `num_layers == 1`

2. **Mode Selection**:
   - `seq_len <= 64`: Use `fused_recurrent` (faster for short sequences)
   - `seq_len > 64`: Use `chunk` (better for long sequences)

3. **Attention Mask**:
   - Only supports 2D mask `[batch, seq]` for padding
   - Does NOT support arbitrary 4D causal masks

4. **KV Cache**:
   - Stores `recurrent_state` instead of key/value tensors
   - State shape: `[batch, num_heads, head_dim, head_dim]`
   - Special handling for left-padding in first generation step

5. **Tensor Layout**:
   - Uses `head_first=False`: `[batch, seq, num_heads, head_dim]`
   - RoPE uses `unsqueeze_dim=2` (not 1)

### 4. MoE Router Configuration

**No new router class needed!** Use existing `TopKRouter` with proper configuration:

```python
# In TransformerConfig
moe_router_score_function = "sigmoid"  # Not softmax
moe_router_num_groups = 8  # Expert groups
moe_router_group_topk = 4  # Groups to select
moe_router_enable_expert_bias = True
moe_router_topk_scaling_factor = 2.5
```

**How it works:**

1. Compute logits: `logits = linear(hidden, weight)`
2. Apply sigmoid: `scores = sigmoid(logits)`
3. Add expert bias: `scores_for_routing = scores + expert_bias`
4. Group-limited topk:
   - Organize experts into `num_groups` groups
   - Select top-2 experts per group
   - Select top `group_topk` groups
   - Final topk from selected groups
5. Normalize and scale: `probs = scores / sum(scores) * scaling_factor`

**Expert Bias Update:**

```python
# During training, TopKRouter automatically updates expert_bias
# based on token distribution to balance load
```

### 5. MTP Layer Extension

**File:** `megatron/core/transformer/multi_token_prediction.py`

Extend `MultiTokenPredictionBlock` to support two variants:

**Standard MTP:**
```
hidden_states → roll → transformer_layer → predict
```

**Fused MTP:**
```
input_embeds → enorm ─┐
                       ├→ concat → eh_proj → transformer_layer → final_norm → predict
hidden_states → hnorm ─┘
```

**Key Implementation:**

```python
class MultiTokenPredictionBlock:
    def __init__(self, config, transformer_layer_spec, num_mtp_layers, mtp_variant):
        if mtp_variant == 'fused':
            self.enorm = RMSNorm(...)
            self.hnorm = RMSNorm(...)
            self.eh_proj = Linear(hidden_size * 2, hidden_size)
            self.final_layernorm = RMSNorm(...)

        self.layers = [transformer_layer_spec(...) for _ in range(num_mtp_layers)]

    def forward(self, hidden_states, input_ids, embedding_module):
        for layer in self.layers:
            shifted_input_ids = roll(input_ids, -1)

            if self.mtp_variant == 'fused':
                embeds = self.enorm(embedding_module(shifted_input_ids))
                hidden = self.hnorm(hidden_states)
                hidden_states = self.eh_proj(torch.cat([embeds, hidden], -1))

            hidden_states = layer(hidden_states)

            if self.mtp_variant == 'fused':
                hidden_states = self.final_layernorm(hidden_states)
```

### 6. Layer Spec Definition

**File:** `megatron/core/models/gpt/bailing_moe_linear_v2_layer_specs.py`

Dynamic layer specification based on layer index:

```python
def get_gpt_layer_with_linear_attention_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    layer_group_size: int = 5,
) -> ModuleSpec:

    def _get_attention_module(layer_idx: int):
        # Standard attention every layer_group_size layers
        if (layer_idx + 1) % layer_group_size == 0:
            return MLASelfAttentionSubmodules(...)  # MLA
        else:
            return SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,
                core_attention=LinearAttention,  # GLA
                linear_proj=TERowParallelLinear,
                linear_g_proj=TELinear,  # For gating
                q_layernorm=TENorm if qk_layernorm else IdentityOp,
                k_layernorm=TENorm if qk_layernorm else IdentityOp,
            )

    mlp_spec = _get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=_get_attention_module,  # Dynamic!
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=mlp_spec,
            mlp_bda=get_bias_dropout_add,
        ),
    )
```

### 7. GPTModel Integration

**File:** `megatron/core/models/gpt/gpt_model.py`

**Key Modifications:**

1. **Attention Mask Handling**:
   ```python
   for layer_idx, layer in enumerate(self.decoder.layers):
       is_linear_attn = (layer_idx + 1) % config.layer_group_size != 0

       if is_linear_attn:
           # Convert 4D mask to 2D for linear attention
           if attention_mask.dim() == 4:
               mask_2d = attention_mask[:, 0, -1, :].to(torch.int32)
               mask_2d = (mask_2d > -1e4).to(torch.int32)
           else:
               mask_2d = attention_mask
           layer_mask = mask_2d
       else:
           layer_mask = attention_mask  # Keep 4D for standard attention

       hidden_states = layer(hidden_states, attention_mask=layer_mask, ...)
   ```

2. **MTP Integration**:
   ```python
   if self.mtp_block is not None:
       mtp_hidden_states = self.mtp_block(
           hidden_states=hidden_states,
           input_ids=input_ids,
           embedding_module=self.embedding.word_embeddings,
           attention_mask=attention_mask,
       )
   ```

3. **Loss Computation**:
   ```python
   if labels is not None and mtp_hidden_states is not None:
       shift_labels = labels.clone()
       for mtp_hidden in mtp_hidden_states:
           shift_labels = roll(shift_labels, -1, fill_value=-100)
           mtp_logits = self.output_layer(mtp_hidden)
           mtp_loss = loss_func(mtp_logits, shift_labels)
           total_loss += mtp_loss * config.mtp_loss_scaling_factor
   ```

## Implementation Checklist

### New Files to Create

- [ ] `megatron/core/transformer/group_rms_norm.py`
- [ ] `megatron/core/transformer/linear_attention.py`
- [ ] `megatron/core/models/gpt/bailing_moe_linear_v2_layer_specs.py`

### Files to Modify

- [ ] `megatron/core/transformer/transformer_config.py` - Add new config parameters
- [ ] `megatron/core/transformer/multi_token_prediction.py` - Add fused MTP variant
- [ ] `megatron/core/models/gpt/gpt_model.py` - Integrate linear attention and MTP
- [ ] `megatron/training/arguments.py` - Add command-line arguments
- [ ] `pretrain_gpt.py` - Use new layer spec

### Dependencies

- [ ] Ensure `flash-linear-attention` is installed and accessible
- [ ] Verify Megatron's MLA implementation is available
- [ ] Verify Megatron's TopKRouter supports all required features

## Testing Strategy

### Unit Tests

1. **GroupRMSNorm**:
   - Test output shape preservation
   - Test numerical correctness vs reference implementation
   - Test gradient flow

2. **LinearAttention**:
   - Test with different sequence lengths (trigger both modes)
   - Test with/without KV cache
   - Test with padding masks
   - Test slope tensor computation

3. **MTP Block**:
   - Test both standard and fused variants
   - Test loss computation
   - Test with multiple MTP layers

### Integration Tests

1. **Layer Spec**:
   - Verify correct attention type selection per layer
   - Test with different layer_group_size values

2. **End-to-End**:
   - Small model training run (few iterations)
   - Verify loss convergence
   - Verify checkpoint save/load

### Validation

1. **Numerical Equivalence**:
   - Compare outputs with reference implementation (bailing_moe_linear_v2.py)
   - Test on same inputs with same random seed

2. **Performance**:
   - Measure throughput vs reference
   - Verify memory usage is reasonable

## Migration Path

### Phase 1: Core Components (Week 1)
1. Implement GroupRMSNorm
2. Implement LinearAttention
3. Unit tests for both

### Phase 2: Integration (Week 2)
1. Create layer spec
2. Modify GPTModel
3. Add command-line arguments
4. Integration tests

### Phase 3: MTP and MoE (Week 3)
1. Extend MTP block
2. Configure MoE router
3. End-to-end tests

### Phase 4: Validation (Week 4)
1. Numerical equivalence testing
2. Performance benchmarking
3. Documentation

## Risk Mitigation

### Risk 1: flash-linear-attention Compatibility
- **Mitigation**: Pin specific version, test installation in CI
- **Fallback**: Implement basic GLA in pure PyTorch (slower but functional)

### Risk 2: KV Cache Handling
- **Mitigation**: Extensive testing of inference with different sequence lengths
- **Fallback**: Disable caching for linear attention layers initially

### Risk 3: Numerical Differences
- **Mitigation**: Careful dtype handling, match reference implementation exactly
- **Fallback**: Accept small differences if training converges

## Open Questions

1. **Q**: Should we support mixed precision (FP8) for linear attention?
   **A**: Defer to Phase 2, focus on BF16/FP16 first

2. **Q**: How to handle very long sequences (>100K tokens)?
   **A**: Rely on flash-linear-attention's chunk mode, test separately

3. **Q**: Should MTP layers also use linear attention?
   **A**: Follow main model's layer pattern (configurable via layer_group_size)

## References

- Lightning Attention-2 Paper: https://arxiv.org/abs/2401.04658
- flash-linear-attention: https://github.com/fla-org/flash-linear-attention
- Megatron-LM Documentation: https://github.com/NVIDIA/Megatron-LM
- DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434

## Appendix: Training Script Example

```bash
#!/bin/bash
# Example training command with new parameters

torchrun --nproc_per_node 8 pretrain_gpt.py \
    # Model architecture
    --num-layers 20 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --layer-group-size 5 \
    --linear-attn-norm-group-size 4 \
    \
    # MLA (for standard attention layers)
    --multi-latent-attention \
    --q-lora-rank 256 \
    --kv-lora-rank 512 \
    \
    # MoE
    --num-experts 256 \
    --moe-router-topk 8 \
    --moe-router-score-function sigmoid \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-router-enable-expert-bias \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-ffn-hidden-size 512 \
    --moe-shared-expert-intermediate-size 2048 \
    \
    # MTP
    --mtp-num-layers 1 \
    --mtp-variant fused \
    --mtp-loss-scaling-factor 0.1 \
    \
    # Training
    --micro-batch-size 2 \
    --global-batch-size 512 \
    --seq-length 4096 \
    --bf16 \
    --use-flash-attn
```

---

**Design Approved:** 2026-03-07
**Next Step:** Create implementation plan
