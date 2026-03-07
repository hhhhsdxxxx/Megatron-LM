# BailingMoeLinearV2 Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate BailingMoeLinearV2 model architecture (GLA + MLA + MoE + MTP) into Megatron-LM

**Architecture:** Hybrid approach extending existing Megatron components with new LinearAttention and GroupRMSNorm modules, using layer_spec mechanism for dynamic layer type selection

**Tech Stack:** PyTorch, Megatron-LM, flash-linear-attention, Transformer Engine

---

## Phase 1: Core Components

### Task 1: Implement GroupRMSNorm

**Files:**
- Create: `megatron/core/transformer/group_rms_norm.py`
- Test: `tests/unit_tests/transformer/test_group_rms_norm.py`

**Step 1: Write the failing test**

Create test file with basic shape preservation test:

```python
# tests/unit_tests/transformer/test_group_rms_norm.py
import pytest
import torch
from megatron.core.transformer.group_rms_norm import GroupRMSNorm


def test_group_rms_norm_shape_preservation():
    """Test that GroupRMSNorm preserves input shape"""
    batch_size, seq_len, hidden_size = 2, 128, 2048
    group_norm_size = 4

    norm = GroupRMSNorm(hidden_size=hidden_size, group_norm_size=group_norm_size)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)

    output = norm(input_tensor)

    assert output.shape == input_tensor.shape
    assert output.dtype == input_tensor.dtype


def test_group_rms_norm_invalid_group_size():
    """Test that invalid group_norm_size raises error"""
    with pytest.raises(AssertionError):
        GroupRMSNorm(hidden_size=2048, group_norm_size=3)  # Not divisible
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit_tests/transformer/test_group_rms_norm.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'megatron.core.transformer.group_rms_norm'"

**Step 3: Write minimal implementation**

```python
# megatron/core/transformer/group_rms_norm.py
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch


class GroupRMSNorm(torch.nn.Module):
    """
    Group RMS Normalization.

    Divides hidden_size into groups and normalizes each group independently.
    This differs from standard RMSNorm which normalizes over the entire hidden dimension.

    Args:
        hidden_size (int): Total hidden dimension size
        group_norm_size (int): Size of each normalization group
        eps (float): Small constant for numerical stability
    """

    def __init__(self, hidden_size: int, group_norm_size: int, eps: float = 1e-6):
        super().__init__()
        assert hidden_size % group_norm_size == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"group_norm_size ({group_norm_size})"
        )

        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.group_norm_size = group_norm_size
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape [..., hidden_size]

        Returns:
            Normalized tensor of same shape as input
        """
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.size()

        # Reshape to separate groups
        # [..., hidden_size] -> [..., group_norm_size, hidden_size // group_norm_size]
        group_input_shape = input_shape[:-1] + (
            self.group_norm_size,
            input_shape[-1] // self.group_norm_size
        )
        hidden_states = hidden_states.view(group_input_shape)

        # Compute RMS normalization in fp32 for stability
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Restore original shape and dtype
        hidden_states = hidden_states.to(input_dtype).view(input_shape)
        return self.weight * hidden_states
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit_tests/transformer/test_group_rms_norm.py -v`
Expected: PASS (2 tests)

**Step 5: Add numerical correctness test**

Add to test file:

```python
def test_group_rms_norm_numerical_correctness():
    """Test numerical correctness against reference implementation"""
    torch.manual_seed(42)
    batch_size, seq_len, hidden_size = 2, 4, 8
    group_norm_size = 2

    norm = GroupRMSNorm(hidden_size=hidden_size, group_norm_size=group_norm_size, eps=1e-6)
    # Initialize weight to ones (default)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)

    output = norm(input_tensor)

    # Manual computation for verification
    reshaped = input_tensor.view(batch_size, seq_len, group_norm_size, hidden_size // group_norm_size)
    variance = reshaped.pow(2).mean(-1, keepdim=True)
    expected = reshaped * torch.rsqrt(variance + 1e-6)
    expected = expected.view(batch_size, seq_len, hidden_size)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
```

**Step 6: Run numerical test**

Run: `pytest tests/unit_tests/transformer/test_group_rms_norm.py::test_group_rms_norm_numerical_correctness -v`
Expected: PASS

**Step 7: Commit GroupRMSNorm**

```bash
git add megatron/core/transformer/group_rms_norm.py tests/unit_tests/transformer/test_group_rms_norm.py
git commit -m "feat: add GroupRMSNorm for linear attention

- Implements group-wise RMS normalization
- Divides hidden_size into groups for independent normalization
- Includes comprehensive unit tests

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Add Configuration Parameters

**Files:**
- Modify: `megatron/core/transformer/transformer_config.py`
- Test: `tests/unit_tests/transformer/test_transformer_config.py`

**Step 1: Write the failing test**

Add to existing test file:

```python
def test_linear_attention_config_parameters():
    """Test that linear attention config parameters are accessible"""
    config = TransformerConfig(
        num_layers=20,
        hidden_size=2048,
        num_attention_heads=16,
        layer_group_size=5,
        linear_attn_norm_group_size=4,
        linear_attn_norm_group_type="group_diff",
    )

    assert config.layer_group_size == 5
    assert config.linear_attn_norm_group_size == 4
    assert config.linear_attn_norm_group_type == "group_diff"


def test_moe_router_extended_config():
    """Test extended MoE router configuration parameters"""
    config = TransformerConfig(
        num_layers=20,
        hidden_size=2048,
        num_attention_heads=16,
        moe_router_num_groups=8,
        moe_router_group_topk=4,
        moe_router_topk_scaling_factor=2.5,
        moe_router_bias_update_rate=1e-3,
    )

    assert config.moe_router_num_groups == 8
    assert config.moe_router_group_topk == 4
    assert config.moe_router_topk_scaling_factor == 2.5
    assert config.moe_router_bias_update_rate == 1e-3


def test_mtp_variant_config():
    """Test MTP variant configuration"""
    config = TransformerConfig(
        num_layers=20,
        hidden_size=2048,
        num_attention_heads=16,
        mtp_variant="fused",
        mtp_loss_scaling_factor=0.1,
    )

    assert config.mtp_variant == "fused"
    assert config.mtp_loss_scaling_factor == 0.1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit_tests/transformer/test_transformer_config.py::test_linear_attention_config_parameters -v`
Expected: FAIL with "TypeError: __init__() got an unexpected keyword argument"

**Step 3: Add configuration parameters**

Modify `megatron/core/transformer/transformer_config.py`:

```python
@dataclass
class TransformerConfig(ModelParallelConfig):
    # ... existing parameters ...

    # Linear Attention parameters
    layer_group_size: int = 1
    """Number of layers per group. Standard attention used every N layers."""

    use_linear_attention: bool = False
    """Whether to use linear attention (GLA) in non-standard-attention layers."""

    linear_attn_norm_group_size: int = 4
    """Group size for GroupRMSNorm in linear attention layers."""

    linear_attn_norm_group_type: str = "group_diff"
    """Type of group normalization: 'group_diff' or 'group_same'."""

    # MoE Router extended parameters
    moe_router_num_groups: Optional[int] = None
    """Number of expert groups for group-limited routing."""

    moe_router_group_topk: Optional[int] = None
    """Number of groups to select per token in group-limited routing."""

    moe_router_topk_scaling_factor: float = 1.0
    """Scaling factor applied to top-k routing weights."""

    moe_router_bias_update_rate: float = 1e-3
    """Learning rate for expert bias updates during training."""

    # MTP parameters
    mtp_variant: str = "standard"
    """MTP variant: 'standard' or 'fused'."""

    mtp_loss_scaling_factor: float = 0.1
    """Scaling factor for MTP loss contribution to total loss."""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit_tests/transformer/test_transformer_config.py::test_linear_attention_config_parameters -v`
Expected: PASS (3 new tests)

**Step 5: Commit configuration changes**

```bash
git add megatron/core/transformer/transformer_config.py tests/unit_tests/transformer/test_transformer_config.py
git commit -m "feat: add config parameters for linear attention, MoE, and MTP

- Add layer_group_size for hybrid attention layers
- Add linear_attn_norm_group_size and type
- Extend MoE router config with group routing params
- Add MTP variant and loss scaling config

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Implement LinearAttention Core (Part 1: Slope Tensor)

**Files:**
- Create: `megatron/core/transformer/linear_attention.py`
- Test: `tests/unit_tests/transformer/test_linear_attention.py`

**Step 1: Write the failing test for slope tensor**

```python
# tests/unit_tests/transformer/test_linear_attention.py
import pytest
import torch
import math
from megatron.core.transformer.linear_attention import LinearAttention


def test_build_slope_tensor_power_of_2():
    """Test slope tensor for power-of-2 head counts"""
    num_heads = 16
    slopes = LinearAttention._build_slope_tensor(num_heads)

    assert slopes.shape == (num_heads,)
    assert slopes.dtype == torch.float
    # Slopes should be positive and decreasing
    assert torch.all(slopes > 0)
    assert torch.all(slopes[:-1] >= slopes[1:])


def test_build_slope_tensor_non_power_of_2():
    """Test slope tensor for non-power-of-2 head counts"""
    num_heads = 12
    slopes = LinearAttention._build_slope_tensor(num_heads)

    assert slopes.shape == (num_heads,)
    assert torch.all(slopes > 0)


def test_slope_layer_dependency():
    """Test that slope varies by layer index"""
    # This will test the full __init__ later
    pass  # Placeholder for now
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit_tests/transformer/test_linear_attention.py::test_build_slope_tensor_power_of_2 -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement slope tensor builder**

```python
# megatron/core/transformer/linear_attention.py
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Optional, Tuple

import torch

from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.process_groups_config import ProcessGroupCollection


class LinearAttention(Attention):
    """
    Gated Linear Attention (GLA) implementation based on Lightning Attention-2.

    Uses flash-linear-attention library for optimized computation.
    Reference: https://arxiv.org/abs/2401.04658
    """

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int) -> torch.Tensor:
        """
        Build slope tensor for Lightning Attention-2.

        Slopes control the decay rate of attention scores. The computation
        is optimized for power-of-2 head counts but supports arbitrary counts.

        Args:
            n_attention_heads: Number of attention heads

        Returns:
            Tensor of shape [n_attention_heads] with slope values

        Reference:
            https://github.com/OpenNLPLab/lightning-attention/blob/main/lightning_attn/utils/utils.py
        """
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                # For non-power-of-2, use workaround
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                # Get slopes from closest power of 2
                slopes_1 = get_slopes_power_of_2(closest_power_of_2)
                # Get additional slopes from next power of 2, taking even indices
                slopes_2_all = get_slopes_power_of_2(2 * closest_power_of_2)
                slopes_2 = slopes_2_all[0::2][:n - closest_power_of_2]
                return torch.cat([slopes_1, slopes_2])

        slopes = torch.tensor(get_slopes(n_attention_heads), dtype=torch.float)
        return slopes
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit_tests/transformer/test_linear_attention.py::test_build_slope_tensor_power_of_2 -v`
Expected: PASS (2 tests)

**Step 5: Commit slope tensor implementation**

```bash
git add megatron/core/transformer/linear_attention.py tests/unit_tests/transformer/test_linear_attention.py
git commit -m "feat: add LinearAttention with slope tensor builder

- Implement _build_slope_tensor for Lightning Attention-2
- Support both power-of-2 and arbitrary head counts
- Add comprehensive unit tests

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Implement LinearAttention Core (Part 2: Initialization)

**Step 1: Write the failing test for initialization**

Add to test file:

```python
def test_linear_attention_initialization():
    """Test LinearAttention module initialization"""
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.attention import SelfAttentionSubmodules
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    config = TransformerConfig(
        num_layers=20,
        hidden_size=2048,
        num_attention_heads=16,
        kv_channels=128,
        layer_group_size=5,
        linear_attn_norm_group_size=4,
    )

    submodules = SelfAttentionSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=None,  # Not used in LinearAttention
        linear_proj=RowParallelLinear,
    )

    layer_idx = 2  # Not a standard attention layer
    attn = LinearAttention(
        config=config,
        submodules=submodules,
        layer_number=layer_idx,
        attn_mask_type=AttnMaskType.causal,
    )

    # Check slope is layer-dependent
    assert hasattr(attn, 'slope')
    assert attn.slope.shape == (config.num_attention_heads,)

    # Check GroupRMSNorm exists
    assert hasattr(attn, 'g_norm')

    # Check gating projection exists
    assert hasattr(attn, 'g_proj')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit_tests/transformer/test_linear_attention.py::test_linear_attention_initialization -v`
Expected: FAIL with "__init__() missing required positional argument"

**Step 3: Implement __init__ method**

Add to `LinearAttention` class:

```python
def __init__(
    self,
    config: TransformerConfig,
    submodules,
    layer_number: int,
    attn_mask_type: AttnMaskType,
    attention_type: str = "self",
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    super().__init__(
        config=config,
        submodules=submodules,
        layer_number=layer_number,
        attn_mask_type=attn_mask_type,
        attention_type=attention_type,
        pg_collection=pg_collection,
    )

    # Import GroupRMSNorm
    from megatron.core.transformer.group_rms_norm import GroupRMSNorm
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    # Gating projection
    self.g_proj = ColumnParallelLinear(
        self.config.hidden_size,
        self.config.num_attention_heads * self.config.kv_channels,
        config=self.config,
        init_method=self.config.init_method,
        bias=False,
        gather_output=False,
        skip_bias_add=False,
    )

    # Group RMS Norm
    self.g_norm = GroupRMSNorm(
        hidden_size=self.config.num_attention_heads * self.config.kv_channels,
        group_norm_size=self.config.linear_attn_norm_group_size,
        eps=self.config.layernorm_epsilon,
    )

    # Compute layer-dependent slope
    base_slope = self._build_slope_tensor(self.config.num_attention_heads)
    layer_factor = 1 - (self.layer_number - 1) / max(self.config.num_layers - 1, 1) + 1e-5
    slope = -base_slope * layer_factor
    self.register_buffer('slope', slope, persistent=False)

    # Import GLA operators
    try:
        from fla.ops.simple_gla.chunk import chunk_simple_gla
        from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
        self.gla_ops = {
            'chunk': chunk_simple_gla,
            'fused_recurrent': fused_recurrent_simple_gla,
        }
    except ImportError:
        raise ImportError(
            "flash-linear-attention library is required for LinearAttention. "
            "Install from: https://github.com/fla-org/flash-linear-attention"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit_tests/transformer/test_linear_attention.py::test_linear_attention_initialization -v`
Expected: PASS

**Step 5: Commit initialization**

```bash
git add megatron/core/transformer/linear_attention.py tests/unit_tests/transformer/test_linear_attention.py
git commit -m "feat: implement LinearAttention initialization

- Add __init__ with layer-dependent slope computation
- Initialize GroupRMSNorm and gating projection
- Import flash-linear-attention operators
- Add initialization test

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Phase 2: LinearAttention Forward Pass

### Task 5: Implement LinearAttention Forward (Part 1: QKV Processing)

**Step 1: Write failing test for forward pass structure**

Add to test file:

```python
def test_linear_attention_forward_shape():
    """Test LinearAttention forward pass output shape"""
    # This test will be expanded as we implement forward
    # For now, just test that forward exists and returns correct shape
    pass  # Will implement after forward method exists
```

**Step 2: Implement forward method skeleton**

Add to `LinearAttention` class:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    key_value_states: Optional[torch.Tensor] = None,
    inference_params=None,
    rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    packed_seq_params=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward pass for Gated Linear Attention.

    Args:
        hidden_states: Input tensor [batch, seq, hidden]
        attention_mask: 2D mask [batch, seq] for padding (NOT 4D causal mask)
        key_value_states: Not used (for compatibility)
        inference_params: Inference parameters for KV caching
        rotary_pos_emb: Tuple of (cos, sin) for RoPE
        packed_seq_params: Not used

    Returns:
        Tuple of (output_tensor, None)
    """
    # Validate attention mask format
    if attention_mask is not None:
        assert attention_mask.dim() == 2, (
            "LinearAttention only supports 2D attention_mask [batch, seq] "
            "for padding. Arbitrary 4D causal masks are not supported."
        )

    batch_size, seq_len, _ = hidden_states.size()

    # Select GLA mode based on sequence length
    mode = 'fused_recurrent' if seq_len <= 64 else 'chunk'

    # QKV projection
    mixed_qkv, _ = self.linear_qkv(hidden_states)

    # Split into Q, K, V
    # Shape: [batch, seq, (num_heads + 2*num_kv_heads) * head_dim]
    # TODO: Implement split logic

    # Apply QK norm if enabled
    # TODO: Implement

    # Apply RoPE
    # TODO: Implement with unsqueeze_dim=2

    # Handle GQA (expand K, V if needed)
    # TODO: Implement

    # Call GLA kernel
    # TODO: Implement

    # GroupRMSNorm + Gating
    # TODO: Implement

    # Output projection
    # TODO: Implement

    # Placeholder return
    return hidden_states, None
```

**Step 3: Commit forward skeleton**

```bash
git add megatron/core/transformer/linear_attention.py
git commit -m "feat: add LinearAttention forward method skeleton

- Define forward signature matching Attention interface
- Add attention mask validation
- Add mode selection logic
- Add TODO markers for implementation steps

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

**Note:** The implementation plan continues with detailed steps for:
- Task 6-10: Complete LinearAttention forward pass implementation
- Task 11-15: Layer spec and GPTModel integration
- Task 16-20: MTP extension
- Task 21-25: Command-line arguments and end-to-end testing

Due to length constraints, I'll save this first part and indicate that the full plan continues.

---

## Remaining Tasks (Summary)

### Phase 2 Continued: LinearAttention Forward
- Task 6: QKV split and QK norm
- Task 7: RoPE application
- Task 8: GLA kernel invocation
- Task 9: GroupRMSNorm and gating
- Task 10: Output projection and testing

### Phase 3: Integration
- Task 11: Create layer spec file
- Task 12: Implement dynamic attention selection
- Task 13: Modify GPTModel for mask handling
- Task 14: Add MTP block integration
- Task 15: Integration testing

### Phase 4: Configuration and CLI
- Task 16: Add command-line arguments
- Task 17: Wire up config to model
- Task 18: Test with training script
- Task 19: End-to-end validation
- Task 20: Documentation

---

**Implementation Status:** Phase 1 tasks defined in detail
**Next:** Continue with Phase 2-4 detailed task breakdown
