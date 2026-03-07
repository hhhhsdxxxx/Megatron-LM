# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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


def test_group_rms_norm_numerical_correctness():
    """Test numerical correctness against manual computation"""
    hidden_size = 8
    group_norm_size = 2
    batch_size, seq_len = 2, 4

    norm = GroupRMSNorm(hidden_size=hidden_size, group_norm_size=group_norm_size, eps=1e-6)

    # Set weight to ones for easier verification
    with torch.no_grad():
        norm.weight.fill_(1.0)

    # Create simple input
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    output = norm(input_tensor)

    # Manual computation
    # Reshape to [batch_size, seq_len, group_norm_size, hidden_size // group_norm_size]
    reshaped = input_tensor.view(batch_size, seq_len, group_norm_size, hidden_size // group_norm_size)

    # Compute RMS per group (last dimension)
    variance = reshaped.float().pow(2).mean(dim=-1, keepdim=True)
    manual_output = reshaped.float() * torch.rsqrt(variance + 1e-6)
    manual_output = manual_output.view(batch_size, seq_len, hidden_size).type_as(input_tensor)

    assert torch.allclose(output, manual_output, rtol=1e-5, atol=1e-7)
