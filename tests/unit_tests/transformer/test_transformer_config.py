# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


def test_linear_attention_config_parameters():
    """Test that linear attention config parameters are accessible"""
    config = TransformerConfig(
        num_layers=20,
        hidden_size=2048,
        num_attention_heads=16,
        use_linear_attention=True,
        layer_group_size=5,
        linear_attn_norm_group_size=4,
        linear_attn_norm_group_type="group_diff",
    )

    assert config.use_linear_attention is True
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
