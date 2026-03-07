# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import math
from megatron.core.transformer.linear_attention import LinearAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from tests.unit_tests.test_utilities import Utils


class TestLinearAttentionSlopeTensor:
    """Test suite for LinearAttention slope tensor builder"""

    def test_build_slope_tensor_power_of_2(self):
        """Test slope tensor for power-of-2 head counts"""
        num_heads = 16
        slopes = LinearAttention._build_slope_tensor(num_heads)

        assert slopes.shape == (num_heads,)
        assert slopes.dtype == torch.float
        # Slopes should be positive
        assert torch.all(slopes > 0)
        # For power-of-2, slopes should be decreasing
        assert torch.all(slopes[:-1] >= slopes[1:])

    def test_build_slope_tensor_non_power_of_2(self):
        """Test slope tensor for non-power-of-2 head counts"""
        num_heads = 12
        slopes = LinearAttention._build_slope_tensor(num_heads)

        assert slopes.shape == (num_heads,)
        assert slopes.dtype == torch.float
        assert torch.all(slopes > 0)
        # For non-power-of-2, slopes are interleaved from two sequences
        # so they may not be strictly monotonic, but should provide diversity

    def test_build_slope_tensor_small_heads(self):
        """Test slope tensor for small head counts"""
        for num_heads in [1, 2, 4, 8]:
            slopes = LinearAttention._build_slope_tensor(num_heads)
            assert slopes.shape == (num_heads,)
            assert torch.all(slopes > 0)

    def test_build_slope_tensor_large_heads(self):
        """Test slope tensor for large head counts"""
        num_heads = 64
        slopes = LinearAttention._build_slope_tensor(num_heads)

        assert slopes.shape == (num_heads,)
        assert torch.all(slopes > 0)
        # For power-of-2, should be decreasing
        assert torch.all(slopes[:-1] >= slopes[1:])

    def test_slope_tensor_values_power_of_2(self):
        """Test that slope values match expected computation for power-of-2"""
        num_heads = 8
        slopes = LinearAttention._build_slope_tensor(num_heads)

        # For power-of-2, verify the geometric sequence
        # start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        n = num_heads
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start

        expected = torch.tensor([start * (ratio ** i) for i in range(num_heads)])

        # Check if values are close (allowing for floating point precision)
        assert torch.allclose(slopes, expected, rtol=1e-5)

    def test_slope_tensor_values_non_power_of_2(self):
        """Test numerical correctness of slope tensor for non-power-of-2 head counts"""
        num_heads = 12
        slopes = LinearAttention._build_slope_tensor(num_heads)

        # For non-power-of-2, verify the concatenation logic
        # closest_power_of_2 = 8, need 4 more elements
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        assert closest_power_of_2 == 8

        # Build expected slopes manually
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return torch.tensor([start * (ratio ** i) for i in range(n)], dtype=torch.float32)

        # First 8 slopes from power-of-2 sequence
        slopes_1 = get_slopes_power_of_2(8)

        # Next 4 slopes from power-of-2(16), taking even indices [0::2]
        slopes_2_all = get_slopes_power_of_2(16)
        slopes_2 = slopes_2_all[0::2][:4]  # Take indices 0, 2, 4, 6

        expected = torch.cat([slopes_1, slopes_2])

        # Verify shape
        assert slopes.shape == (12,)
        assert expected.shape == (12,)

        # Verify numerical correctness
        assert torch.allclose(slopes, expected, rtol=1e-5), \
            f"Slope mismatch:\nGot: {slopes}\nExpected: {expected}"

    def test_slope_layer_dependency(self):
        """Test that slope varies by layer index"""
        # This will test the full __init__ later
        pass  # Placeholder for now


class TestLinearAttentionInitialization:
    """Test suite for LinearAttention initialization"""

    def setup_method(self, method):
        """Setup test environment"""
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        """Cleanup test environment"""
        Utils.destroy_model_parallel()

    def test_linear_attention_initialization(self):
        """Test LinearAttention initialization with layer-dependent slope"""
        # Configuration
        num_layers = 24
        num_heads = 16
        hidden_size = 1024
        kv_channels = 64

        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            kv_channels=kv_channels,
            linear_attn_norm_group_size=4,
            use_cpu_initialization=True,
        )

        # Create submodules (minimal for testing)
        submodules = SelfAttentionSubmodules(
            linear_qkv=None,
            core_attention=None,
            linear_proj=None,
        )

        # Test initialization for different layers
        for layer_idx in [1, 12, 24]:
            layer = LinearAttention(
                config=config,
                submodules=submodules,
                layer_number=layer_idx,
                attn_mask_type=AttnMaskType.causal,
            )

            # Check slope tensor exists and has correct shape
            assert hasattr(layer, 'slope'), f"Layer {layer_idx} missing slope tensor"
            assert layer.slope.shape == (num_heads,), \
                f"Layer {layer_idx} slope shape mismatch: {layer.slope.shape}"

            # Check g_norm exists
            assert hasattr(layer, 'g_norm'), f"Layer {layer_idx} missing g_norm"
            assert layer.g_norm is not None, f"Layer {layer_idx} g_norm is None"

            # Check g_proj exists
            assert hasattr(layer, 'g_proj'), f"Layer {layer_idx} missing g_proj"
            assert layer.g_proj is not None, f"Layer {layer_idx} g_proj is None"

            # Verify slope values are layer-dependent
            # slope = -base_slope * (1 - (layer_idx - 1) / (num_layers - 1) + 1e-5)
            base_slope = LinearAttention._build_slope_tensor(num_heads)
            expected_slope = -base_slope * (1 - (layer_idx - 1) / (num_layers - 1) + 1e-5)

            assert torch.allclose(layer.slope, expected_slope, rtol=1e-5), \
                f"Layer {layer_idx} slope values don't match expected"

    def test_linear_attention_gla_operators_import(self):
        """Test that GLA operators are imported correctly"""
        # This test verifies the import mechanism
        # We'll check if the module has the operators after initialization
        config = TransformerConfig(
            num_layers=12,
            hidden_size=512,
            num_attention_heads=8,
            kv_channels=64,
            use_cpu_initialization=True,
        )

        submodules = SelfAttentionSubmodules(
            linear_qkv=None,
            core_attention=None,
            linear_proj=None,
        )

        layer = LinearAttention(
            config=config,
            submodules=submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        # Check that the layer has references to GLA operators
        # (they should be imported in __init__)
        assert hasattr(layer, 'gla_ops'), "Missing gla_ops dict"
        assert 'chunk' in layer.gla_ops, "Missing 'chunk' key in gla_ops"
        assert 'fused_recurrent' in layer.gla_ops, "Missing 'fused_recurrent' key in gla_ops"


class TestLinearAttentionForward:
    """Test suite for LinearAttention forward pass"""

    def setup_method(self, method):
        """Setup test environment"""
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        """Cleanup test environment"""
        Utils.destroy_model_parallel()

    def test_linear_attention_forward_shape(self):
        """Test LinearAttention forward pass returns correct shape (placeholder)"""
        # This is a placeholder test for the forward method skeleton
        # Full implementation will be tested in later tasks
        config = TransformerConfig(
            num_layers=12,
            hidden_size=512,
            num_attention_heads=8,
            kv_channels=64,
            use_cpu_initialization=True,
        )

        submodules = SelfAttentionSubmodules(
            linear_qkv=None,
            core_attention=None,
            linear_proj=None,
        )

        layer = LinearAttention(
            config=config,
            submodules=submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        # Test that forward method exists and has correct signature
        assert hasattr(layer, 'forward'), "Missing forward method"

        # TODO: Add full forward pass test once implementation is complete

    def test_linear_attention_forward_mask_validation(self):
        """Test that forward pass validates attention mask correctly"""
        # This test will verify that 4D causal masks are rejected
        # Placeholder for now - will be implemented with forward method
        pass

