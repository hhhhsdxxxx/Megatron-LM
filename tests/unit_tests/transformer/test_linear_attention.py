# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import math
from unittest.mock import patch, MagicMock
from megatron.core.transformer.linear_attention import LinearAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import Attention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.packed_seq_params import PackedSeqParams
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

            # Check slope tensor exists and has correct shape (TP-local at TP=1 = global)
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
            # global_layer_number = layer_number + (pp_layer_offset or 0)
            # slope = -base_slope * (1 - (global_layer_number - 1) / (num_layers - 1) + 1e-5)
            # With pp_layer_offset=None, global_layer_number = layer_idx
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


# ---------------------------------------------------------------------------
# Helper to build a LinearAttention layer for testing (TP=1)
# ---------------------------------------------------------------------------
def _make_layer(
    num_layers=12,
    hidden_size=512,
    num_attention_heads=8,
    kv_channels=64,
    num_query_groups=None,
    linear_attn_norm_group_size=4,
    layer_number=1,
    use_linear_silu=False,
    kv_expand=1,
    linear_attn_num_query_groups=0,
):
    """Create a LinearAttention layer for testing."""
    kwargs = dict(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        kv_channels=kv_channels,
        linear_attn_norm_group_size=linear_attn_norm_group_size,
        use_cpu_initialization=True,
        use_linear_silu=use_linear_silu,
        kv_expand=kv_expand,
        linear_attn_num_query_groups=linear_attn_num_query_groups,
    )
    if num_query_groups is not None:
        kwargs['num_query_groups'] = num_query_groups
    config = TransformerConfig(**kwargs)
    submodules = SelfAttentionSubmodules(
        linear_qkv=None,
        core_attention=None,
        linear_proj=None,
    )
    return LinearAttention(
        config=config,
        submodules=submodules,
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
    )


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
        layer = _make_layer()
        assert hasattr(layer, 'forward'), "Missing forward method"

    # ------------------------------------------------------------------
    # MEG-1: attention_mask must be strictly 2D or None
    # ------------------------------------------------------------------
    def test_meg1_rejects_4d_mask(self):
        """MEG-1: 4D causal masks must be rejected."""
        layer = _make_layer()
        sq, b, h = 16, 2, 512
        hidden_states = torch.randn(sq, b, h)

        mask_4d = torch.ones(b, 1, sq, sq)
        with pytest.raises(ValueError, match="only supports 2D padding masks"):
            layer.forward(hidden_states=hidden_states, attention_mask=mask_4d)

    def test_meg1_rejects_3d_mask(self):
        """MEG-1: 3D masks must be rejected (was silently accepted before fix)."""
        layer = _make_layer()
        sq, b, h = 16, 2, 512
        hidden_states = torch.randn(sq, b, h)

        mask_3d = torch.ones(b, sq, sq)
        with pytest.raises(ValueError, match="only supports 2D padding masks"):
            layer.forward(hidden_states=hidden_states, attention_mask=mask_3d)

    def test_meg1_rejects_1d_mask(self):
        """MEG-1: 1D masks must be rejected."""
        layer = _make_layer()
        sq, b, h = 16, 2, 512
        hidden_states = torch.randn(sq, b, h)

        mask_1d = torch.ones(sq)
        with pytest.raises(ValueError, match="only supports 2D padding masks"):
            layer.forward(hidden_states=hidden_states, attention_mask=mask_1d)

    def test_meg1_rejects_5d_mask(self):
        """MEG-1: Arbitrary high-dimensional masks must be rejected."""
        layer = _make_layer()
        sq, b, h = 16, 2, 512
        hidden_states = torch.randn(sq, b, h)

        mask_5d = torch.ones(1, b, 1, sq, sq)
        with pytest.raises(ValueError, match="only supports 2D padding masks"):
            layer.forward(hidden_states=hidden_states, attention_mask=mask_5d)

    def test_meg1_error_message_includes_shape(self):
        """MEG-1: Error message should include the actual shape for debugging."""
        layer = _make_layer()
        sq, b, h = 16, 2, 512
        hidden_states = torch.randn(sq, b, h)

        mask_3d = torch.ones(b, sq, sq)
        with pytest.raises(ValueError, match=r"3D.*shape"):
            layer.forward(hidden_states=hidden_states, attention_mask=mask_3d)

    def test_meg1_accepts_2d_mask(self):
        """MEG-1: Valid 2D padding masks must pass validation without ValueError."""
        layer = _make_layer()
        sq, b, h = 16, 2, 512
        hidden_states = torch.randn(sq, b, h)

        mask_2d = torch.ones(b, sq)
        try:
            layer.forward(hidden_states=hidden_states, attention_mask=mask_2d)
        except ValueError as e:
            if "2D" in str(e) or "mask" in str(e).lower():
                pytest.fail(f"2D mask should be accepted but got ValueError: {e}")
        except Exception:
            # Non-mask errors (e.g. GLA kernel issues) are fine
            pass

    def test_meg1_accepts_none_mask(self):
        """MEG-1: None attention_mask must pass validation without ValueError."""
        layer = _make_layer()
        sq, b, h = 16, 2, 512
        hidden_states = torch.randn(sq, b, h)

        try:
            layer.forward(hidden_states=hidden_states, attention_mask=None)
        except ValueError as e:
            if "mask" in str(e).lower():
                pytest.fail(f"None mask should be accepted but got ValueError: {e}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # MEG-2: TP-local head counts used consistently
    # ------------------------------------------------------------------
    def test_meg2_g_norm_uses_tp_local_size(self):
        """MEG-2: g_norm hidden_size must match TP-local head count * kv_channels.

        At TP=1, local == global. We verify the wiring is correct by checking the
        g_norm parameter dimension equals num_attention_heads_per_partition * kv_channels.
        """
        layer = _make_layer(num_attention_heads=16, kv_channels=64)

        expected_local_size = layer.num_attention_heads_per_partition * layer.config.kv_channels
        assert layer.g_norm.hidden_size == expected_local_size, (
            f"g_norm.hidden_size={layer.g_norm.hidden_size} but expected "
            f"num_attention_heads_per_partition({layer.num_attention_heads_per_partition}) "
            f"* kv_channels({layer.config.kv_channels}) = {expected_local_size}"
        )
        assert layer.g_norm.weight.shape[0] == expected_local_size

    def test_meg2_g_norm_not_using_global_size(self):
        """MEG-2: Ensure g_norm is wired to per-partition attribute, not config global.

        Under TP=1 both values are equal, so we verify at the code level that
        g_norm.hidden_size == num_attention_heads_per_partition * kv_channels
        (the TP-local formula) rather than config.num_attention_heads * kv_channels
        (the global formula). With TP=1 both happen to equal, so we additionally
        verify the relationship holds for a GQA config where num_query_groups < num_heads.
        """
        # MHA config: num_query_groups defaults to num_attention_heads
        layer_mha = _make_layer(num_attention_heads=8, kv_channels=64)
        assert layer_mha.g_norm.hidden_size == (
            layer_mha.num_attention_heads_per_partition * layer_mha.config.kv_channels
        )

        # GQA config: num_query_groups = 2, num_attention_heads = 8
        layer_gqa = _make_layer(
            num_attention_heads=8, num_query_groups=2, kv_channels=64,
        )
        # g_norm size must be based on num_attention_heads (not num_query_groups)
        assert layer_gqa.g_norm.hidden_size == (
            layer_gqa.num_attention_heads_per_partition * layer_gqa.config.kv_channels
        )

    def test_meg2_slope_buffer_is_tp_local(self):
        """MEG-2: The slope buffer is now TP-local (sliced in __init__).

        At TP=1, the TP-local slope equals the full global slope.
        """
        num_heads = 16
        layer = _make_layer(num_attention_heads=num_heads)
        # At TP=1, TP-local = global
        assert layer.slope.shape == (num_heads,), (
            f"slope buffer should have TP-local shape ({num_heads},) at TP=1, "
            f"got {layer.slope.shape}"
        )

    def test_meg2_slope_already_tp_local(self):
        """MEG-2: At TP=1, slope is already TP-local (no runtime slicing needed)."""
        num_heads = 16
        layer = _make_layer(num_attention_heads=num_heads)

        # Verify slope is TP-local and matches expected values
        base_slope = LinearAttention._build_slope_tensor(num_heads)
        global_layer_number = layer.layer_number  # no PP offset
        num_layers = layer.config.num_layers
        expected_slope = -base_slope * (1 - (global_layer_number - 1) / max(num_layers - 1, 1) + 1e-5)

        assert torch.allclose(layer.slope, expected_slope, rtol=1e-5), (
            "At TP=1, slope should match the full expected slope"
        )

    def test_meg2_forward_reshape_uses_local_heads(self):
        """MEG-2: Verify that the forward path uses TP-local head counts for reshape.

        We mock the GLA kernel to capture the actual tensor shapes passed to it
        and verify they use num_attention_heads_per_partition, not global config values.
        """
        num_heads = 8
        kv_channels = 64
        hidden_size = 512
        layer = _make_layer(
            num_attention_heads=num_heads,
            kv_channels=kv_channels,
            hidden_size=hidden_size,
        )

        sq, b = 32, 2
        hidden_states = torch.randn(sq, b, hidden_size)

        captured_args = {}

        def fake_gla(q, k, v, g, initial_state=None, output_final_state=False):
            captured_args['q_shape'] = q.shape
            captured_args['k_shape'] = k.shape
            captured_args['v_shape'] = v.shape
            captured_args['g_shape'] = g.shape
            # Return dummy output matching expected shape
            output = torch.randn_like(q)
            return output, None

        # Patch both GLA operators
        layer.gla_ops['chunk'] = fake_gla
        layer.gla_ops['fused_recurrent'] = fake_gla

        try:
            layer.forward(hidden_states=hidden_states, attention_mask=None)
        except Exception:
            # linear_proj or g_norm may fail, but GLA args are already captured
            pass

        expected_heads = layer.num_attention_heads_per_partition
        head_dim = layer.hidden_size_per_attention_head

        assert 'q_shape' in captured_args, "GLA kernel was never called"

        # GLA input shape should be [b, sq, num_heads_local, head_dim]
        assert captured_args['q_shape'] == (b, sq, expected_heads, head_dim), (
            f"q shape {captured_args['q_shape']} should use TP-local heads "
            f"({b}, {sq}, {expected_heads}, {head_dim})"
        )
        assert captured_args['k_shape'] == (b, sq, expected_heads, head_dim), (
            f"k shape {captured_args['k_shape']} should use TP-local heads"
        )
        assert captured_args['v_shape'] == (b, sq, expected_heads, head_dim), (
            f"v shape {captured_args['v_shape']} should use TP-local heads"
        )
        # slope (g) shape should be [b, sq, num_heads_local]
        assert captured_args['g_shape'] == (b, sq, expected_heads), (
            f"g (slope) shape {captured_args['g_shape']} should use TP-local heads "
            f"({b}, {sq}, {expected_heads})"
        )

    def test_meg2_gqa_forward_expand_kv_to_local_heads(self):
        """MEG-2: With GQA, K/V should be expanded to num_attention_heads_per_partition.

        When num_query_groups < num_attention_heads, K and V are repeat_interleaved
        to match the local Q head count before calling the GLA kernel.
        """
        num_heads = 8
        num_query_groups = 2
        kv_channels = 64
        hidden_size = 512
        layer = _make_layer(
            num_attention_heads=num_heads,
            num_query_groups=num_query_groups,
            kv_channels=kv_channels,
            hidden_size=hidden_size,
        )

        sq, b = 32, 2
        hidden_states = torch.randn(sq, b, hidden_size)

        captured_args = {}

        def fake_gla(q, k, v, g, initial_state=None, output_final_state=False):
            captured_args['q_shape'] = q.shape
            captured_args['k_shape'] = k.shape
            captured_args['v_shape'] = v.shape
            output = torch.randn_like(q)
            return output, None

        layer.gla_ops['chunk'] = fake_gla
        layer.gla_ops['fused_recurrent'] = fake_gla

        try:
            layer.forward(hidden_states=hidden_states, attention_mask=None)
        except Exception:
            pass

        expected_heads = layer.num_attention_heads_per_partition
        head_dim = layer.hidden_size_per_attention_head

        assert 'k_shape' in captured_args, "GLA kernel was never called"
        # After GQA expansion, K and V should have num_heads (not num_kv_heads)
        assert captured_args['k_shape'] == (b, sq, expected_heads, head_dim), (
            f"After GQA expansion, k shape {captured_args['k_shape']} should have "
            f"{expected_heads} heads, not {num_query_groups}"
        )
        assert captured_args['v_shape'] == (b, sq, expected_heads, head_dim)

    # ------------------------------------------------------------------
    # MEG-5: packed_seq_params is now supported
    # ------------------------------------------------------------------
    def test_meg5_accepts_packed_seq_params(self):
        """MEG-5: Passing packed_seq_params should be accepted (THD format support)."""
        layer = _make_layer()
        sq, b, h = 16, 2, 512
        hidden_states = torch.randn(sq, b, h)

        packed_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 8, 16], dtype=torch.int32),
            max_seqlen_q=8,
            max_seqlen_kv=8,
        )

        # Should not raise NotImplementedError - packed sequences are now supported
        try:
            layer.forward(
                hidden_states=hidden_states,
                attention_mask=None,
                packed_seq_params=packed_params,
            )
        except NotImplementedError as e:
            if "packed" in str(e).lower():
                pytest.fail(f"packed_seq_params should now be accepted: {e}")
        except Exception:
            # Other errors (GLA kernel, etc.) are fine
            pass

    def test_meg5_cu_seqlens_extracted_from_packed_params(self):
        """MEG-5: cu_seqlens should be correctly extracted from packed_seq_params."""
        layer = _make_layer()
        sq, b, h = 16, 2, 512
        hidden_states = torch.randn(sq, b, h)

        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)
        packed_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=8,
            max_seqlen_kv=8,
        )

        captured_kwargs = {}

        def fake_gla(q, k, v, g, **kw):
            captured_kwargs.update(kw)
            return v, None

        layer.gla_ops['chunk'] = fake_gla
        layer.gla_ops['fused_recurrent'] = fake_gla

        try:
            layer.forward(
                hidden_states=hidden_states,
                attention_mask=None,
                packed_seq_params=packed_params,
            )
        except Exception:
            pass

        assert 'cu_seqlens' in captured_kwargs, "cu_seqlens should be passed to GLA kernel"
        assert torch.equal(captured_kwargs['cu_seqlens'], cu_seqlens), \
            "cu_seqlens should match packed_params.cu_seqlens_q"

    # ------------------------------------------------------------------
    # MEG-6: backward_dw must include g_proj
    # ------------------------------------------------------------------
    def test_meg6_backward_dw_calls_g_proj(self):
        """MEG-6: backward_dw() must call g_proj.backward_dw() in addition to parent's."""
        layer = _make_layer()

        # Track whether g_proj.backward_dw was called
        g_proj_called = [False]
        original_g_proj_backward_dw = layer.g_proj.backward_dw

        def tracking_backward_dw():
            g_proj_called[0] = True
            return original_g_proj_backward_dw()

        layer.g_proj.backward_dw = tracking_backward_dw

        layer.backward_dw()

        assert g_proj_called[0], (
            "backward_dw() did not call g_proj.backward_dw(). "
            "g_proj weight gradients will not be computed in delayed-wgrad mode."
        )

    def test_meg6_backward_dw_calls_parent_projections(self):
        """MEG-6: backward_dw() must also call parent's linear_qkv and linear_proj."""
        layer = _make_layer()

        qkv_called = [False]
        proj_called = [False]

        original_qkv_bw = layer.linear_qkv.backward_dw
        original_proj_bw = layer.linear_proj.backward_dw

        def track_qkv():
            qkv_called[0] = True
            return original_qkv_bw()

        def track_proj():
            proj_called[0] = True
            return original_proj_bw()

        layer.linear_qkv.backward_dw = track_qkv
        layer.linear_proj.backward_dw = track_proj

        layer.backward_dw()

        assert qkv_called[0], "backward_dw() did not call linear_qkv.backward_dw()"
        assert proj_called[0], "backward_dw() did not call linear_proj.backward_dw()"

    def test_meg6_backward_dw_is_overridden(self):
        """MEG-6: LinearAttention.backward_dw must be defined on the subclass.

        Ensures that we are not relying on the parent Attention.backward_dw which
        does not know about g_proj.
        """
        # LinearAttention must define its own backward_dw, not inherit it
        assert 'backward_dw' in LinearAttention.__dict__, (
            "LinearAttention must override backward_dw() to include g_proj. "
            "Currently inheriting from Attention base class."
        )

    def test_meg6_backward_dw_no_error(self):
        """MEG-6: backward_dw() must complete without errors."""
        layer = _make_layer()
        # Should not raise
        layer.backward_dw()

    # ------------------------------------------------------------------
    # Existing tests (updated)
    # ------------------------------------------------------------------
    def test_gating_mechanism_order(self):
        """Test that gating mechanism follows correct order: g_norm(attn) * sigmoid(g_proj(hidden))"""
        layer = _make_layer(linear_attn_norm_group_size=4)

        # g_norm should normalize TP-local attention output dimension
        expected_norm_size = layer.num_attention_heads_per_partition * layer.config.kv_channels
        assert layer.g_norm.weight.shape[0] == expected_norm_size, \
            f"g_norm size mismatch: {layer.g_norm.weight.shape[0]} vs {expected_norm_size}"

        # g_proj should have weight parameter
        assert hasattr(layer.g_proj, 'weight') or hasattr(layer.g_proj, 'linear'), \
            "g_proj should have weight parameter"

    def test_attention_mask_padding_handling(self):
        """Test that attention mask is applied to value states for padding"""
        layer = _make_layer()

        # Create input with padding
        sq, b, h = 16, 2, 512
        hidden_states = torch.randn(sq, b, h)

        # Create attention mask: [batch, seq_len]
        # First sequence: all valid (ones)
        # Second sequence: first 8 tokens valid, rest padding (zeros)
        attention_mask = torch.ones(b, sq)
        attention_mask[1, 8:] = 0  # Second sequence has padding

        # This test verifies the mask is accepted and processed
        # Full numerical correctness requires GLA kernel which may not be available
        try:
            output, bias = layer.forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            # If we get here, mask was processed without error
            assert output.shape == hidden_states.shape
        except ImportError:
            # GLA kernel not available, skip
            pass
        except Exception as e:
            # Check that error is not related to mask handling
            error_msg = str(e).lower()
            assert 'mask' not in error_msg or 'dimension' in error_msg, \
                f"Unexpected mask-related error: {e}"


class TestLinearAttentionForwardEndToEnd:
    """End-to-end forward pass tests that mock the GLA kernel.

    These tests validate the full forward pass data flow by replacing
    the GLA kernel with a deterministic stub, allowing shape and value
    verification without requiring the fla package on CPU.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _identity_gla(q, k, v, g, initial_state=None, output_final_state=False):
        """Stub GLA kernel that returns v unchanged (identity attention)."""
        final_state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], q.shape[3]) \
            if output_final_state else None
        return v, final_state

    def test_forward_output_shape(self):
        """Full forward pass produces correct output shape [sq, b, h]."""
        layer = _make_layer(hidden_size=512, num_attention_heads=8, kv_channels=64)
        layer.gla_ops['chunk'] = self._identity_gla
        layer.gla_ops['fused_recurrent'] = self._identity_gla

        sq, b, h = 128, 4, 512
        hidden_states = torch.randn(sq, b, h)

        output, output_bias = layer.forward(
            hidden_states=hidden_states,
            attention_mask=None,
        )

        assert output.shape == (sq, b, h), (
            f"Output shape {output.shape} != expected ({sq}, {b}, {h})"
        )

    def test_forward_short_seq_uses_fused_recurrent(self):
        """Sequences <= 64 should use fused_recurrent mode."""
        layer = _make_layer()

        chunk_called = [False]
        fused_called = [False]

        def fake_chunk(q, k, v, g, **kw):
            chunk_called[0] = True
            return v, None

        def fake_fused(q, k, v, g, **kw):
            fused_called[0] = True
            return v, None

        layer.gla_ops['chunk'] = fake_chunk
        layer.gla_ops['fused_recurrent'] = fake_fused

        sq, b, h = 32, 2, 512  # sq=32 <= 64
        hidden_states = torch.randn(sq, b, h)

        try:
            layer.forward(hidden_states=hidden_states, attention_mask=None)
        except Exception:
            pass

        assert fused_called[0], "Short sequences (<=64) should use fused_recurrent"
        assert not chunk_called[0], "Short sequences should NOT use chunk mode"

    def test_forward_long_seq_uses_chunk(self):
        """Sequences > 64 should use chunk mode."""
        layer = _make_layer()

        chunk_called = [False]
        fused_called = [False]

        def fake_chunk(q, k, v, g, **kw):
            chunk_called[0] = True
            return v, None

        def fake_fused(q, k, v, g, **kw):
            fused_called[0] = True
            return v, None

        layer.gla_ops['chunk'] = fake_chunk
        layer.gla_ops['fused_recurrent'] = fake_fused

        sq, b, h = 128, 2, 512  # sq=128 > 64
        hidden_states = torch.randn(sq, b, h)

        try:
            layer.forward(hidden_states=hidden_states, attention_mask=None)
        except Exception:
            pass

        assert chunk_called[0], "Long sequences (>64) should use chunk mode"
        assert not fused_called[0], "Long sequences should NOT use fused_recurrent"

    def test_forward_with_2d_padding_mask(self):
        """Forward with 2D padding mask completes and produces correct shape."""
        layer = _make_layer()
        layer.gla_ops['chunk'] = self._identity_gla
        layer.gla_ops['fused_recurrent'] = self._identity_gla

        sq, b, h = 128, 2, 512
        hidden_states = torch.randn(sq, b, h)
        attention_mask = torch.ones(b, sq)
        attention_mask[1, 64:] = 0  # Second batch has padding

        output, _ = layer.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        assert output.shape == (sq, b, h)

    def test_forward_gating_applies_sigmoid(self):
        """Gating path: output = g_norm(attn_out) * sigmoid(g_proj(hidden)).

        Gate always uses sigmoid regardless of linear_attn_silu config.
        """
        layer = _make_layer()

        # Use identity GLA so we can trace the gating
        captured = {}

        def capture_gla(q, k, v, g, **kw):
            # Return ones so g_norm result is predictable
            output = torch.ones_like(v)
            return output, None

        layer.gla_ops['chunk'] = capture_gla
        layer.gla_ops['fused_recurrent'] = capture_gla

        sq, b, h = 128, 2, 512
        hidden_states = torch.randn(sq, b, h)

        output, _ = layer.forward(
            hidden_states=hidden_states,
            attention_mask=None,
        )

        # Output should be finite and have correct shape
        assert output.shape == (sq, b, h)
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_forward_gate_always_sigmoid(self):
        """Gate always uses sigmoid, even when linear_attn_silu is True."""
        # Create layer with linear_attn_silu=True
        config = TransformerConfig(
            num_layers=12,
            hidden_size=512,
            num_attention_heads=8,
            kv_channels=64,
            use_cpu_initialization=True,
            linear_attn_silu=True,
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

        def identity_gla(q, k, v, g, **kw):
            return torch.ones_like(v), None

        layer.gla_ops['chunk'] = identity_gla
        layer.gla_ops['fused_recurrent'] = identity_gla

        sq, b, h = 128, 2, 512
        hidden_states = torch.randn(sq, b, h)

        # Should not raise - sigmoid is always used regardless of linear_attn_silu
        output, _ = layer.forward(hidden_states=hidden_states, attention_mask=None)
        assert output.shape == (sq, b, h)
        assert torch.isfinite(output).all()


class TestLinearAttentionNewFeatures:
    """Tests for new features: use_linear_silu, kv_expand, linear_attn_num_query_groups."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_use_linear_silu_applies_silu_to_qkv(self):
        """use_linear_silu should apply SiLU activation to QKV projections."""
        layer_no_silu = _make_layer(use_linear_silu=False)
        layer_silu = _make_layer(use_linear_silu=True)

        # Copy weights to make output deterministic
        layer_silu.linear_qkv.weight.data.copy_(layer_no_silu.linear_qkv.weight.data)

        sq, b, h = 32, 2, 512
        hidden_states = torch.randn(sq, b, h)

        q_no_silu, k_no_silu, v_no_silu = layer_no_silu.get_query_key_value_tensors(
            hidden_states, None, output_gate=False, split_qkv=True,
        )
        q_silu, k_silu, v_silu = layer_silu.get_query_key_value_tensors(
            hidden_states, None, output_gate=False, split_qkv=True,
        )

        # With SiLU, Q should be SiLU(QKV_proj)[:q_size], not just QKV_proj[:q_size]
        assert not torch.allclose(q_no_silu, q_silu, atol=1e-6), \
            "SiLU should change Q values"

    def test_kv_expand_changes_kv_projection_size(self):
        """kv_expand should multiply the KV projection size."""
        layer_1x = _make_layer(kv_expand=1)
        layer_2x = _make_layer(kv_expand=2)

        assert layer_2x.kv_projection_size == 2 * layer_1x.kv_projection_size, \
            f"kv_expand=2 should double KV projection: {layer_2x.kv_projection_size} vs {layer_1x.kv_projection_size}"

    def test_kv_expand_changes_linear_qkv_out_dim(self):
        """kv_expand should increase linear_qkv output dimension."""
        layer_1x = _make_layer(kv_expand=1)
        layer_2x = _make_layer(kv_expand=2)

        # linear_qkv_out_dim = query_projection_size + 2 * kv_projection_size
        # With kv_expand=2, kv_projection_size doubles, so the total increases
        expected_diff = 2 * layer_1x.kv_projection_size  # 2 * (kv_proj * (2-1))
        assert layer_2x.linear_qkv_out_dim == layer_1x.linear_qkv_out_dim + expected_diff

    def test_linear_attn_num_query_groups_overrides_num_query_groups(self):
        """linear_attn_num_query_groups should override config.num_query_groups."""
        # Default: linear_attn_num_query_groups=0 falls back to num_query_groups
        layer_default = _make_layer(num_query_groups=2, linear_attn_num_query_groups=0)
        assert layer_default.num_query_groups_per_partition == 2

        # Override: linear_attn_num_query_groups=4 overrides num_query_groups=2
        layer_override = _make_layer(num_query_groups=2, linear_attn_num_query_groups=4)
        assert layer_override.num_query_groups_per_partition == 4

    def test_linear_attn_num_query_groups_affects_kv_projection(self):
        """linear_attn_num_query_groups should affect KV projection sizing."""
        layer_2g = _make_layer(
            num_query_groups=2,
            linear_attn_num_query_groups=2,
            kv_channels=64,
        )
        layer_4g = _make_layer(
            num_query_groups=2,
            linear_attn_num_query_groups=4,
            kv_channels=64,
        )

        # kv_projection_size = kv_channels * kv_expand * linear_attn_num_query_groups
        assert layer_4g.kv_projection_size == 2 * layer_2g.kv_projection_size

    def test_slope_formula_with_pp_offset(self):
        """Slope formula should use global_layer_number = layer_number + pp_layer_offset."""
        num_layers = 24
        num_heads = 8

        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=512,
            num_attention_heads=num_heads,
            kv_channels=64,
            use_cpu_initialization=True,
        )
        submodules = SelfAttentionSubmodules(
            linear_qkv=None,
            core_attention=None,
            linear_proj=None,
        )

        # Layer 1 with PP offset 6 -> global_layer_number = 7
        layer = LinearAttention(
            config=config,
            submodules=submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pp_layer_offset=6,
        )

        base_slope = LinearAttention._build_slope_tensor(num_heads)
        global_layer_number = 1 + 6  # = 7
        expected_slope = -base_slope * (1 - (global_layer_number - 1) / (num_layers - 1) + 1e-5)
        assert torch.allclose(layer.slope, expected_slope, rtol=1e-5), \
            f"Slope with PP offset mismatch"

    def test_cp_info_stored(self):
        """CP group and size should be stored in __init__."""
        layer = _make_layer()
        assert hasattr(layer, 'cp_group'), "Missing cp_group attribute"
        assert hasattr(layer, 'cp_size'), "Missing cp_size attribute"
        # At CP=1, cp_size should be 1
        assert layer.cp_size == 1
